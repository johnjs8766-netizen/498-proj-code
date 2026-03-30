import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torchaudio

from torch.utils.data import DataLoader

from datasets import WatermarkDataset
from models import (
    HybridWatermarker,
    ResidualSpectrogramDetector,
    HybridBlindDetector,
)
from losses import (
    MultiResolutionSTFTLoss,
    WatermarkEnergyLoss,
    CodeOrthogonalityLoss,
    MaskSparsityLoss,
    DeviceConsistencyLoss,
)


def set_seed(seed=42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def compute_confusion_matrix(preds, labels, num_classes):
    cm = torch.zeros(num_classes, num_classes, dtype=torch.long)
    for t, p in zip(labels.view(-1), preds.view(-1)):
        cm[t.long(), p.long()] += 1
    return cm


def print_confusion_matrix(cm, title="Confusion Matrix"):
    print(title)
    for i in range(cm.size(0)):
        row = " ".join([f"{int(v):4d}" for v in cm[i]])
        print(f"  {i}: {row}")


@torch.no_grad()
def save_epoch_audio_samples(generator, sample_audio, epoch, stage_name, device, num_devices, sample_rate=16000):
    generator.eval()

    epoch_dir = os.path.join("audio_samples", f"{stage_name}_epoch_{epoch+1:02d}")
    os.makedirs(epoch_dir, exist_ok=True)

    sample_audio = sample_audio.to(device)

    torchaudio.save(
        os.path.join(epoch_dir, "clean.wav"),
        sample_audio.squeeze(0).cpu(),
        sample_rate
    )

    for user_id in range(num_devices):
        label = torch.tensor([user_id], device=device)
        watermarked_audio, residual_audio, _, _, _ = generator(sample_audio, label)

        torchaudio.save(
            os.path.join(epoch_dir, f"watermarked_user{user_id}.wav"),
            watermarked_audio.squeeze(0).cpu(),
            sample_rate
        )

        torchaudio.save(
            os.path.join(epoch_dir, f"residual_user{user_id}.wav"),
            residual_audio.squeeze(0).cpu(),
            sample_rate
        )

        residual_amp = torch.clamp(residual_audio * 20.0, min=-1.0, max=1.0)
        torchaudio.save(
            os.path.join(epoch_dir, f"residual_x20_user{user_id}.wav"),
            residual_amp.squeeze(0).cpu(),
            sample_rate
        )


@torch.no_grad()
def evaluate_residual(generator, residual_detector, test_loader, device, num_devices):
    generator.eval()
    residual_detector.eval()

    total = 0
    correct = 0
    cm_total = torch.zeros(num_devices, num_devices, dtype=torch.long)

    for raw_audio, _ in test_loader:
        raw_audio = raw_audio.to(device)
        bsz = raw_audio.size(0)
        labels = torch.randint(0, num_devices, (bsz,), device=device)

        _, residual_audio, _, _, _ = generator(raw_audio, labels)
        logits = residual_detector(residual_audio)
        preds = torch.argmax(logits, dim=1)

        correct += (preds == labels).sum().item()
        total += bsz
        cm_total += compute_confusion_matrix(preds.cpu(), labels.cpu(), num_devices)

    acc = 100.0 * correct / max(total, 1)
    print(f"\nResidual Classification Acc: {acc:.2f}%")
    print_confusion_matrix(cm_total, "Residual Confusion Matrix")
    return acc


@torch.no_grad()
def evaluate_watermarked(generator, blind_detector, test_loader, device, num_devices):
    generator.eval()
    blind_detector.eval()

    total = 0
    correct = 0
    cm_total = torch.zeros(num_devices, num_devices, dtype=torch.long)

    for raw_audio, _ in test_loader:
        raw_audio = raw_audio.to(device)
        bsz = raw_audio.size(0)
        labels = torch.randint(0, num_devices, (bsz,), device=device)

        watermarked_audio, _, _, _, _ = generator(raw_audio, labels)
        logits = blind_detector(watermarked_audio)
        preds = torch.argmax(logits, dim=1)

        correct += (preds == labels).sum().item()
        total += bsz
        cm_total += compute_confusion_matrix(preds.cpu(), labels.cpu(), num_devices)

    acc = 100.0 * correct / max(total, 1)
    print(f"\nWatermarked-Audio Classification Acc: {acc:.2f}%")
    print_confusion_matrix(cm_total, "Watermarked-Audio Confusion Matrix")
    return acc


def train():
    set_seed(42)
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}")

    batch_size = 16
    num_devices = 6

    epochs_stage_a = 10
    epochs_stage_b = 12
    epochs_stage_c = 8

    alpha_stft = 0.05
    alpha_wave = 0.02
    code_time_steps = 16

    lr_G = 1e-4
    lr_resD = 2e-4
    lr_blindD = 2e-4
    lr_joint_G = 1e-4
    lr_joint_D = 1e-4

    lambda_det_max = 2.0
    lambda_stft = 1.0
    lambda_time = 2.0
    lambda_energy = 1.0
    lambda_code_orth = 0.5
    lambda_mask_sparse = 0.02
    lambda_consistency = 0.5
    max_gen_ce = 5.0

    lambda_joint_cls = 2.0
    lambda_joint_stft = 0.5
    lambda_joint_time = 1.0
    lambda_joint_energy = 0.5
    max_joint_ce = 3.0

    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("audio_samples", exist_ok=True)
    os.makedirs("user_keys", exist_ok=True)

    train_dataset = WatermarkDataset(
        "dataset_libritts/dataset/metadata.csv",
        target_sr=16000,
        duration=3.0,
        mode="train"
    )
    test_dataset = WatermarkDataset(
        "dataset_libritts/dataset/metadata.csv",
        target_sr=16000,
        duration=3.0,
        mode="test"
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    fixed_sample_audio = None
    for batch_audio, _ in test_loader:
        fixed_sample_audio = batch_audio[0:1].to(device)
        break

    generator = HybridWatermarker(
        n_fft=512,
        hop_length=160,
        win_length=400,
        num_devices=num_devices,
        alpha_stft=alpha_stft,
        alpha_wave=alpha_wave,
        code_time_steps=code_time_steps
    ).to(device)

    residual_detector = ResidualSpectrogramDetector(
        num_devices=num_devices,
        n_fft=512,
        hop_length=160,
        win_length=400
    ).to(device)

    blind_detector = HybridBlindDetector(
        num_devices=num_devices
    ).to(device)

    ce_loss = nn.CrossEntropyLoss()
    l1_loss = nn.L1Loss()
    stft_loss = MultiResolutionSTFTLoss().to(device)
    energy_loss = WatermarkEnergyLoss().to(device)
    code_orth_loss = CodeOrthogonalityLoss().to(device)
    mask_sparse_loss = MaskSparsityLoss().to(device)
    consistency_loss = DeviceConsistencyLoss().to(device)

    opt_G = optim.Adam(generator.parameters(), lr=lr_G, weight_decay=1e-5)
    opt_resD = optim.Adam(residual_detector.parameters(), lr=lr_resD, weight_decay=1e-5)
    opt_blindD = optim.Adam(blind_detector.parameters(), lr=lr_blindD, weight_decay=1e-5)

    # ==================== Stage A ====================
    print("\n================ STAGE A: Train Generator + Residual Detector ================\n")
    for epoch in range(epochs_stage_a):
        generator.train()
        residual_detector.train()

        lambda_det = lambda_det_max * ((epoch + 1) / epochs_stage_a)

        running_loss_g = 0.0
        running_loss_d = 0.0

        for i, (raw_audio, _) in enumerate(train_loader):
            raw_audio = raw_audio.to(device)
            bsz = raw_audio.size(0)
            labels = torch.randint(0, num_devices, (bsz,), device=device)

            # residual detector update
            generator.eval()
            residual_detector.train()
            opt_resD.zero_grad()

            with torch.no_grad():
                _, residual_audio_det, _, _, _ = generator(raw_audio, labels)

            logits_d = residual_detector(residual_audio_det)
            loss_D = ce_loss(logits_d, labels)
            loss_D.backward()
            torch.nn.utils.clip_grad_norm_(residual_detector.parameters(), 1.0)
            opt_resD.step()

            running_loss_d += loss_D.item()

            # generator update
            generator.train()
            residual_detector.eval()
            opt_G.zero_grad()

            watermarked_audio, residual_audio, delta_mag, mask, _ = generator(raw_audio, labels)
            logits_g = residual_detector(residual_audio)

            loss_G_det = torch.clamp(ce_loss(logits_g, labels), max=max_gen_ce)
            loss_G_stft = stft_loss(raw_audio, watermarked_audio)
            loss_G_time = l1_loss(raw_audio, watermarked_audio)
            loss_G_energy = energy_loss(residual_audio)

            normalized_codes = generator.codebook.normalized_all()
            loss_G_code = code_orth_loss(normalized_codes)
            loss_G_mask = mask_sparse_loss(mask)
            loss_G_cons = consistency_loss(delta_mag, labels, num_devices)

            loss_G = (
                lambda_det * loss_G_det
                + lambda_stft * loss_G_stft
                + lambda_time * loss_G_time
                + lambda_energy * loss_G_energy
                + lambda_code_orth * loss_G_code
                + lambda_mask_sparse * loss_G_mask
                + lambda_consistency * loss_G_cons
            )

            loss_G.backward()
            torch.nn.utils.clip_grad_norm_(generator.parameters(), 1.0)
            opt_G.step()

            running_loss_g += loss_G.item()

            if i % 10 == 0:
                print(
                    f"[Stage A][Epoch {epoch+1}/{epochs_stage_a}] Batch {i} | "
                    f"Loss D: {loss_D.item():.4f} | Loss G: {loss_G.item():.4f} | "
                    f"L_det: {loss_G_det.item():.4f} | L_stft: {loss_G_stft.item():.4f} | "
                    f"L_time: {loss_G_time.item():.6f} | L_energy: {loss_G_energy.item():.6f} | "
                    f"L_code: {loss_G_code.item():.6f} | L_mask: {loss_G_mask.item():.4f} | "
                    f"L_cons: {loss_G_cons.item():.4f}"
                )

        print(f"\n[Stage A] Epoch {epoch+1} Summary")
        print(f"Avg Loss D: {running_loss_d / len(train_loader):.4f}")
        print(f"Avg Loss G: {running_loss_g / len(train_loader):.4f}")

        evaluate_residual(generator, residual_detector, test_loader, device, num_devices)
        save_epoch_audio_samples(generator, fixed_sample_audio, epoch, "stageA", device, num_devices)

        torch.save(generator.state_dict(), f"checkpoints/stageA_generator_epoch_{epoch+1}.pth")
        torch.save(residual_detector.state_dict(), f"checkpoints/stageA_residual_detector_epoch_{epoch+1}.pth")

    # ==================== Stage B ====================
    print("\n================ STAGE B: Freeze Generator, Train Blind Detector ================\n")
    for p in generator.parameters():
        p.requires_grad = False
    generator.eval()

    for epoch in range(epochs_stage_b):
        blind_detector.train()
        running_loss = 0.0
        total = 0
        correct = 0

        for i, (raw_audio, _) in enumerate(train_loader):
            raw_audio = raw_audio.to(device)
            bsz = raw_audio.size(0)
            labels = torch.randint(0, num_devices, (bsz,), device=device)

            with torch.no_grad():
                watermarked_audio, _, _, _, _ = generator(raw_audio, labels)

            opt_blindD.zero_grad()
            logits = blind_detector(watermarked_audio)
            loss = ce_loss(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(blind_detector.parameters(), 1.0)
            opt_blindD.step()

            running_loss += loss.item()
            preds = torch.argmax(logits, dim=1)
            correct += (preds == labels).sum().item()
            total += bsz

            if i % 10 == 0:
                print(
                    f"[Stage B][Epoch {epoch+1}/{epochs_stage_b}] Batch {i} | "
                    f"Loss: {loss.item():.4f} | Train Acc: {100.0 * correct / max(total,1):.2f}%"
                )

        print(f"\n[Stage B] Epoch {epoch+1} Summary")
        print(f"Avg Loss: {running_loss / len(train_loader):.4f}")
        print(f"Train Acc: {100.0 * correct / max(total,1):.2f}%")

        evaluate_watermarked(generator, blind_detector, test_loader, device, num_devices)
        save_epoch_audio_samples(generator, fixed_sample_audio, epoch, "stageB", device, num_devices)

        torch.save(blind_detector.state_dict(), f"checkpoints/stageB_blind_detector_epoch_{epoch+1}.pth")

    # ==================== Stage C ====================
    print("\n================ STAGE C: Joint Fine-tuning ================\n")
    for p in generator.parameters():
        p.requires_grad = True

    opt_joint_G = optim.Adam(generator.parameters(), lr=lr_joint_G, weight_decay=1e-5)
    opt_joint_D = optim.Adam(blind_detector.parameters(), lr=lr_joint_D, weight_decay=1e-5)

    for epoch in range(epochs_stage_c):
        generator.train()
        blind_detector.train()

        running_loss_g = 0.0
        running_loss_d = 0.0

        for i, (raw_audio, _) in enumerate(train_loader):
            raw_audio = raw_audio.to(device)
            bsz = raw_audio.size(0)
            labels = torch.randint(0, num_devices, (bsz,), device=device)

            # blind detector update
            generator.eval()
            blind_detector.train()
            opt_joint_D.zero_grad()

            with torch.no_grad():
                watermarked_audio_det, _, _, _, _ = generator(raw_audio, labels)

            logits_d = blind_detector(watermarked_audio_det)
            loss_D = ce_loss(logits_d, labels)
            loss_D.backward()
            torch.nn.utils.clip_grad_norm_(blind_detector.parameters(), 1.0)
            opt_joint_D.step()

            running_loss_d += loss_D.item()

            # generator update
            generator.train()
            blind_detector.eval()
            opt_joint_G.zero_grad()

            watermarked_audio, residual_audio, _, _, _ = generator(raw_audio, labels)
            logits_g = blind_detector(watermarked_audio)

            loss_G_cls = torch.clamp(ce_loss(logits_g, labels), max=max_joint_ce)
            loss_G_stft = stft_loss(raw_audio, watermarked_audio)
            loss_G_time = l1_loss(raw_audio, watermarked_audio)
            loss_G_energy = energy_loss(residual_audio)

            loss_G = (
                lambda_joint_cls * loss_G_cls
                + lambda_joint_stft * loss_G_stft
                + lambda_joint_time * loss_G_time
                + lambda_joint_energy * loss_G_energy
            )

            loss_G.backward()
            torch.nn.utils.clip_grad_norm_(generator.parameters(), 1.0)
            opt_joint_G.step()

            running_loss_g += loss_G.item()

            if i % 10 == 0:
                print(
                    f"[Stage C][Epoch {epoch+1}/{epochs_stage_c}] Batch {i} | "
                    f"Loss D: {loss_D.item():.4f} | Loss G: {loss_G.item():.4f} | "
                    f"L_cls: {loss_G_cls.item():.4f} | L_stft: {loss_G_stft.item():.4f} | "
                    f"L_time: {loss_G_time.item():.6f} | L_energy: {loss_G_energy.item():.6f}"
                )

        print(f"\n[Stage C] Epoch {epoch+1} Summary")
        print(f"Avg Loss D: {running_loss_d / len(train_loader):.4f}")
        print(f"Avg Loss G: {running_loss_g / len(train_loader):.4f}")

        evaluate_watermarked(generator, blind_detector, test_loader, device, num_devices)
        save_epoch_audio_samples(generator, fixed_sample_audio, epoch, "stageC", device, num_devices)

        torch.save(generator.state_dict(), f"checkpoints/stageC_generator_epoch_{epoch+1}.pth")
        torch.save(blind_detector.state_dict(), f"checkpoints/stageC_blind_detector_epoch_{epoch+1}.pth")

        with torch.no_grad():
            codes = generator.codebook.normalized_all().detach().cpu()
            for u in range(num_devices):
                torch.save(codes[u], f"user_keys/user_{u}_tf_codeword.pt")

    print("\nTraining finished.")


if __name__ == "__main__":
    train()
