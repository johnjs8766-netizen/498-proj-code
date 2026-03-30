import os
import random
import json
import torch
import torchaudio

from torch.utils.data import DataLoader

from datasets import WatermarkDataset
from models import HybridWatermarker, HybridBlindDetector


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


def snr_db(clean, test, eps=1e-12):
    # clean, test: (B,1,T)
    noise = test - clean
    sig_pow = torch.mean(clean ** 2, dim=(-1, -2))      # (B,)
    noise_pow = torch.mean(noise ** 2, dim=(-1, -2)) + eps
    return 10.0 * torch.log10((sig_pow + eps) / noise_pow)


def lsd_db(clean, test, n_fft=512, hop_length=160, win_length=400, eps=1e-8):
    device = clean.device
    window = torch.hann_window(win_length, device=device)

    clean_spec = torch.stft(
        clean.squeeze(1),
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=window,
        return_complex=True
    )
    test_spec = torch.stft(
        test.squeeze(1),
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=window,
        return_complex=True
    )

    clean_log = 20.0 * torch.log10(torch.abs(clean_spec) + eps)
    test_log = 20.0 * torch.log10(torch.abs(test_spec) + eps)

    diff = clean_log - test_log
    lsd = torch.sqrt(torch.mean(diff ** 2, dim=1))   # avg over freq => (B,T)
    lsd = torch.mean(lsd, dim=1)                     # avg over time => (B,)
    return lsd


def attack_none(x, sr):
    return x


def attack_gaussian_noise(x, sr, sigma=0.005):
    y = x + sigma * torch.randn_like(x)
    return torch.clamp(y, -1.0, 1.0)


def attack_resample_8k(x, sr):
    y = torchaudio.functional.resample(x, sr, 8000)
    y = torchaudio.functional.resample(y, 8000, sr)
    return torch.clamp(y, -1.0, 1.0)


def attack_lowpass_3k(x, sr):
    y = torchaudio.functional.lowpass_biquad(x, sr, cutoff_freq=3000.0)
    return torch.clamp(y, -1.0, 1.0)


def attack_amplitude_scale(x, sr, scale=0.8):
    y = x * scale
    return torch.clamp(y, -1.0, 1.0)


def attack_crop_center_90(x, sr):
    # zero out 10% of the center
    B, C, T = x.shape
    remove = int(T * 0.10)
    left = (T - remove) // 2
    right = left + remove

    y = x.clone()
    y[:, :, left:right] = 0.0
    return y


def attack_mp3_like(x, sr):
    # rough approximation if real mp3 pipeline is unavailable
    y = torchaudio.functional.lowpass_biquad(x, sr, cutoff_freq=3500.0)
    y = torchaudio.functional.resample(y, sr, 12000)
    y = torchaudio.functional.resample(y, 12000, sr)
    return torch.clamp(y, -1.0, 1.0)


ATTACKS = {
    "none": attack_none,
    "gaussian_noise": attack_gaussian_noise,
    "resample_8k": attack_resample_8k,
    "lowpass_3k": attack_lowpass_3k,
    "amplitude_scale": attack_amplitude_scale,
    "crop_center_90": attack_crop_center_90,
    "mp3_like": attack_mp3_like,
}


@torch.no_grad()
def evaluate_attack(
    generator,
    detector,
    loader,
    device,
    num_devices,
    attack_name,
    sample_rate=16000,
    save_examples=False,
    save_dir=None,
    max_save=5,
):
    generator.eval()
    detector.eval()

    attack_fn = ATTACKS[attack_name]

    total = 0
    correct = 0
    cm_total = torch.zeros(num_devices, num_devices, dtype=torch.long)

    snr_vals = []
    lsd_vals = []

    saved = 0

    for batch_idx, (raw_audio, _) in enumerate(loader):
        raw_audio = raw_audio.to(device)
        bsz = raw_audio.size(0)

        labels = torch.randint(0, num_devices, (bsz,), device=device)

        watermarked_audio, _, _, _, _ = generator(raw_audio, labels)
        attacked_audio = attack_fn(watermarked_audio, sample_rate)
        attacked_audio = attacked_audio[:, :, :watermarked_audio.size(-1)]

        logits = detector(attacked_audio)
        preds = torch.argmax(logits, dim=1)

        correct += (preds == labels).sum().item()
        total += bsz
        cm_total += compute_confusion_matrix(preds.cpu(), labels.cpu(), num_devices)

        batch_snr = snr_db(raw_audio, attacked_audio).detach().cpu().view(-1)
        batch_lsd = lsd_db(raw_audio, attacked_audio).detach().cpu().view(-1)

        snr_vals.extend(batch_snr.tolist())
        lsd_vals.extend(batch_lsd.tolist())

        if save_examples and save_dir is not None and saved < max_save:
            for i in range(min(bsz, max_save - saved)):
                subdir = os.path.join(save_dir, attack_name)
                os.makedirs(subdir, exist_ok=True)

                torchaudio.save(
                    os.path.join(subdir, f"sample_{saved}_clean.wav"),
                    raw_audio[i].cpu(),
                    sample_rate
                )
                torchaudio.save(
                    os.path.join(subdir, f"sample_{saved}_watermarked.wav"),
                    watermarked_audio[i].cpu(),
                    sample_rate
                )
                torchaudio.save(
                    os.path.join(subdir, f"sample_{saved}_attacked.wav"),
                    attacked_audio[i].cpu(),
                    sample_rate
                )
                saved += 1

    mean_snr = float(torch.tensor(snr_vals).mean().item()) if len(snr_vals) > 0 else 0.0
    mean_lsd = float(torch.tensor(lsd_vals).mean().item()) if len(lsd_vals) > 0 else 0.0
    acc = 100.0 * correct / max(total, 1)

    return {
        "attack": attack_name,
        "accuracy": acc,
        "snr_db": mean_snr,
        "lsd_db": mean_lsd,
        "confusion_matrix": cm_total.tolist(),
    }


def main():
    set_seed(42)
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Evaluating on: {device}")

    # ------------------------------------------------
    # Config
    # ------------------------------------------------
    batch_size = 16
    num_devices = 6
    sample_rate = 16000

    # IMPORTANT: change to your best checkpoints
    generator_ckpt = "checkpoints/stageC_generator_epoch_8.pth"
    detector_ckpt = "checkpoints/stageC_blind_detector_epoch_8.pth"

    results_dir = "robustness_results"
    audio_examples_dir = os.path.join(results_dir, "audio_examples")
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(audio_examples_dir, exist_ok=True)

    # ------------------------------------------------
    # Data
    # ------------------------------------------------
    test_dataset = WatermarkDataset(
        "dataset_libritts/dataset/metadata.csv",
        target_sr=sample_rate,
        duration=3.0,
        mode="test"
    )
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # ------------------------------------------------
    # Models
    # ------------------------------------------------
    generator = HybridWatermarker(
        n_fft=512,
        hop_length=160,
        win_length=400,
        num_devices=num_devices,
        alpha_stft=0.05,
        alpha_wave=0.02,
        code_time_steps=16
    ).to(device)

    detector = HybridBlindDetector(
        num_devices=num_devices
    ).to(device)

    generator.load_state_dict(torch.load(generator_ckpt, map_location=device))
    detector.load_state_dict(torch.load(detector_ckpt, map_location=device))

    generator.eval()
    detector.eval()

    # ------------------------------------------------
    # Evaluate all attacks
    # ------------------------------------------------
    all_results = []

    for attack_name in ATTACKS.keys():
        print(f"\n=== Evaluating attack: {attack_name} ===")

        result = evaluate_attack(
            generator=generator,
            detector=detector,
            loader=test_loader,
            device=device,
            num_devices=num_devices,
            attack_name=attack_name,
            sample_rate=sample_rate,
            save_examples=True,
            save_dir=audio_examples_dir,
            max_save=3,
        )

        all_results.append(result)

        print(f"Attack: {result['attack']}")
        print(f"  Accuracy: {result['accuracy']:.2f}%")
        print(f"  Mean SNR: {result['snr_db']:.2f} dB")
        print(f"  Mean LSD: {result['lsd_db']:.2f} dB")

        cm = torch.tensor(result["confusion_matrix"])
        print_confusion_matrix(cm, title=f"Confusion Matrix - {attack_name}")

    # ------------------------------------------------
    # Save JSON summary
    # ------------------------------------------------
    summary_path = os.path.join(results_dir, "robustness_summary.json")
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\nSaved results to: {summary_path}")


if __name__ == "__main__":
    main()
