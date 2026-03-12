import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR 
from torch.utils.data import DataLoader
import torchaudio

from datasets import WatermarkDataset
from models import PluginSEANetGenerator, SOTAAudioDetector
from losses import MultiResolutionSTFTLoss

def train():
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
        
    batch_size = 32  
    epochs = 30
    lr_G, lr_D = 2e-4, 1e-5 
    alpha = 2.0 
    lambda_det = 1.5
    num_devices = 6
    unwatermarked_label = 6 
    warmup_epochs = 3
    key_dim = 128

    # the secure key parameters held by users
    key_vault = nn.Embedding(num_embeddings=num_devices, embedding_dim=key_dim).to(device)
    nn.init.orthogonal_(key_vault.weight)
    key_vault.weight.requires_grad = False 

    # DataLoaders
    train_dataset = WatermarkDataset("dataset_libritts/dataset/metadata.csv", target_sr=16000, duration=3.0, mode='train')
    test_dataset = WatermarkDataset("dataset_libritts/dataset/metadata.csv", target_sr=16000, duration=3.0, mode='test')
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Models
    generator = PluginSEANetGenerator(key_dim=key_dim).to(device)
    detector = SOTAAudioDetector(num_classes=num_devices + 1).to(device)

    # Optimizers, schedulers and losses
    # opt_G = optim.Adam(generator.parameters(), lr=lr_G)
    # opt_D = optim.Adam(detector.parameters(), lr=lr_D) 
    opt_G = optim.Adam(generator.parameters(), lr=lr_G, weight_decay=1e-5)
    opt_D = optim.Adam(detector.parameters(), lr=lr_D, weight_decay=1e-5)
    scheduler_G = StepLR(opt_G, step_size=15, gamma=0.5)
    scheduler_D = StepLR(opt_D, step_size=15, gamma=0.5)

    mr_stft_loss = MultiResolutionSTFTLoss().to(device)
    l1_loss = nn.L1Loss()
    ce_loss = nn.CrossEntropyLoss(label_smoothing=0.1)

    # save checkpoints, watermarked audio samples and secure keys held by users
    os.makedirs("checkpoint", exist_ok=True)
    os.makedirs("audio_samples", exist_ok=True)
    os.makedirs("user_keys", exist_ok=True) 

    # Training Loop
    for epoch in range(epochs):
        imp_weight = min(1.0, epoch / warmup_epochs)
        
        for i, (raw_audio, _) in enumerate(train_loader):
            raw_audio = raw_audio.to(device)
            current_batch_size = raw_audio.size(0)
            target_labels = torch.randint(0, num_devices, (current_batch_size,)).to(device)
            
            # Fetch the raw key vectors for this specific batch
            batch_keys = key_vault(target_labels)

            # train detector
            detector.train()     
            generator.eval()    
            opt_D.zero_grad()
            
            preds_unwatermarked = detector(raw_audio)
            labels_unwatermarked = torch.full((current_batch_size,), unwatermarked_label, dtype=torch.long).to(device)
            loss_D_clean = ce_loss(preds_unwatermarked, labels_unwatermarked)
            
            # Pass the raw keys into the generator instead of labels
            wm = generator(raw_audio, batch_keys) 
            speech_envelope = torch.abs(raw_audio) 
            scaled_wm = wm * speech_envelope
            watermarked_audio = torch.clamp(raw_audio + (alpha * scaled_wm), min=-1.0, max=1.0)
            
            preds_watermarked = detector(watermarked_audio.detach()) 
            loss_D_watermarked = ce_loss(preds_watermarked, target_labels)

            loss_D = loss_D_clean + loss_D_watermarked
            loss_D.backward()
            opt_D.step()

            # train watermark generator
            detector.eval()      
            generator.train()   
            opt_G.zero_grad()
            
            preds_for_G = detector(watermarked_audio) 
            loss_G_det = ce_loss(preds_for_G, target_labels)
            
            loss_G_freq = mr_stft_loss(raw_audio, watermarked_audio)
            loss_G_time = l1_loss(raw_audio, watermarked_audio)
            
            loss_G_imp_raw = loss_G_freq + loss_G_time
            loss_G_imp_weighted = loss_G_imp_raw * imp_weight
            
            loss_G = loss_G_imp_weighted + (lambda_det * loss_G_det) 
            loss_G.backward()
            opt_G.step()

            if i % 10 == 0:
                print(f"Epoch [{epoch}/{epochs}] Batch {i} | Loss D: {loss_D.item():.4f} | Loss G: {loss_G.item():.4f} | L_imp: {loss_G_imp_raw.item():.4f} | w: {imp_weight:.1f}")

        torch.nn.utils.clip_grad_norm_(generator.parameters(), max_norm=1.0)
        torch.nn.utils.clip_grad_norm_(detector.parameters(), max_norm=1.0)
        scheduler_G.step()
        scheduler_D.step()

        # evaluate acc
        generator.eval()
        detector.eval()
        correct_wm, total_wm, correct_clean, total_clean = 0, 0, 0, 0
        
        with torch.no_grad():
            for raw_audio, _ in test_loader:
                raw_audio = raw_audio.to(device)
                current_batch_size = raw_audio.size(0)
                target_labels = torch.randint(0, num_devices, (current_batch_size,)).to(device)
                batch_keys = key_vault(target_labels)
                
                preds_clean = detector(raw_audio)
                correct_clean += (torch.argmax(preds_clean, dim=1) == unwatermarked_label).sum().item()
                total_clean += current_batch_size

                wm = generator(raw_audio, batch_keys)
                speech_envelope = torch.abs(raw_audio)
                scaled_wm = wm * speech_envelope
                watermarked_audio = torch.clamp(raw_audio + (alpha * scaled_wm), min=-1.0, max=1.0)
                
                preds_wm = detector(watermarked_audio)
                correct_wm += (torch.argmax(preds_wm, dim=1) == target_labels).sum().item()
                total_wm += current_batch_size

        print(f"\nEpoch {epoch} Test Acc -> Watermarked: {(correct_wm/total_wm)*100:.2f}% | Clean: {(correct_clean/total_clean)*100:.2f}%\n")

        # save detector and generator ckpts
        torch.save(detector.state_dict(), f"checkpoint/detector_epoch_{epoch}.pth")
        
        torch.save(generator.state_dict(), f"checkpoint/generator_epoch_{epoch}.pth")

        # save distinct user keys for each device
        for u in range(num_devices):
            user_key_tensor = key_vault.weight[u].detach().cpu()
            torch.save(user_key_tensor, f"user_keys/user_{u}_plugin_key.pt")

        with torch.no_grad():
            sample_clean = raw_audio[0:1]
            sample_label = torch.tensor([0]).to(device) 
            sample_key = key_vault(sample_label) # Grab User 0's key
            sample_wm = generator(sample_clean, sample_key)
            sample_watermarked = torch.clamp(sample_clean + (alpha * sample_wm * torch.abs(sample_clean)), min=-1.0, max=1.0)
            
        torchaudio.save(f"audio_samples/epoch_{epoch}_clean.wav", sample_clean.squeeze(0).cpu(), 16000)
        torchaudio.save(f"audio_samples/epoch_{epoch}_watermarked_user0.wav", sample_watermarked.squeeze(0).cpu(), 16000)

if __name__ == "__main__":
    train()