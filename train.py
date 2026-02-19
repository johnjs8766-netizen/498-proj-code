import torch
import torch.optim as optim
import os
from torch.utils.data import DataLoader
from dataset import LibriTTSDataset
from models import CELPWatermarkEncoder, WatermarkDetector
from losses import MultiResolutionSTFTLoss, parameter_reg_loss

def main():
    # --- CONFIG ---
    NUM_DEVICES = 5          
    NUM_CLASSES = 6          
    BATCH_SIZE = 16          
    LR = 1e-3
    EPOCHS = 20

    # --- DEVICE SELECTION ---
    if torch.backends.mps.is_available():
        DEVICE = torch.device("mps")
    elif torch.cuda.is_available():
        DEVICE = torch.device("cuda")
    else:
        DEVICE = torch.device("cpu")

    # --- DATA ---
    DATA_PATH = "./data/LibriTTS" # Ensure this path is correct relative to train.py

    if not os.path.exists(DATA_PATH):
        print(f"ERROR: Dataset path '{DATA_PATH}' not found.")
        return

    try:
        ds = LibriTTSDataset(root_dir=DATA_PATH, length=16000)
        # CRITICAL FIX: Set num_workers=0 for Mac to avoid BrokenPipeError
        loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0) 
    except Exception as e:
        print(f"Dataset Error: {e}")
        return

    # --- MODELS ---
    encoder = CELPWatermarkEncoder(num_devices=NUM_DEVICES).to(DEVICE)
    detector = WatermarkDetector(num_classes=NUM_CLASSES).to(DEVICE)
    stft_loss_fn = MultiResolutionSTFTLoss().to(DEVICE)

    # Joint Optimizer
    optimizer = optim.Adam([
        {'params': detector.parameters(), 'lr': LR},
        {'params': encoder.parameters(), 'lr': LR * 0.1} 
    ])
    ce_loss = torch.nn.CrossEntropyLoss()

    print(f"Starting training on {DEVICE} for {NUM_CLASSES} classes...")

    for epoch in range(EPOCHS):
        total_loss = 0
        correct = 0
        total = 0
        
        for i, audio in enumerate(loader):
            audio = audio.to(DEVICE)
            current_batch_size = audio.shape[0]
            
            # 1. GENERATE LABELS
            target_labels = torch.randint(0, NUM_CLASSES, (current_batch_size,)).to(DEVICE)
            
            # 2. PREPARE INPUTS
            mask_watermark = (target_labels > 0)
            final_audio = audio.clone()
            
            if mask_watermark.any():
                audio_to_encode = audio[mask_watermark]
                device_indices = target_labels[mask_watermark] - 1
                encoded_audio = encoder(audio_to_encode, device_indices)
                final_audio[mask_watermark] = encoded_audio

            optimizer.zero_grad()

            # 3. DETECT
            logits = detector(final_audio) 
            
            # 4. LOSSES
            loss_cls = ce_loss(logits, target_labels)
            loss_quality = torch.tensor(0.0).to(DEVICE)
            loss_reg = torch.tensor(0.0).to(DEVICE)
            
            if mask_watermark.any():
                # Quality Loss
                watermarked_subset = final_audio[mask_watermark]
                original_subset = audio[mask_watermark]
                loss_quality = stft_loss_fn(watermarked_subset, original_subset)
                
                # Reg Loss
                indices = target_labels[mask_watermark] - 1
                curr_offsets = encoder.device_offsets(indices)
                curr_scales = encoder.device_scales(indices)
                loss_reg = torch.mean(curr_offsets**2) + torch.mean((curr_scales - 1.0)**2)
                
            loss = loss_cls + (2.0 * loss_quality) + (1.0 * loss_reg)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(encoder.parameters(), 1.0)
            optimizer.step()
            
            total_loss += loss.item()
            
            preds = torch.argmax(logits, dim=1)
            correct += (preds == target_labels).sum().item()
            total += current_batch_size
            
            if i % 10 == 0:
                print(f"Batch {i}/{len(loader)} Loss: {loss.item():.4f}")

        avg_loss = total_loss / len(loader)
        acc = correct / total
        print(f"Epoch {epoch+1} | Loss: {avg_loss:.4f} | Accuracy: {acc:.2%}")
        
        # Debug print
        with torch.no_grad():
            print(f"   Device 1 Scales (First 5): {encoder.device_scales.weight[0][:5].cpu().numpy()}")

if __name__ == '__main__':
    main()