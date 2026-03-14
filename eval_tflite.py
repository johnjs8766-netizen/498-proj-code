import os
import torch
import numpy as np
import tensorflow as tf
from torch.utils.data import DataLoader
from datasets import WatermarkDataset
from models import PluginSEANetGenerator
from losses import MultiResolutionSTFTLoss
import time
# ------------------------------
# Settings
# ------------------------------
device = "cpu"
key_dim = 128
num_devices = 6
alpha = 2.0
audio_len = 16000 * 3
batch_size = 1

pt_checkpoint = "checkpoints/generator_epoch_16.pth"
tflite_path = "embedded_models/generator_int8.tflite"

#https://wiki.seeedstudio.com/XIAO-BLE-Sense-TFLite-Getting-Started/
#https://openelab.io/blogs/learn/tensorflow-lite-on-esp32?srsltid=AfmBOopGvm-_gLD4mVV7rxlUGS6S7ubEvEyGfU7mNeAHYISa1AsoQi31

# ------------------------------
# Load PyTorch generator
# ------------------------------
generator = PluginSEANetGenerator(key_dim=key_dim).to(device)
generator.load_state_dict(torch.load(pt_checkpoint, map_location=torch.device('cpu')))
generator.eval()

# Load key vault
key_vault = torch.zeros(num_devices, key_dim).to(device)
for u in range(num_devices):
    key_vault[u] = torch.load(f"user_keys/user_{u}_plugin_key.pt", map_location=torch.device('cpu')).to(device)

# ------------------------------
# Load TFLite model
# ------------------------------
interpreter = tf.lite.Interpreter(model_path=tflite_path, experimental_delegates=None)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# ------------------------------
# Prepare dataset
# ------------------------------
test_dataset = WatermarkDataset("dataset_libritts/dataset/metadata.csv", target_sr=16000, duration=3.0, mode='test')
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# ------------------------------
# Loss function
# ------------------------------
mr_stft_loss = MultiResolutionSTFTLoss()

# ------------------------------
# Evaluation loop
# ------------------------------
total_euclid = 0.0
total_loss_pt = 0.0
total_loss_tflite = 0.0
num_samples = 0
torch_time = 0.0
tflite_time = 0.0

with torch.no_grad():
    for raw_audio, _ in test_loader:
        # ------------------------------
        # Prepare inputs
        # ------------------------------
        raw_audio_np = raw_audio.numpy()  # (1, 1, T)
        raw_audio_torch = raw_audio.to(device)
        batch_size = raw_audio_np.shape[0]

        # Use fixed user assignment for both PyTorch & TFLite
        target_label = np.random.randint(0, num_devices, size=(batch_size,))
        batch_keys_np = np.stack([key_vault[u].cpu().numpy() for u in target_label], axis=0)  # (B, key_dim)
        batch_keys_torch = torch.from_numpy(batch_keys_np).to(device)

        # ------------------------------
        # PyTorch model
        # ------------------------------
        torch_start_time = time.perf_counter()
        wm_pt = generator(raw_audio_torch, batch_keys_torch)
        torch_end_time = time.perf_counter()
        watermarked_pt = torch.clamp(raw_audio_torch + (alpha * wm_pt * torch.abs(raw_audio_torch)), -1.0, 1.0)
        loss_pt = mr_stft_loss(raw_audio_torch, watermarked_pt)
        total_loss_pt += loss_pt.item()

        # ------------------------------
        # TFLite model
        # ------------------------------
        interpreter.set_tensor(input_details[0]['index'], raw_audio_np)
        interpreter.set_tensor(input_details[1]['index'], batch_keys_np)

        tflite_start_time = time.perf_counter()
        interpreter.invoke()
        tflite_end_time = time.perf_counter()

        wm_float = interpreter.get_tensor(output_details[0]['index'])
        watermarked_tflite = np.clip(raw_audio_np + (alpha * wm_float * np.abs(raw_audio_np)), -1.0, 1.0)

        # Compute Euclidean loss
        wm_pt_np = wm_pt.cpu().numpy()
        euclid = np.linalg.norm(wm_pt_np - wm_float)
        total_euclid += euclid

        # Compute STFT loss
        loss_tflite = mr_stft_loss(torch.from_numpy(raw_audio_np), torch.from_numpy(watermarked_tflite))
        total_loss_tflite += loss_tflite.item()

        #benchmarking
        torch_time += torch_end_time - torch_start_time
        tflite_time += tflite_end_time - tflite_start_time

        num_samples += 1

avg_loss_pt = total_loss_pt / num_samples
avg_loss_tflite = total_loss_tflite / num_samples
avg_euclid = total_euclid / num_samples
avg_torch_time = torch_time / num_samples
avg_tflite_time = tflite_time / num_samples

print(f"Average MultiResolutionSTFTLoss - PyTorch: {avg_loss_pt:.6f}")
print(f"Average MultiResolutionSTFTLoss - TFLite INT8: {avg_loss_tflite:.6f}")
print(f"Generator Output Euclidean Distance: {avg_euclid:.6f}")
print(f"Average Inference Time - Pytorch: {avg_torch_time:.6f}")
print(f"Average Inference Time - TFLite INT8: {avg_tflite_time:.6f}")

# Average MultiResolutionSTFTLoss - PyTorch: 0.074865
# Average MultiResolutionSTFTLoss - TFLite INT8: 0.074815
# Generator Output Euclidean Distance: 0.018093
# Average Inference Time - Pytorch: 0.125551
# Average Inference Time - TFLite INT8: 0.301928
# Elapsed time: 9.169998520519584e-07 seconds

