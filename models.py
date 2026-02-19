import torch
import torch.nn as nn
import torchaudio
from dsp_utils import lpc_to_lsf, lsf_to_lpc_diff, lpc_torch

class CELPWatermarkEncoder(nn.Module):
    def __init__(self, num_devices=5, lpc_order=16):
        super().__init__()
        self.lpc_order = lpc_order
        self.num_devices = num_devices
        
        # Learnable Offsets and Scales for each LSF band
        self.device_offsets = nn.Embedding(num_devices, lpc_order)
        nn.init.constant_(self.device_offsets.weight, 0.0)
        
        self.device_scales = nn.Embedding(num_devices, lpc_order)
        nn.init.constant_(self.device_scales.weight, 1.0)

    def forward(self, audio, device_indices):
        # 1. Analysis: LPC Coefficients
        a_coeffs = lpc_torch(audio, self.lpc_order)
        
        # Create Identity Coefficients [1.0, 0.0, ...] with same shape as a_coeffs
        # This acts as the "1" in the filter H(z) = A(z) / 1
        one_coeffs = torch.zeros_like(a_coeffs)
        one_coeffs[:, 0] = 1.0
        
        # Extract Residual
        # Filter: H(z) = A(z) / 1
        # lfilter(waveform, a_coeffs_denom, b_coeffs_num)
        residual = torchaudio.functional.lfilter(
            audio, one_coeffs, a_coeffs, clamp=False
        )
        
        with torch.no_grad():
            lsf_orig = lpc_to_lsf(a_coeffs) 
            
        # 2. Per-Band Transformation
        offsets = self.device_offsets(device_indices) 
        scales = self.device_scales(device_indices)   
        
        lsf_new = (lsf_orig * scales) + (offsets * 0.1)
        
        # 3. Stability Enforcement
        lsf_sorted, _ = torch.sort(lsf_new, dim=1)
        lsf_clamped = torch.clamp(lsf_sorted, 0.01, 3.14) 
        
        # 4. Synthesis
        a_coeffs_new = lsf_to_lpc_diff(lsf_clamped)
        
        # Synthesis Filter: H(z) = 1 / A'(z)
        # Denominator (a) = a_coeffs_new
        # Numerator (b) = one_coeffs (Identity)
        watermarked_audio = torchaudio.functional.lfilter(
            residual, a_coeffs_new, one_coeffs, clamp=False
        )
        
        return watermarked_audio

class SEANetResnetBlock(nn.Module):
    def __init__(self, dim, dilation=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.ELU(),
            nn.Conv1d(dim, dim, 3, dilation=dilation, padding=dilation),
            nn.ELU(),
            nn.Conv1d(dim, dim, 1)
        )
    def forward(self, x):
        return x + self.block(x)

class WatermarkDetector(nn.Module):
    def __init__(self, num_classes=6, channels=32):
        super().__init__()
        self.conv_in = nn.Conv1d(1, channels, 7, padding=3)
        self.layers = nn.ModuleList()
        curr = channels
        for _ in range(4): 
            self.layers.append(nn.Sequential(
                nn.ELU(),
                nn.Conv1d(curr, curr*2, 4, stride=2, padding=1)
            ))
            curr *= 2
            self.layers.append(SEANetResnetBlock(curr, dilation=1))
            self.layers.append(SEANetResnetBlock(curr, dilation=3))
        self.final_norm = nn.GroupNorm(8, curr)
        self.classifier = nn.Conv1d(curr, num_classes, 1)

    def forward(self, x):
        if x.dim() == 2: x = x.unsqueeze(1)
        x = self.conv_in(x)
        for layer in self.layers:
            x = layer(x)
        x = self.final_norm(x)
        logits = self.classifier(x)
        return logits.mean(dim=-1)