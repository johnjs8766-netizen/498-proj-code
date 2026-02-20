import torch
import torch.nn as nn
import torchaudio
import numpy as np
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
        """
        Forward pass for CELPWatermarkEncoder.
        Ensures stable LPC synthesis and prevents NaNs.
        """
        #get lpc coefficients from audio
        a_coeffs = lpc_torch(audio, self.lpc_order)  # [B, lpc_order+1]

        #identity filter
        one_coeffs = torch.zeros_like(a_coeffs)
        one_coeffs[:, 0] = 1.0

        # Extract residual: residual = audio filtered by 1/A(z)
        residual = torchaudio.functional.lfilter(audio, one_coeffs, a_coeffs, clamp=False)

        #should be no NaNs in residual
        if torch.isnan(residual).any():
            print("NaN in residual")
        if torch.isinf(residual).any():
            print("Inf in residual")

        # Linear Predictive Coding coefficients -> Line Spectral Frequencies
        #LSFs always correspond to stable filters if they are strictly increasing and within (0,pi)
        with torch.no_grad():
            lsf_orig = lpc_to_lsf(a_coeffs)  # [B, lpc_order]

        # Learnable LSF band scales and offsets for each device
        raw_offsets = self.device_offsets(device_indices) #adjusts spectral envelope
        raw_scales = self.device_scales(device_indices) #adds small perturbation

        # Clamp to ensure stability
        scales = torch.clamp(1.0 + 0.15 * torch.tanh(raw_scales), 0.8, 1.0)
        offsets = torch.clamp(0.15 * torch.tanh(raw_offsets), -0.05, 0.05)
        lsf_new = lsf_orig * scales + offsets

        
        # enforce min gap and strictly increasing requirment
        eps = 1e-5
        min_gap = 0.15
        B, P = lsf_new.shape
        lsf_sorted, _ = torch.sort(lsf_new, dim=1) #must be strictly increasing
        first = torch.clamp(lsf_sorted[:, 0:1], eps, np.pi - min_gap*(P-1)) #clamp first value
        increments = torch.arange(P, device=lsf_new.device, dtype=lsf_new.dtype) * min_gap
        lsf_stable = first + increments.unsqueeze(0) #all other values are min_gap away from the previous
        lsf_stable = torch.clamp(lsf_stable, eps, np.pi - eps) #all values between (eps, pi - eps)

        # reconstruct LPC coefficients
        a_coeffs_new = lsf_to_lpc_diff(lsf_stable)
        a_coeffs_new[:, 1:] *= 0.8  # shrink to reduce pole magnitudes
        B, N = a_coeffs_new.shape
        a_stable = a_coeffs_new.clone()
        
        #scale any coefficients with poles outside the unit circle down
        for b in range(B):
            a = a_coeffs_new[b].detach().cpu().numpy()
            roots_ = np.roots(a) # Compute poles
            mag = np.abs(roots_)
            roots_[mag >= 0.99] *= 0.99 / mag[mag >= 0.99] # Scale any root outside 0.99 back
            a_stable[b] = torch.from_numpy(np.real_if_close(np.poly(roots_))).to(a_coeffs_new.device, dtype=a_coeffs_new.dtype) # Reconstruct polynomial
        
        # apply LPC filter to residual
        watermarked_audio = torchaudio.functional.lfilter(
            residual.double(),
            a_stable.double(),
            one_coeffs.double(),
            clamp=False
        ).float()

        # normalize
        max_val = watermarked_audio.abs().amax(dim=1, keepdim=True) + 1e-6
        watermarked_audio = watermarked_audio / max_val

        # diagnostics
        a_np = a_stable[0].detach().cpu().numpy()
        roots = np.roots(a_np)
        print(f"Max pole radius: {np.max(np.abs(roots)):.4f}") #must be < 1
        print(f"Min LSF gap: {torch.min(lsf_stable[:,1:] - lsf_stable[:,:-1]):.4f}") #must be > 0
        print(f"Min LSF: {torch.min(lsf_stable):.4f}, Max LSF: {torch.max(lsf_stable):.4f}") #min must be > 0, max must be < pi

        #no nans in produced audio
        if torch.isnan(watermarked_audio).any():
            print("NaN in watermarked_audio")
        if torch.isinf(watermarked_audio).any():
            print("Inf in watermarked_audio")

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