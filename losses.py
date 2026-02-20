import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiResolutionSTFTLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.fft_sizes = [1024, 2048, 512]
        self.hop_sizes = [120, 240, 50]
        self.win_lengths = [600, 1200, 240]

    def forward(self, x_fake, x_real):
        loss = 0
        for fs, hs, wl in zip(self.fft_sizes, self.hop_sizes, self.win_lengths):
            window = torch.hann_window(wl).to(x_fake.device)
            # Safe STFT
            real_stft = torch.stft(x_real, fs, hs, wl, window, return_complex=True).abs() + 1e-7
            fake_stft = torch.stft(x_fake, fs, hs, wl, window, return_complex=True).abs() + 1e-7
            
            if (real_stft == 0).any():
                print("Real stft has 0")
            if (fake_stft == 0).any():
                print("Fake stft has 0")
                
            # Spectral Convergence + Log Magnitude
            loss += torch.norm(real_stft - fake_stft, 'fro') / torch.norm(real_stft, 'fro')
            loss += F.l1_loss(torch.log(real_stft), torch.log(fake_stft))
        return loss / len(self.fft_sizes)

def parameter_reg_loss(params):
    # Penalize deviation from 1.0 (Identity)
    return F.mse_loss(params, torch.ones_like(params))