import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiResolutionSTFTLoss(nn.Module):
    def __init__(self, fft_sizes=[1024, 2048, 512], hop_sizes=[120, 240, 50], win_lengths=[600, 1200, 240]):
        super().__init__()
        self.fft_sizes = fft_sizes
        self.hop_sizes = hop_sizes
        self.win_lengths = win_lengths

    def stft_loss(self, x, y, n_fft, hop_length, win_length):
        x_stft = torch.stft(x.squeeze(1), n_fft=n_fft, hop_length=hop_length, win_length=win_length, return_complex=True)
        y_stft = torch.stft(y.squeeze(1), n_fft=n_fft, hop_length=hop_length, win_length=win_length, return_complex=True)
        
        x_mag = torch.abs(x_stft) + 1e-7
        y_mag = torch.abs(y_stft) + 1e-7
        
        sc_loss = torch.norm(y_mag - x_mag, p="fro") / torch.norm(x_mag, p="fro")
        log_mag_loss = F.l1_loss(torch.log(y_mag), torch.log(x_mag))
        
        return sc_loss + log_mag_loss

    def forward(self, x, y):
        loss = 0.0
        for n_fft, hop_length, win_length in zip(self.fft_sizes, self.hop_sizes, self.win_lengths):
            loss += self.stft_loss(x, y, n_fft, hop_length, win_length)
        return loss / len(self.fft_sizes)