import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiResolutionSTFTLoss(nn.Module):
    def __init__(
        self,
        fft_sizes=[512, 1024, 2048],
        hop_sizes=[50, 120, 240],
        win_lengths=[240, 600, 1200]
    ):
        super().__init__()
        self.fft_sizes = fft_sizes
        self.hop_sizes = hop_sizes
        self.win_lengths = win_lengths

    def stft_loss(self, x, y, n_fft, hop_length, win_length):
        window = torch.hann_window(win_length, device=x.device)

        x_stft = torch.stft(
            x.squeeze(1),
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            window=window,
            return_complex=True
        )
        y_stft = torch.stft(
            y.squeeze(1),
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            window=window,
            return_complex=True
        )

        x_mag = torch.abs(x_stft) + 1e-7
        y_mag = torch.abs(y_stft) + 1e-7

        sc_loss = torch.norm(y_mag - x_mag, p="fro") / (torch.norm(x_mag, p="fro") + 1e-7)
        log_mag_loss = F.l1_loss(torch.log(y_mag), torch.log(x_mag))
        return sc_loss + log_mag_loss

    def forward(self, x, y):
        loss = 0.0
        for n_fft, hop, win in zip(self.fft_sizes, self.hop_sizes, self.win_lengths):
            loss += self.stft_loss(x, y, n_fft, hop, win)
        return loss / len(self.fft_sizes)


class WatermarkEnergyLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, residual_audio):
        return residual_audio.abs().mean()


class CodeOrthogonalityLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, normalized_codebook):
        flat = normalized_codebook.flatten(1)
        gram = flat @ flat.t()
        c = gram.size(0)
        eye = torch.eye(c, device=gram.device)
        return ((gram - eye) ** 2).mean()


class MaskSparsityLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, mask):
        return mask.mean()


class DeviceConsistencyLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, delta_mag, labels, num_devices):
        summary = delta_mag.flatten(1)
        summary = F.normalize(summary, dim=1)

        loss = 0.0
        count = 0
        for d in range(num_devices):
            idx = (labels == d).nonzero(as_tuple=True)[0]
            if idx.numel() >= 2:
                group = summary[idx]
                center = F.normalize(group.mean(dim=0, keepdim=True), dim=1)
                sims = F.cosine_similarity(group, center.expand_as(group), dim=1)
                loss += (1.0 - sims).mean()
                count += 1

        if count == 0:
            return torch.tensor(0.0, device=delta_mag.device)
        return loss / count
