import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock2d(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size, stride=stride, padding=padding, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class ResidualBlock2d(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.act = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        return self.act(x + self.net(x))


class ConvBlock1d(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=7, stride=1, padding=3):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size, stride=stride, padding=padding, bias=False),
            nn.BatchNorm1d(out_ch),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class ResidualBlock1d(nn.Module):
    def __init__(self, channels, kernel_size=5, dilation=1):
        super().__init__()
        padding = dilation * (kernel_size - 1) // 2
        self.net = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size, padding=padding, dilation=dilation, bias=False),
            nn.BatchNorm1d(channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(channels, channels, kernel_size, padding=padding, dilation=dilation, bias=False),
            nn.BatchNorm1d(channels),
        )
        self.act = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        return self.act(x + self.net(x))


class STFTMaskNet(nn.Module):
    def __init__(self, in_ch=1, base_ch=32):
        super().__init__()
        self.net = nn.Sequential(
            ConvBlock2d(in_ch, base_ch),
            ResidualBlock2d(base_ch),
            ResidualBlock2d(base_ch),
            ConvBlock2d(base_ch, base_ch),
            nn.Conv2d(base_ch, 1, kernel_size=1),
        )

    def forward(self, logmag):
        return torch.sigmoid(self.net(logmag))


class DeviceTFCodebook(nn.Module):
    def __init__(self, num_devices, n_freq_bins, code_time_steps=16):
        super().__init__()
        self.num_devices = num_devices
        self.n_freq_bins = n_freq_bins
        self.code_time_steps = code_time_steps
        self.codes = nn.Parameter(torch.randn(num_devices, n_freq_bins, code_time_steps))
        nn.init.normal_(self.codes, mean=0.0, std=0.1)

    def forward(self, labels, target_time_steps):
        code = self.codes[labels]  # (B,F,Tc)
        code = F.normalize(code.flatten(1), dim=1).view_as(code)
        code = F.interpolate(
            code.unsqueeze(1),
            size=(self.n_freq_bins, target_time_steps),
            mode="bilinear",
            align_corners=False
        ).squeeze(1)
        return code

    def normalized_all(self):
        c = self.codes
        c = F.normalize(c.flatten(1), dim=1).view_as(c)
        return c


class WaveLabelConditioner(nn.Module):
    def __init__(self, num_devices, cond_dim):
        super().__init__()
        self.embedding = nn.Embedding(num_devices, cond_dim)
        nn.init.orthogonal_(self.embedding.weight)

    def forward(self, labels):
        cond = self.embedding(labels)
        cond = F.normalize(cond, dim=1)
        return cond


class WaveResidualWatermarker(nn.Module):
    """
    Training-time version with internal label embedding.
    """
    def __init__(self, num_devices=6, cond_dim=32, base_ch=32):
        super().__init__()
        self.cond = WaveLabelConditioner(num_devices, cond_dim)

        self.in_proj = nn.Sequential(
            ConvBlock1d(1, base_ch, kernel_size=7, padding=3),
            ConvBlock1d(base_ch, base_ch, kernel_size=7, padding=3),
        )

        self.cond_proj = nn.Linear(cond_dim, base_ch)

        self.res_stack = nn.Sequential(
            ResidualBlock1d(base_ch, dilation=1),
            ResidualBlock1d(base_ch, dilation=2),
            ResidualBlock1d(base_ch, dilation=4),
            ResidualBlock1d(base_ch, dilation=8),
        )

        self.out = nn.Sequential(
            nn.Conv1d(base_ch, 1, kernel_size=7, padding=3),
            nn.Tanh()
        )

    def forward(self, clean_audio, labels):
        x = self.in_proj(clean_audio)
        c = self.cond(labels).unsqueeze(-1)
        c = self.cond_proj(c.squeeze(-1)).unsqueeze(-1)
        x = x + c
        x = self.res_stack(x)
        residual = self.out(x)
        return residual


class PublicWaveResidualWatermarker(nn.Module):
    """
    Deployment-time version with external wave key.
    """
    def __init__(self, cond_dim=32, base_ch=32):
        super().__init__()
        self.cond_dim = cond_dim

        self.in_proj = nn.Sequential(
            ConvBlock1d(1, base_ch, kernel_size=7, padding=3),
            ConvBlock1d(base_ch, base_ch, kernel_size=7, padding=3),
        )

        self.cond_proj = nn.Linear(cond_dim, base_ch)

        self.res_stack = nn.Sequential(
            ResidualBlock1d(base_ch, dilation=1),
            ResidualBlock1d(base_ch, dilation=2),
            ResidualBlock1d(base_ch, dilation=4),
            ResidualBlock1d(base_ch, dilation=8),
        )

        self.out = nn.Sequential(
            nn.Conv1d(base_ch, 1, kernel_size=7, padding=3),
            nn.Tanh()
        )

    def forward(self, clean_audio, wave_key):
        # wave_key: (B, cond_dim)
        wave_key = F.normalize(wave_key, dim=1)

        x = self.in_proj(clean_audio)
        c = self.cond_proj(wave_key).unsqueeze(-1)
        x = x + c
        x = self.res_stack(x)
        residual = self.out(x)
        return residual


class HybridWatermarker(nn.Module):
    """
    Training-time version with internal user tables.
    """
    def __init__(
        self,
        n_fft=512,
        hop_length=160,
        win_length=400,
        num_devices=6,
        alpha_stft=0.05,
        alpha_wave=0.02,
        code_time_steps=16
    ):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.alpha_stft = alpha_stft
        self.alpha_wave = alpha_wave

        self.n_freq_bins = n_fft // 2 + 1
        self.mask_net = STFTMaskNet(in_ch=1, base_ch=32)
        self.codebook = DeviceTFCodebook(
            num_devices=num_devices,
            n_freq_bins=self.n_freq_bins,
            code_time_steps=code_time_steps
        )
        self.wave_branch = WaveResidualWatermarker(num_devices=num_devices, cond_dim=32, base_ch=32)

        window = torch.hann_window(win_length)
        self.register_buffer("window", window)

    def stft(self, audio):
        x = audio.squeeze(1)
        return torch.stft(
            x,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window,
            return_complex=True
        )

    def istft(self, spec, length):
        wav = torch.istft(
            spec,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window,
            length=length
        )
        return wav.unsqueeze(1)

    def forward(self, clean_audio, labels):
        B, _, T = clean_audio.shape

        clean_spec = self.stft(clean_audio)
        clean_mag = torch.abs(clean_spec) + 1e-8
        clean_phase = clean_spec / clean_mag
        TT = clean_mag.size(-1)

        logmag = torch.log(clean_mag).unsqueeze(1)
        mask = self.mask_net(logmag)
        code = self.codebook(labels, target_time_steps=TT)

        delta_mag = self.alpha_stft * clean_mag * mask.squeeze(1) * code
        wm_mag = torch.clamp(clean_mag + delta_mag, min=1e-8)
        wm_spec = wm_mag * clean_phase
        stft_watermarked_audio = self.istft(wm_spec, length=T)
        residual_stft = stft_watermarked_audio - clean_audio

        wave_residual_raw = self.wave_branch(clean_audio, labels)
        wave_residual = self.alpha_wave * torch.tanh(wave_residual_raw)

        residual_audio = residual_stft + wave_residual
        watermarked_audio = torch.clamp(clean_audio + residual_audio, min=-1.0, max=1.0)
        residual_audio = torch.clamp(watermarked_audio - clean_audio, min=-1.0, max=1.0)

        return watermarked_audio, residual_audio, delta_mag, mask, clean_mag


class PublicHybridWatermarker(nn.Module):
    """
    Deployment-time public encoder:
    no internal user tables, accepts external keys.
    """
    def __init__(
        self,
        n_fft=512,
        hop_length=160,
        win_length=400,
        alpha_stft=0.05,
        alpha_wave=0.02,
        code_time_steps=16,
        cond_dim=32
    ):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.alpha_stft = alpha_stft
        self.alpha_wave = alpha_wave
        self.code_time_steps = code_time_steps
        self.cond_dim = cond_dim

        self.n_freq_bins = n_fft // 2 + 1
        self.mask_net = STFTMaskNet(in_ch=1, base_ch=32)
        self.wave_branch = PublicWaveResidualWatermarker(cond_dim=cond_dim, base_ch=32)

        window = torch.hann_window(win_length)
        self.register_buffer("window", window)

    def stft(self, audio):
        x = audio.squeeze(1)
        return torch.stft(
            x,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window,
            return_complex=True
        )

    def istft(self, spec, length):
        wav = torch.istft(
            spec,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window,
            length=length
        )
        return wav.unsqueeze(1)

    def forward(self, clean_audio, tf_code, wave_key):
        """
        tf_code:  (B, F, Tc) or (F, Tc)
        wave_key: (B, cond_dim) or (cond_dim,)
        """
        B, _, T = clean_audio.shape

        if tf_code.dim() == 2:
            tf_code = tf_code.unsqueeze(0).expand(B, -1, -1)
        if wave_key.dim() == 1:
            wave_key = wave_key.unsqueeze(0).expand(B, -1)

        clean_spec = self.stft(clean_audio)
        clean_mag = torch.abs(clean_spec) + 1e-8
        clean_phase = clean_spec / clean_mag
        TT = clean_mag.size(-1)

        logmag = torch.log(clean_mag).unsqueeze(1)
        mask = self.mask_net(logmag)

        tf_code = F.normalize(tf_code.flatten(1), dim=1).view_as(tf_code)
        tf_code = F.interpolate(
            tf_code.unsqueeze(1),  # (B,1,F,Tc)
            size=(self.n_freq_bins, TT),
            mode="bilinear",
            align_corners=False
        ).squeeze(1)  # (B,F,TT)

        delta_mag = self.alpha_stft * clean_mag * mask.squeeze(1) * tf_code
        wm_mag = torch.clamp(clean_mag + delta_mag, min=1e-8)
        wm_spec = wm_mag * clean_phase
        stft_watermarked_audio = self.istft(wm_spec, length=T)
        residual_stft = stft_watermarked_audio - clean_audio

        wave_residual_raw = self.wave_branch(clean_audio, wave_key)
        wave_residual = self.alpha_wave * torch.tanh(wave_residual_raw)

        residual_audio = residual_stft + wave_residual
        watermarked_audio = torch.clamp(clean_audio + residual_audio, min=-1.0, max=1.0)
        residual_audio = torch.clamp(watermarked_audio - clean_audio, min=-1.0, max=1.0)

        return watermarked_audio, residual_audio, delta_mag, mask, clean_mag


class ResidualSpectrogramDetector(nn.Module):
    def __init__(self, num_devices=6, n_fft=512, hop_length=160, win_length=400):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length

        window = torch.hann_window(win_length)
        self.register_buffer("window", window)

        self.features = nn.Sequential(
            ConvBlock2d(1, 16, stride=1),
            ConvBlock2d(16, 32, stride=2),
            ResidualBlock2d(32),
            ConvBlock2d(32, 64, stride=2),
            ResidualBlock2d(64),
            ConvBlock2d(64, 128, stride=2),
            ResidualBlock2d(128),
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(128, num_devices)

    def forward(self, residual_audio):
        x = residual_audio.squeeze(1)
        spec = torch.stft(
            x,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window,
            return_complex=True
        )
        logmag = torch.log(torch.abs(spec) + 1e-8).unsqueeze(1)
        feat = self.features(logmag)
        feat = self.pool(feat).flatten(1)
        return self.fc(feat)


class SingleResBranch(nn.Module):
    def __init__(self, in_ch=1, base_ch=16):
        super().__init__()
        self.net = nn.Sequential(
            ConvBlock2d(in_ch, base_ch, stride=1),
            ConvBlock2d(base_ch, base_ch * 2, stride=2),
            ResidualBlock2d(base_ch * 2),
            ConvBlock2d(base_ch * 2, base_ch * 4, stride=2),
            ResidualBlock2d(base_ch * 4),
            ConvBlock2d(base_ch * 4, base_ch * 8, stride=2),
            ResidualBlock2d(base_ch * 8),
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.out_dim = base_ch * 8

    def forward(self, x):
        x = self.net(x)
        x = self.pool(x).flatten(1)
        return x


class WaveBlindBranch(nn.Module):
    def __init__(self, base_ch=32):
        super().__init__()
        self.net = nn.Sequential(
            ConvBlock1d(1, base_ch, kernel_size=9, stride=2, padding=4),
            ConvBlock1d(base_ch, base_ch * 2, kernel_size=9, stride=2, padding=4),
            ResidualBlock1d(base_ch * 2, dilation=1),
            ConvBlock1d(base_ch * 2, base_ch * 4, kernel_size=9, stride=2, padding=4),
            ResidualBlock1d(base_ch * 4, dilation=2),
            ConvBlock1d(base_ch * 4, base_ch * 4, kernel_size=9, stride=2, padding=4),
            ResidualBlock1d(base_ch * 4, dilation=4),
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.out_dim = base_ch * 4

    def forward(self, x):
        x = self.net(x)
        x = self.pool(x).squeeze(-1)
        return x


class HybridBlindDetector(nn.Module):
    def __init__(self, num_devices=6):
        super().__init__()

        self.spec_cfgs = [
            (256, 80, 200),
            (512, 160, 400),
            (1024, 320, 800),
        ]

        self.windows = nn.ParameterList([
            nn.Parameter(torch.hann_window(w), requires_grad=False)
            for _, _, w in self.spec_cfgs
        ])

        self.branch_256 = SingleResBranch(in_ch=1, base_ch=16)
        self.branch_512 = SingleResBranch(in_ch=1, base_ch=16)
        self.branch_1024 = SingleResBranch(in_ch=1, base_ch=16)
        self.wave_branch = WaveBlindBranch(base_ch=32)

        fusion_dim = (
            self.branch_256.out_dim +
            self.branch_512.out_dim +
            self.branch_1024.out_dim +
            self.wave_branch.out_dim
        )

        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, num_devices)
        )

    def _spec(self, x, n_fft, hop_length, win_length, window):
        spec = torch.stft(
            x,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            window=window,
            return_complex=True
        )
        logmag = torch.log(torch.abs(spec) + 1e-8)
        return logmag.unsqueeze(1)

    def forward(self, watermarked_audio):
        x = watermarked_audio.squeeze(1)

        s256 = self._spec(x, *self.spec_cfgs[0], self.windows[0])
        s512 = self._spec(x, *self.spec_cfgs[1], self.windows[1])
        s1024 = self._spec(x, *self.spec_cfgs[2], self.windows[2])

        f256 = self.branch_256(s256)
        f512 = self.branch_512(s512)
        f1024 = self.branch_1024(s1024)
        fwave = self.wave_branch(watermarked_audio)

        feat = torch.cat([f256, f512, f1024, fwave], dim=1)
        return self.classifier(feat)
