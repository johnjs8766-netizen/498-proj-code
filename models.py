import torch
import torch.nn as nn
from transformers import Wav2Vec2Model

class DepthwiseSeparableConv1d(nn.Module):
    #this conv module is capable for embedded devices with limited resources
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__()
        self.depthwise = nn.Conv1d(in_channels, in_channels, kernel_size=kernel_size, 
                                   stride=stride, padding=padding, groups=in_channels)
        self.pointwise = nn.Conv1d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.pointwise(self.depthwise(x))

class PluginSEANetGenerator(nn.Module):
    def __init__(self, key_dim=32):
        super().__init__()
        
        self.encoder = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=7, stride=1, padding=3),
            nn.LeakyReLU(0.2),
            nn.Conv1d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv1d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2)
        )
        
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(128 + key_dim, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose1d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv1d(32, 1, kernel_size=7, stride=1, padding=3),
            nn.Tanh() 
        )

    def forward(self, audio, plugin_key):
        # audio shape: (Batch, 1, Time)
        # plugin_key shape: (Batch, key_dim)
        
        encoded_audio = self.encoder(audio) 
        
        # Stretch the provided plugin key to match the audio's length
        k = plugin_key.unsqueeze(-1) # Shape: (B, key_dim, 1)
        k = k.expand(-1, -1, encoded_audio.size(-1)) # Shape: (B, key_dim, T_compressed)
        
        combined_features = torch.cat([encoded_audio, k], dim=1) 
        
        watermark = self.decoder(combined_features)
        return watermark

class SOTAAudioDetector(nn.Module):
    def __init__(self, num_classes=7):
        super().__init__()
        #incorporate wav2vec2 as the feature embedding module
        self.backbone = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base")
        self.classifier = nn.Linear(self.backbone.config.hidden_size, num_classes)

    def forward(self, x):
        x = x.squeeze(1) 
        outputs = self.backbone(x).last_hidden_state
        pooled_features = outputs.mean(dim=1) 
        logits = self.classifier(pooled_features)
        return logits