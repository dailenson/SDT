import torch
import torch.nn as nn
import torchvision.models as models
from models.transformer import *
from einops import rearrange

### content encoder
class Content_TR(nn.Module):
    def __init__(self, d_model=256, nhead=8, num_encoder_layers=3,
                 dim_feedforward=2048, dropout=0.1, activation="relu",
                 normalize_before=True):
        super(Content_TR, self).__init__()
        self.Feat_Encoder = nn.Sequential(*([nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)] +list(models.resnet18(pretrained=True).children())[1:-2]))
        encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
        self.add_position = PositionalEncoding(dropout=0.1, dim=d_model)
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, norm=encoder_norm)

    def forward(self, x):
        x = self.Feat_Encoder(x)
        #x = self.recti_channel(x)
        x = rearrange(x, 'n c h w -> (h w) n c')
        x = self.add_position(x)
        x = self.encoder(x)
        return x

### For the training of Chinese handwriting generation task, 
### we first pre-train the content encoder for character classification.
### No need to pre-train the encoder in other languages (e.g, Japanese, English and Indic).

class Content_Cls(nn.Module):
    def __init__(self, d_model=512, num_encoder_layers=3, num_classes=6763) -> None:
        super(Content_Cls, self).__init__()
        self.feature_ext = Content_TR(d_model, num_encoder_layers)
        self.cls_head = nn.Linear(d_model, num_classes)
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(self, x):
        x = self.feature_ext(x)
        x = torch.mean(x, 0)
        out = self.cls_head(x)
        return out