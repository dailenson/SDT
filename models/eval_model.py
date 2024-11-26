import torch
from torch import nn
import torch.nn.functional as F
from ptflops import get_model_complexity_info

class offline_style(nn.Module):
    def __init__(self, num_class=240, vote=False):
        super(offline_style, self).__init__()
        self.l1 = nn.Sequential(nn.Conv2d(1,96,5,2),nn.BatchNorm2d(96),nn.ReLU(),nn.MaxPool2d(3,2))
        # self.l1 = nn.Sequential(nn.Conv2d(1,96,5,2),nn.BatchNorm2d(96),nn.ReLU())
        self.l2 = nn.Sequential(nn.Conv2d(96,256,3,1,1),nn.BatchNorm2d(256),nn.ReLU(),nn.MaxPool2d(3,2))
        self.l3 = nn.Sequential(nn.Conv2d(256,384,3,1,1),nn.BatchNorm2d(384),nn.ReLU(),
                                nn.Conv2d(384,384,3,1,1),nn.BatchNorm2d(384),nn.ReLU(),nn.MaxPool2d(3,2))
        self.l4 = nn.Sequential(nn.Conv2d(384,256,3,1,1),nn.BatchNorm2d(256),nn.ReLU(),nn.MaxPool2d(3,2))
        self.fc1 = nn.Sequential(nn.Flatten(1),nn.Linear(1024, num_class))
        self.vote = vote

    def forward(self,x):
        if self.vote:
            n,c,h,w = x.size()
            if not self.training:
                x = x.view(n*c,1,h,w)
            out = self.l1(x)
            out = self.l2(out)
            out = self.l3(out)
            out = self.l4(out)
            n1,c1,h1,w1 = out.size()
            out = self.fc1(out)
            if not self.training:
                out = out.view(n,c,-1)
            return out
        else:
            n,c,h,w = x.size()
            x = x.view(n*c,1,h,w)
            out = self.l1(x)
            out = self.l2(out)
            out = self.l3(out)
            out = self.l4(out)
            n1,c1,h1,w1 = out.size()
            out = torch.mean(out.view(n,c,c1,h1,w1),1)
            out = self.fc1(out)
            # if not self.training:
            #     out = out.view(n,c,-1)
            return out
class Character_Net(nn.Module):
    def __init__(self, nclass=3755):
        super(Character_Net, self).__init__()
        self.l1 = nn.Sequential(nn.Conv1d(5, 64, kernel_size=7, stride=1, padding=3), nn.ReLU(), nn.BatchNorm1d(64))

        self.l2 = nn.MaxPool1d(kernel_size=2, stride=2)

        self.l3 = nn.Sequential(nn.Conv1d(64, 64, kernel_size=3, stride=1, padding=1), nn.ReLU(), nn.BatchNorm1d(64))

        self.l4 = nn.Sequential(nn.Conv1d(64, 128, kernel_size=3, stride=1, padding=1), nn.ReLU(), nn.BatchNorm1d(128))

        self.l5 = nn.MaxPool1d(kernel_size=2, stride=2)

        self.l6 = nn.Sequential(nn.Conv1d(128, 128, kernel_size=3, stride=1, padding=1), nn.ReLU(), nn.BatchNorm1d(128))

        self.l7 = nn.Sequential(nn.Conv1d(128, 256, kernel_size=3, stride=1, padding=1), nn.ReLU(), nn.BatchNorm1d(256))

        self.l8 = nn.MaxPool1d(kernel_size=2, stride=2)

        self.l9 = nn.Sequential(nn.Conv1d(256, 256, kernel_size=3, stride=1, padding=1), nn.ReLU())

        # self.l10 = nn.AvgPool1d()#Custom_maskAveragePooling # stride = 8
        print('num of character is {}'.format(nclass))
        self.l11 = nn.Linear(256, nclass)

    def forward(self, x, l):
        x = self.l1(x)
        x = self.l2(x)
        x = self.l3(x)
        x = self.l4(x)
        x = self.l5(x)
        x = self.l6(x)
        x = self.l7(x)
        x = self.l8(x)
        x = self.l9(x)
        hidden = mask_avr_pooling(x, torch.div(l, 8, rounding_mode='floor'))
        x = self.l11(hidden)
        return x

def mask_avr_pooling_rnn(x, l):
    # x: NTC, l:N
    N,T,C = x.size()
    mask = length_to_mask(l, max_len=T)
    mask = mask.unsqueeze(-1)
    # print(mask.shape)
    o = torch.sum(x*mask, dim=-2, keepdim=False)
    o = o/(l.unsqueeze(-1)+1e-5)
    return o

def mask_avr_pooling(x, l):
    # x: NTC, l:N
    N,C,T = x.size()
    mask = length_to_mask(l, max_len=T)
    mask = mask.unsqueeze(1)
    # print(mask.shape)
    o = torch.sum(x*mask, dim=-1, keepdim=False)
    o = o/(l.unsqueeze(-1)+1e-5)
    return o

def length_to_mask(length, max_len=None, dtype=None):
    """length: B.
    return B x max_len.
    If max_len is None, then max of length will be used.
    """
    assert len(length.shape) == 1, 'Length shape should be 1 dimensional.'
    max_len = max_len or length.max().item()
    mask = torch.arange(max_len, device=length.device,
                        dtype=length.dtype).expand(len(length), max_len) < length.unsqueeze(1)
    if dtype is not None:
        mask = torch.as_tensor(mask, dtype=dtype, device=length.device)
    return mask