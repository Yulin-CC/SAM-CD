import os
import torch
from torch import nn
from .FastSAM.fastsam import FastSAM
from torch.nn import functional as F
from typing import Dict, List
from utils.misc import initialize_weights

def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)

class Space_Attention(nn.Module):
    def __init__(self, in_channels, out_channels, reduction=4):
        super(Space_Attention, self).__init__()
        self.SA = nn.Sequential( 
            nn.Conv2d(in_channels, in_channels // reduction, 1),
            nn.BatchNorm2d(in_channels // reduction, momentum=0.95),
            nn.ReLU(inplace=False),
            nn.Conv2d(in_channels // reduction, out_channels, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        b, c, h, w = x.size()        
        A = self.SA(x)
        return A

class _DecoderBlock(nn.Module):
    def __init__(self, in_channels_high, in_channels_low, out_channels):
        super(_DecoderBlock, self).__init__()
        self.up = nn.ConvTranspose2d(in_channels_high, in_channels_high, kernel_size=2, stride=2)
        in_channels = in_channels_high + in_channels_low
        self.decode = nn.Sequential(
            conv3x3(in_channels, out_channels),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            conv3x3(out_channels, out_channels),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x, low_feat):
        x = self.up(x)
        x = torch.cat((x, low_feat), dim=1)
        x = self.decode(x)        
        return x

class ResBlock(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(ResBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride
    
    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out

class SAM_CD(nn.Module):
    def __init__(
        self,
        num_embed=8,
        model_name: str='FastSAM-x.pt',
        device: str='cuda',
        conf: float=0.4,
        iou: float=0.9,
        imgsz: int=1024,
        retina_masks: bool=True,
        done_warmup: bool=True,
        ):
        super(SAM_CD, self).__init__()
        model_name = self._resolve_fastsam_weights(model_name)
        self.model = FastSAM(model_name)
        self.device = device
        self.retina_masks = retina_masks
        self.imgsz = imgsz
        self.conf = conf
        self.iou = iou
        self.image = None
        self.image_feats = None        
         
        self.Adapter32 = nn.Sequential(nn.Conv2d(640, 160, kernel_size=1, stride=1, padding=0, bias=False),
                                       nn.BatchNorm2d(160), nn.ReLU())
        self.Adapter16 = nn.Sequential(nn.Conv2d(640, 160, kernel_size=1, stride=1, padding=0, bias=False),
                                       nn.BatchNorm2d(160), nn.ReLU())
        self.Adapter8 = nn.Sequential(nn.Conv2d(320, 80, kernel_size=1, stride=1, padding=0, bias=False),
                                      nn.BatchNorm2d(80), nn.ReLU())
        self.Adapter4 = nn.Sequential(nn.Conv2d(160, 40, kernel_size=1, stride=1, padding=0, bias=False),
                                      nn.BatchNorm2d(40), nn.ReLU())
                                       
        self.Dec2 = _DecoderBlock(160, 160, 80)
        self.Dec1 = _DecoderBlock(80, 80, 40)  
        self.Dec0 = _DecoderBlock(40, 40, 64)
        
        self.SA = Space_Attention(16, 16, 4)
        self.segmenter = nn.Conv2d(64, num_embed, kernel_size=1)        
        self.resCD = self._make_layer(ResBlock, 128, 128, 6, stride=1)
        self.headC = nn.Sequential(nn.Conv2d(128, 16, kernel_size=1, stride=1, padding=0, bias=False), nn.BatchNorm2d(16), nn.ReLU())
        self.segmenterC = nn.Conv2d(16, 1, kernel_size=1)
                                        
        for param in self.model.model.parameters():
            param.requires_grad = False
        initialize_weights(self.Adapter32, self.Adapter16, self.Adapter8, self.Adapter4, self.Dec2, self.Dec1, self.Dec0,\
                           self.segmenter, self.resCD, self.headC, self.segmenterC)

    def _resolve_fastsam_weights(self, model_name: str) -> str:
        """Resolve FastSAM weights path across common project layouts."""
        if os.path.isabs(model_name) and os.path.exists(model_name):
            return model_name
        if os.path.exists(model_name):
            return model_name

        repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        candidates = [
            os.path.join(repo_root, model_name),
            os.path.join(repo_root, 'weights', os.path.basename(model_name)),
            os.path.join(repo_root, os.path.basename(model_name)),
        ]
        for path in candidates:
            if os.path.exists(path):
                return path
        raise FileNotFoundError(
            f'FastSAM weights not found: {model_name}. '
            f'Tried: {candidates}'
        )

    def run_encoder(self, image):
        self.image = image
        current_device = image.device
        # FastSAM uses Ultralytics inference pipeline; run in FP32 to avoid
        # NMS / mask ops breaking under AMP, then cast outputs back to the
        # caller's dtype so downstream adapter layers see the right dtype.
        with torch.autocast('cuda', enabled=False):
            image_fp32 = image.float()
            feats = self.model(
                image_fp32,
                device=str(current_device),
                retina_masks=self.retina_masks,
                imgsz=self.imgsz,
                conf=self.conf,
                iou=self.iou
                )
        # FastSAM may run internally on cuda:0 regardless of `device`.
        # Synchronize that device before P2P copy so rank>0 doesn't block
        # waiting for incomplete CUDA operations on the source device.
        torch.cuda.synchronize(torch.device('cuda:0'))
        # Move all feature tensors to the caller's device (no-op for rank0).
        return [f.to(current_device) if isinstance(f, torch.Tensor) else f for f in feats]

    def _make_layer(self, block, inplanes, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or inplanes != planes:
            downsample = nn.Sequential(
                conv1x1(inplanes, planes, stride),
                nn.BatchNorm2d(planes) )

        layers = []
        layers.append(block(inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def train(self, mode: bool = True):
        """
        Keep FastSAM backbone frozen in eval mode during SAM-CD training.

        Only the CD heads/adapters are intended to learn in this project.
        """
        super().train(mode)
        self.model.train(False)
        return self

    def forward(self, x1: torch.Tensor, x2: torch.Tensor):
    
        input_shape = x1.shape[-2:]
        featsA = self.run_encoder(x1)
        featsB = self.run_encoder(x2)
        
        featA_s4 = self.Adapter4(featsA[3].clone())
        featA_s8 = self.Adapter8(featsA[0].clone())
        featA_s16 = self.Adapter16(featsA[1].clone())
        featA_s32 = self.Adapter32(featsA[2].clone())
        
        decA_2 = self.Dec2(featA_s32, featA_s16)
        decA_1 = self.Dec1(decA_2, featA_s8)
        decA_0 = self.Dec0(decA_1, featA_s4)
        outA = self.segmenter(decA_0)
        
        featB_s4 = self.Adapter4(featsB[3].clone())
        featB_s8 = self.Adapter8(featsB[0].clone())
        featB_s16 = self.Adapter16(featsB[1].clone())
        featB_s32 = self.Adapter32(featsB[2].clone())              
        
        decB_2 = self.Dec2(featB_s32, featB_s16)
        decB_1 = self.Dec1(decB_2, featB_s8)
        decB_0 = self.Dec0(decB_1, featB_s4)
        outB = self.segmenter(decB_0)
             
        A = self.SA(torch.cat([outA, outB], dim=1))  
        featC = torch.cat([decA_0, decB_0], 1)
        featC = self.resCD(featC)
        featC = self.headC(featC) * A
        outC = self.segmenterC(featC)
        
        return F.interpolate(outC, input_shape, mode="bilinear", align_corners=True),\
               F.interpolate(outA, input_shape, mode="bilinear", align_corners=True),\
               F.interpolate(outB, input_shape, mode="bilinear", align_corners=True)
