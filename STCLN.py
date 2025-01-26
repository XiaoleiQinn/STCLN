import copy
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import sys
from torch.nn.modules.transformer import TransformerEncoder, TransformerEncoderLayer
from torch.nn.modules import LayerNorm
import math

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)

        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            # nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            # self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.up = nn.ConvTranspose2d(in_channels//2, in_channels//2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):

        x1 = self.up(x1) #不下采样

        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])

        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class UTAE(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True,
                 encoder_widths=[32, 64, 256],
                 decoder_widths=[64, 64, 256],
                 agg_mode="att_group",
                 n_head=8,
                 d_model=256,
                 d_k=32):
        super(UTAE, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.d_model=d_model
        self.bilinear = bilinear
        self.encoder_widths = encoder_widths
        self.decoder_widths = decoder_widths
        self.inc = DoubleConv(n_channels, encoder_widths[0])

        self.down1 = Down(encoder_widths[0], encoder_widths[1])

        factor = 2 if bilinear else 1

        self.up3 = Up(decoder_widths[-1]+encoder_widths[-2], decoder_widths[-2] , bilinear)#尝试等于False

        self.temporal_encoder = LTAE2d(
            in_channels=encoder_widths[-1],
            d_model=d_model,
            n_head=n_head,
            mlp=[d_model, encoder_widths[-1]],
            return_att=True,
            d_k=d_k,
        )
        self.temporal_aggregator = Temporal_Aggregator(mode=agg_mode)


    def forward(self, x ,batch_positions=None):

        b_x = x.shape[0]
        l_x = x.shape[1]
        d_x = x.shape[2]
        h_x = x.shape[-1]


        x = x.view(-1, *x.shape[-3:])  # bl,d,h,w

        x1 = self.inc(x)

        x2 = self.down1(x1)  # bl,d,h,w




        obs_embed = x2.view(b_x, l_x, -1, h_x //1, h_x//1)  # bldhw 不下采样
        obs_embed,attn = self.temporal_encoder(
            obs_embed, batch_positions=batch_positions
        ) #(bhw)ld
        # obs_embed = obs_embed.view(b_x,h_x//2,h_x//2,l_x,-1).permute(0,3,4,1,2).contiguous().view(b_x*l_x,-1,h_x//2,h_x//2)
        obs_embed = obs_embed.view(b_x, h_x, h_x , l_x, -1).permute(0, 3, 4, 1, 2).contiguous().view(b_x * l_x,
                                                                                                              -1,
                                                                                                              h_x ,
                                                                                                              h_x) # bldhw 不下采样

        obs_embed = obs_embed.view(b_x,l_x, -1, h_x, h_x ).permute(0,3,4,1,2) # bhwld

        return obs_embed,x2




class UTAEPrediction(nn.Module):
    """
    Proxy task: missing-data imputation
        Given an incomplete time series with some patches being masked randomly,
        the network is asked to regress the central pixels of these masked patches
        based on the residual ones.
    """

    def __init__(self, utae: UTAE, num_features=13,dropout=0.7):
        """
        :param bert: the BERT-Former model acting as a feature extractor
        :param num_features: number of features of an input pixel to be predicted
        """

        super().__init__()
        self.utae = utae
        self.linear = nn.Linear(self.utae.encoder_widths[1], num_features)
        self.midlinear = nn.Linear(self.utae.encoder_widths[1], num_features)
        self.dropout=dropout
        self.MASK_TOKEN = nn.Parameter(torch.zeros(1))
    def forward(self, x,cluster_id_x,pos):
        target=x.clone()
        b_x = x.shape[0]
        l_x = x.shape[1]
        c_x = x.shape[2]
        h_x = x.shape[3]

        # mask = torch.ones(x.shape[1]).cuda()
        # mask = F.dropout(mask, self.dropout) * (1 - self.dropout)
        # mask = mask.unsqueeze(0).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).repeat(
        #     (x.shape[0], 1, x.shape[2], x.shape[3], x.shape[4]))  # b l c h w

        # mask = torch.ones(x.shape[0],x.shape[1]).cuda()
        # mask = F.dropout(mask, self.dropout) * (1 - self.dropout)
        # mask = mask.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).repeat(
        #     (1, 1, x.shape[2], x.shape[3], x.shape[4]))  # b l c h w

        mask = torch.ones(x.shape[0],x.shape[1],x.shape[3],x.shape[4]).cuda()
        mask = F.dropout(mask, self.dropout) * (1 - self.dropout)
        mask = mask.unsqueeze(2).repeat(
            (1, 1, x.shape[2], 1, 1))  # b l c h w

        clusterLmean = torch.mean(cluster_id_x.float(), dim=[3, 4], keepdim=True).repeat(
            (1, 1, 1, mask.shape[3], mask.shape[4]))
        mask[clusterLmean <= 0.9] = 1  # To it smarter?
        x=x.clone()
        x[mask == 0] = self.MASK_TOKEN

        x,x2 = self.utae(x,pos) #bhwld

        return self.linear(x).permute(0,3,4,1,2),self.midlinear(x2.permute(0,2,3,1)).permute(0,3,1,2).view(x.shape[0],x.shape[3],-1,*x2.shape[-2:]),target,mask #.permute(0,3,4,1,2),

# ----------fine-tune-------------
class UTAEClassification(nn.Module):
    """
    Downstream task: Satellite Time Series Classification
    """

    def __init__(self, utae: UTAE):
        """
        :param bert: the BERT-Former model
        :param num_classes: number of classes to be classified
        """

        super().__init__()
        self.utae = utae
        self.outlinear = nn.Linear(self.utae.encoder_widths[-1], self.utae.n_classes)#self.utae.decoder_widths[0], self.utae.n_classes 先unet在TE：self.utae.d_model, self.utae.n_classes
        # self.midlinear = nn.Linear(self.utae.encoder_widths[-1], self.utae.n_classes)#self.utae.decoder_widths[0], self.utae.n_classes 先unet在TE：self.utae.d_model, self.utae.n_classes
        self.conv1 = nn.Conv2d(self.utae.encoder_widths[-1], self.utae.encoder_widths[-1], (1,1), bias=False)
        self.conv2 = nn.Conv2d(self.utae.encoder_widths[-1], self.utae.encoder_widths[-1], (1,1),bias=False)
        self.conv3 = nn.Conv2d(self.utae.encoder_widths[-1], self.utae.encoder_widths[-1], (1, 1), bias=False)
        self.l = nn.Parameter(torch.zeros(1))
        self.dropout = nn.Dropout(0.1)
    def forward(self, x,pos):

        # # # 尝试用原来的输出
        out,x2 = self.utae(x, pos) #out:bhwld x2:bl,d,h,w
        out_1 = self.conv1(out.permute(0,3,4,1,2).view(-1,out.shape[-1],out.shape[1],out.shape[1])).view(out.shape[0],-1,out.shape[1]**2).permute(0,2,1) #b (hw) (ld)

        out, _ = torch.max(out, dim=3)
        x2_2 = self.conv2(x2).view(out.shape[0],-1, x2.shape[-1] ** 2)  # b (ld)(hw)
        attn = torch.matmul(out_1, x2_2)/np.power(self.utae.encoder_widths[-1], 0.5)  # b hw hw
        sfm = nn.Softmax(dim=2)
        attn = sfm(attn)

        out = self.conv3(out.permute(0,3,1,2)).permute(0,2,3,1)

        out = self.l*torch.bmm(attn, out.view(-1, out.shape[1] ** 2, self.utae.encoder_widths[-1]))+out.view(-1, out.shape[1] ** 2, self.utae.encoder_widths[-1]) #b hw d
        out = self.outlinear(out).permute(0,2,1).view(out.shape[0],-1,x2.shape[-1],x2.shape[-1])


        return out,x2


class TemporallySharedBlock(nn.Module):
    """
    Helper module for convolutional encoding blocks that are shared across a sequence.
    This module adds the self.smart_forward() method the the block.
    smart_forward will combine the batch and temporal dimension of an input tensor
    if it is 5-D and apply the shared convolutions to all the (batch x temp) positions.
    """

    def __init__(self, pad_value=None):
        super(TemporallySharedBlock, self).__init__()
        self.out_shape = None
        self.pad_value = pad_value

    def smart_forward(self, input):
        if len(input.shape) == 4:
            return self.forward(input)
        else:
            b, t, c, h, w = input.shape

            if self.pad_value is not None:
                dummy = torch.zeros(input.shape, device=input.device).float()
                self.out_shape = self.forward(dummy.view(b * t, c, h, w)).shape

            out = input.view(b * t, c, h, w)
            if self.pad_value is not None:
                pad_mask = (out == self.pad_value).all(dim=-1).all(dim=-1).all(dim=-1)
                if pad_mask.any():
                    temp = (
                        torch.ones(
                            self.out_shape, device=input.device, requires_grad=False
                        )
                        * self.pad_value
                    )
                    temp[~pad_mask] = self.forward(out[~pad_mask])
                    out = temp
                else:
                    out = self.forward(out)
            else:
                out = self.forward(out)
            _, c, h, w = out.shape
            out = out.view(b, t, c, h, w)
            return out


class ConvLayer(nn.Module):
    def __init__(
        self,
        nkernels,
        norm="batch",
        k=3,
        s=1,
        p=1,
        n_groups=4,
        last_relu=True,
        padding_mode="reflect",
    ):
        super(ConvLayer, self).__init__()
        layers = []
        if norm == "batch":
            nl = nn.BatchNorm2d
        elif norm == "instance":
            nl = nn.InstanceNorm2d
        elif norm == "group":
            nl = lambda num_feats: nn.GroupNorm(
                num_channels=num_feats,
                num_groups=n_groups,
            )
        else:
            nl = None
        for i in range(len(nkernels) - 1):
            layers.append(nn.Conv2d(
                    in_channels=nkernels[i],
                    out_channels=nkernels[i + 1],
                    kernel_size=k,
                    padding=p,
                    stride=s,
                    padding_mode=padding_mode,
                ))

            if nl is not None:
                layers.append(nl(nkernels[i + 1]))

            if last_relu:
                layers.append(nn.ReLU())
            elif i < len(nkernels) - 2:
                layers.append(nn.ReLU())
        self.conv = nn.Sequential(*layers)

    def forward(self, input):
        return self.conv(input)


class ConvBlock(TemporallySharedBlock):
    def __init__(
        self,
        nkernels,
        pad_value=None,
        norm="batch",
        last_relu=True,
        padding_mode="reflect",
    ):
        super(ConvBlock, self).__init__(pad_value=pad_value)
        self.conv = ConvLayer(
            nkernels=nkernels,
            norm=norm,
            last_relu=last_relu,
            padding_mode=padding_mode,
        )

    def forward(self, input):
        return self.conv(input)


class DownConvBlock(TemporallySharedBlock):
    def __init__(
        self,
        d_in,
        d_out,
        k,
        s,
        p,
        pad_value=None,
        norm="batch",
        padding_mode="reflect",
    ):
        super(DownConvBlock, self).__init__(pad_value=pad_value)
        self.down = ConvLayer(
            nkernels=[d_in, d_in],
            norm=norm,
            k=k,
            s=s,
            p=p,
            padding_mode=padding_mode,
        )
        self.conv1 = ConvLayer(
            nkernels=[d_in, d_out],
            norm=norm,
            padding_mode=padding_mode,
        )
        self.conv2 = ConvLayer(
            nkernels=[d_out, d_out],
            norm=norm,
            padding_mode=padding_mode,
        )

    def forward(self, input):
        out = self.down(input)
        out = self.conv1(out)
        out = out + self.conv2(out)
        return out


class UpConvBlock(nn.Module):
    def __init__(
        self, d_in, d_out, k, s, p, norm="batch", d_skip=None, padding_mode="reflect"
    ):
        super(UpConvBlock, self).__init__()
        d = d_out if d_skip is None else d_skip
        self.skip_conv = nn.Sequential(
            nn.Conv2d(in_channels=d, out_channels=d, kernel_size=1),
            nn.BatchNorm2d(d),
            nn.ReLU(),
        )
        self.up = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=d_in, out_channels=d_out, kernel_size=k, stride=s, padding=p
            ),
            nn.BatchNorm2d(d_out),
            nn.ReLU(),
        )
        self.conv1 = ConvLayer(
            nkernels=[d_out + d, d_out], norm=norm, padding_mode=padding_mode
        )
        self.conv2 = ConvLayer(
            nkernels=[d_out, d_out], norm=norm, padding_mode=padding_mode
        )

    def forward(self, input, skip):
        out = self.up(input)
        out = torch.cat([out, self.skip_conv(skip)], dim=1)
        out = self.conv1(out)
        out = out + self.conv2(out)
        return out

class UpConvBlockTs(nn.Module):
    def __init__(
        self, d_in, d_out, k, s, p, norm="batch", d_skip=None, padding_mode="reflect"
    ):
        super(UpConvBlockTs, self).__init__()
        d = d_out if d_skip is None else d_skip
        self.skip_conv = nn.Sequential(
            nn.Conv2d(in_channels=d, out_channels=d, kernel_size=1),
            nn.BatchNorm2d(d),
            nn.ReLU(),
        )
        self.up = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=d_in, out_channels=d_out, kernel_size=k, stride=s, padding=p
            ),
            nn.BatchNorm2d(d_out),
            nn.ReLU(),
        )
        self.conv1 = ConvLayer(
            nkernels=[d_out + d, d_out], norm=norm, padding_mode=padding_mode
        )
        self.conv2 = ConvLayer(
            nkernels=[d_out, d_out], norm=norm, padding_mode=padding_mode
        )

    def forward(self, input, skip):
        out = self.up(input)
        out = torch.cat([out, self.skip_conv(skip)], dim=1)
        out = self.conv1(out)
        out = out + self.conv2(out)
        return out

class Temporal_Aggregator(nn.Module):
    def __init__(self, mode="mean"):
        super(Temporal_Aggregator, self).__init__()
        self.mode = mode

    def forward(self, x, pad_mask=None, attn_mask=None):
        if pad_mask is not None and pad_mask.any():
            if self.mode == "att_group":
                n_heads, b, t, t, h, w = attn_mask.shape
                attn = attn_mask.view(n_heads * b, t * t, h, w)
                if x.shape[-2] > w:
                    attn = nn.Upsample(
                        size=x.shape[-2:], mode="bilinear", align_corners=False
                    )(attn)
                else:
                    attn = nn.AvgPool2d(kernel_size=w // x.shape[-2])(attn)
                attn = attn.view(n_heads, b, t, t, *x.shape[-2:])
                out = torch.stack(x.chunk(n_heads, dim=2))  # hxBxTxC/hxHxW

                # out = attn * out
                out = torch.matmul(attn.permute(0, 1, 4, 5, 2, 3), out.permute(0, 1, 4, 5, 2, 3)).permute(0, 1, 5, 4, 2,
                                                                                                          3)  # hxBxTxC/hxHxW
                # out = out.sum(dim=3)  # sum on temporal dim -> hxBxC/hxHxW

                out = torch.cat([group for group in out], dim=1)  # -> BxCxTxHxW
                return out
            elif self.mode == "att_mean":
                attn = attn_mask.mean(dim=0)  # average over heads -> BxTxHxW
                attn = nn.Upsample(
                    size=x.shape[-2:], mode="bilinear", align_corners=False
                )(attn)
                attn = attn * (~pad_mask).float()[:, :, None, None]
                out = (x * attn[:, :, None, :, :]).sum(dim=1)
                return out
            elif self.mode == "mean":
                out = x * (~pad_mask).float()[:, :, None, None, None]
                out = out.sum(dim=1) / (~pad_mask).sum(dim=1)[:, None, None, None]
                return out
        else:
            if self.mode == "att_group":
                n_heads, b, t,t, h, w = attn_mask.shape
                attn = attn_mask.view(n_heads * b, t*t, h, w)
                if x.shape[-2] > w:
                    attn = nn.Upsample(
                        size=x.shape[-2:], mode="bilinear", align_corners=True
                    )(attn)
                else:
                    attn = nn.AvgPool2d(kernel_size=w // x.shape[-2])(attn)
                attn = attn.view(n_heads, b, t,t, *x.shape[-2:])

                out = torch.stack(x.chunk(n_heads, dim=2))  # hxBxTxC/hxHxW
                out = torch.matmul(attn.permute(0,1,4,5,2,3), out.permute(0,1,4,5,2,3)).permute(0,1,4,5,2,3) #hxBxC/hxTxHxW
                out = torch.cat([group for group in out], dim=2) # -> BxCxTxHxW
                return out
            elif self.mode == "att_mean":

                n_heads, b, t, t, h, w = attn_mask.shape
                # head x b x t x t x h x w
                attn = attn_mask.mean(dim=0)  # average over heads -> BxTxTxHxW
                # attn = attn_mask[0]
                attn = attn.view(b,t*t,h,w)
                attn = nn.Upsample(
                    size=x.shape[-2:], mode="bilinear", align_corners=True
                )(attn)
                attn = attn.view(b,t,t,*attn.shape[-2:])#BxTxTxHxW
                # out = (x * attn[:, :, None, :, :]).sum(dim=1)
                out = torch.matmul(attn.permute(0,3,4,1,2),x.permute(0,3,4,1,2)).permute(0,3,4,1,2)  #x input shpae: BTCHW
                return out
            elif self.mode == "mean":
                return x.permute(0,2,1,3,4)
                # return x.mean(dim=1)

class LTAE2d(nn.Module):
    def __init__(
        self,
        in_channels=128,
        n_head=16,
        d_k=4,
        mlp=[256, 128],
        dropout=0.1, #0.2 源数据量
        d_model=256,

        T=1000,
        return_att=False,
        positional_encoding=True,
    ):
        """
        Lightweight Temporal Attention Encoder (L-TAE) for image time series.
        Attention-based sequence encoding that maps a sequence of images to a single feature map.
        A shared L-TAE is applied to all pixel positions of the image sequence.
        Args:
            in_channels (int): Number of channels of the input embeddings.
            n_head (int): Number of attention heads.
            d_k (int): Dimension of the key and query vectors.
            mlp (List[int]): Widths of the layers of the MLP that processes the concatenated outputs of the attention heads.
            dropout (float): dropout
            d_model (int, optional): If specified, the input tensors will first processed by a fully connected layer
                to project them into a feature space of dimension d_model.
            T (int): Period to use for the positional encoding.
            return_att (bool): If true, the module returns the attention masks along with the embeddings (default False)
            positional_encoding (bool): If False, no positional encoding is used (default True).
        """
        super(LTAE2d, self).__init__()
        self.in_channels = in_channels
        self.mlp = copy.deepcopy(mlp)
        self.return_att = return_att
        self.n_head = n_head

        if d_model is not None:
            self.d_model = d_model
            self.inconv = nn.Conv1d(in_channels, d_model, 1)
        else:
            self.d_model = in_channels
            self.inconv = None
        assert self.mlp[0] == self.d_model

        if positional_encoding:
            self.positional_encoder = PositionalEncoder(
                self.d_model // n_head, T=T, repeat=n_head
            )
        else:
            self.positional_encoder = None

        # if positional_encoding:#用SITFormer的positional_encoding
        #     self.positional_encoder = PositionalEncoding(d_model=self.d_model, max_len=366)
        # else:
        #     self.positional_encoder = None

        #用Yu-Hsiang Huang的
        # self.positional_encoder = nn.Embedding.from_pretrained(
        #     get_sinusoid_encoding_table(T+1, self.d_model, padding_idx=0),
        #     freeze=True)

        # self.attention_heads = MultiHeadAttention(
        #     n_head=n_head,d_model=d_model, d_k=d_k, d_v=d_k
        # )

        # self.layer_stack = nn.ModuleList([
        #     EncoderLayer(d_model, d_model*4, n_head, d_k, d_k, dropout=dropout)
        #     for _ in range(3)])

        # 用另一个TELayer
        self.layer_stack = nn.ModuleList([
            TransformerEncoderLayer(d_model, nhead=8, dim_feedforward=d_model*4, dropout=0.1)
            for _ in range(3)])

        self.in_norm = nn.GroupNorm(
            num_groups=n_head,
            num_channels=self.in_channels,
        )
        self.out_norm = nn.GroupNorm(
            num_groups=n_head,
            num_channels=mlp[-1],
        )

        layers = []
        for i in range(len(self.mlp) - 1):
            layers.extend(
                [
                    nn.Linear(self.mlp[i], self.mlp[i + 1]),
                    nn.BatchNorm1d(self.mlp[i + 1]),
                    nn.ReLU(),
                ]
            )

        self.mlp = nn.Sequential(*layers)
        self.dropout = nn.Dropout(dropout)

        # 用原始的self-attention
        encoder_layer = TransformerEncoderLayer(256, 8, 256*4, 0.1) #是不是dropout原来设为0.3了
        encoder_norm = LayerNorm(256)

        self.transformer_encoder = TransformerEncoder(encoder_layer, 3, encoder_norm)

    def forward(self, x, batch_positions=None, pad_mask=None, return_comp=False):
        sz_b, seq_len, d, h, w = x.shape

        if pad_mask is not None:
            pad_mask = (
                pad_mask.unsqueeze(-1)
                .repeat((1, 1, h))
                .unsqueeze(-1)
                .repeat((1, 1, 1, w))
            )  # BxTxHxW
            pad_mask = (
                pad_mask.permute(0, 2, 3, 1).contiguous().view(sz_b * h * w, seq_len)
            )

        out = x.permute(0, 3, 4, 1, 2).contiguous().view(sz_b * h * w, seq_len, d)
        out = self.in_norm(out.permute(0, 2, 1)).permute(0, 2, 1)

        # if self.inconv is not None:
        #     out = self.inconv(out.permute(0, 2, 1)).permute(0, 2, 1)   #在UNet里保证下采样最后一层就是transformer的hidden，这里就不进行卷积了

        if self.positional_encoder is not None:
            bp = (
                batch_positions.unsqueeze(-1)
                .repeat((1, 1, h))
                .unsqueeze(-1)
                .repeat((1, 1, 1, w))
            )  # BxTxHxW

            bp = bp.permute(0, 2, 3, 1).contiguous().view(sz_b * h * w, seq_len)

            out = out + self.positional_encoder(bp) # b*h*w,l,d

            # # Yu-Hsiang Huang
            # batchsize, seq, d = out.shape
            # src_pos = torch.arange(1, seq + 1, dtype=torch.long).expand(batchsize, seq).to(out.device)
            # out = out + self.positional_encoder(src_pos)

        # out, attn = self.attention_heads(out,out,out)
        # out = self.dropout(out)



        # out = self.transformer_encoder(out.transpose(0,1)).transpose(0,1) # [batch_size, seq_length, embed_size]
        for enc_layer in self.layer_stack:
            out, attn = enc_layer(out)

        # 用另一个网站的TELayer
        attn = attn.view( sz_b, h, w, seq_len,seq_len).unsqueeze(0).permute(
            0, 1, 4,5, 2, 3
        )  # head x b x t x t x h x w

        # attn = attn.view(self.n_head, sz_b, h, w, seq_len,seq_len).permute(
        #     0, 1, 4,5, 2, 3
        # )  # head x b x t x t x h x w

        # attn=out
        if self.return_att:
            return out, attn
        else:
            return out

# https://buomsoo-kim.github.io/attention/2020/04/27/Attention-mechanism-21.md/
class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout,batch_first=True)

        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = F.relu

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = F.relu
        super(TransformerEncoderLayer, self).__setstate__(state)

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        # type: (Tensor, Optional[Tensor], Optional[Tensor]) -> Tensor

        src2, weights = self.self_attn(src, src, src, attn_mask=src_mask,
                                       key_padding_mask=src_key_padding_mask)

        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)

        return src, weights #weights->(N,l,l)

#---------------Yu Hsiang Huang-------------------------
class EncoderLayer(nn.Module):
    ''' Compose with two layers '''

    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.slf_attn = MultiHeadAttention(
            n_head, d_model, d_k, d_v, dropout=dropout)
        self.pos_ffn = PositionwiseFeedForward(d_model, d_inner, dropout=dropout)

    def forward(self, enc_input):
        enc_output, enc_slf_attn = self.slf_attn(
            enc_input, enc_input, enc_input)

        enc_output = self.pos_ffn(enc_output)


        return enc_output, enc_slf_attn

class MultiHeadAttention(nn.Module):
    ''' Multi-Head Attention module '''

    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        super().__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, n_head * d_k)
        self.w_ks = nn.Linear(d_model, n_head * d_k)
        self.w_vs = nn.Linear(d_model, n_head * d_v)
        nn.init.normal_(self.w_qs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_ks.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_vs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_v)))

        self.attention = ScaledDotProductAttention(temperature=np.power(d_k, 0.5))
        self.layer_norm = nn.LayerNorm(d_model)

        self.fc = nn.Linear(n_head * d_v, d_model)
        nn.init.xavier_normal_(self.fc.weight)

        self.dropout = nn.Dropout(dropout)


    def forward(self, q, k, v, mask=None):

        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head

        sz_b, len_q, _ = q.size()
        sz_b, len_k, _ = k.size()
        sz_b, len_v, _ = v.size()

        residual = q

        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k).cuda()
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k).cuda()
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v).cuda()

        q = q.permute(2, 0, 1, 3).contiguous().view(-1, len_q, d_k) # (n*b) x lq x dk
        k = k.permute(2, 0, 1, 3).contiguous().view(-1, len_k, d_k) # (n*b) x lk x dk
        v = v.permute(2, 0, 1, 3).contiguous().view(-1, len_v, d_v) # (n*b) x lv x dv

        # mask = mask.repeat(n_head, 1, 1) # (n*b) x .. x ..
        output, attn = self.attention(q, k, v, mask=mask)

        output = output.view(n_head, sz_b, len_q, d_v)
        output = output.permute(1, 2, 0, 3).contiguous().view(sz_b, len_q, -1) # b x lq x (n*dv)

        output = self.dropout(self.fc(output))
        output = self.layer_norm(output + residual)

        return output, attn

class PositionwiseFeedForward(nn.Module):
    ''' A two-feed-forward-layer module '''

    def __init__(self, d_in, d_hid, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Conv1d(d_in, d_hid, 1) # position-wise
        self.w_2 = nn.Conv1d(d_hid, d_in, 1) # position-wise
        self.layer_norm = nn.LayerNorm(d_in)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        output = x.transpose(1, 2)
        output = self.w_2(F.relu(self.w_1(output)))
        output = output.transpose(1, 2)
        output = self.dropout(output)
        output = self.layer_norm(output + residual)
        return output

class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, q, k, v, mask=None):

        attn = torch.bmm(q, k.transpose(1, 2))
        attn = attn / self.temperature

        if mask is not None:
            attn = attn.masked_fill(mask.bool(), -np.inf)

        attn = self.softmax(attn)
        attn = self.dropout(attn)
        output = torch.bmm(attn, v)

        return output, attn

def get_sinusoid_encoding_table(n_position, d_hid, padding_idx=None):
    ''' Sinusoid position encoding table '''

    def cal_angle(position, hid_idx):
        return position / np.power(10000, 2 * (hid_idx // 2) / d_hid)

    def get_posi_angle_vec(position):
        return [cal_angle(position, hid_j) for hid_j in range(d_hid)]

    sinusoid_table = np.array([get_posi_angle_vec(pos_i) for pos_i in range(n_position)])

    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

    if padding_idx is not None:
        # zero vector for padding dimension
        sinusoid_table[padding_idx] = 0.

    if torch.cuda.is_available():
        return torch.FloatTensor(sinusoid_table).cuda()
    else:
        return torch.FloatTensor(sinusoid_table)

class PositionalEncoder(nn.Module):
    def __init__(self, d, T=30, repeat=None, offset=0):
        super(PositionalEncoder, self).__init__()
        self.d = d
        self.T = T
        self.repeat = repeat
        self.denom = torch.pow(
            T, 2 * (torch.arange(offset, offset + d).float() // 2) / d
        )
        self.updated_location = False

    def forward(self, batch_positions):
        if not self.updated_location:
            self.denom = self.denom.to(batch_positions.device)
            self.updated_location = True
        sinusoid_table = (
            batch_positions[:, :, None] / self.denom[None, None, :]
        )  # B x T x C
        sinusoid_table[:, :, 0::2] = torch.sin(sinusoid_table[:, :, 0::2])  # dim 2i
        sinusoid_table[:, :, 1::2] = torch.cos(sinusoid_table[:, :, 1::2])  # dim 2i+1

        if self.repeat is not None:
            sinusoid_table = torch.cat(
                [sinusoid_table for _ in range(self.repeat)], dim=-1
            )

        return sinusoid_table

class PositionalEncoding(nn.Module):

    def __init__(self, d_model, max_len=366):
        super().__init__()

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len+1, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)         # [max_len, 1]
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()  # [d_model/2,]

        # keep pe[0,:] to zeros
        pe[1:, 0::2] = torch.sin(position * div_term)   # broadcasting to [max_len, d_model/2]
        pe[1:, 1::2] = torch.cos(position * div_term)   # broadcasting to [max_len, d_model/2]

        self.register_buffer('pe', pe)

    def forward(self, time):
        output = torch.stack([torch.index_select(self.pe, 0, time[i, :]) for i in range(time.shape[0])], dim=0)
        return output       # [batch_size, seq_length, embed_dim]


# if __name__=="__main__":
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     transformer = Encoder(batch_size=4,num_hidden=32,img_width=512,input_length=5,depth=5,img=torch.ones((4,3,512,512,6)),channellist=[13,32,62,128,512],filter_size=3)
#     print(transformer)