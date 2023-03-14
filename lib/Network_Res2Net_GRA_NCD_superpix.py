import torch
import torch.nn as nn
import torch.nn.functional as F
from lib.Res2Net_v1b import res2net50_v1b_26w_4s
import torchvision.transforms as transforms

from PIL import Image
import numpy as np
from lib.loss import update_spixl_map, get_spixel_image
def UnNormalize(tensor):
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m)
    return tensor

def normalization(data):
    _range = np.max(data) - np.min(data)
    return (data - np.min(data)) / _range

def tensor2PIL(feat):
    feat=feat[0].cpu().detach().numpy().squeeze()
    feat=255*normalization(feat) #(归一化)
    feat = feat.astype(np.uint8).copy()        
    feat=transforms.ToPILImage()(feat).convert('L')
    return feat
    
def diyplot(xx,S_g_pred,ra4_feat,S_5_pred,ra3_feat,S_4_pred,ra2_feat,S_3_pred):
    imgs=xx[0].cpu().numpy()
    imgs=255*imgs
    imgs = imgs.astype(np.uint8).copy()        
    imgs = xx[0].cpu().clone()
    imgs=UnNormalize(imgs)
    imgs=transforms.ToPILImage()(imgs).convert('RGB')
    # imgs.save("./ori.png")    

    S_g_pred_=tensor2PIL(S_g_pred).convert('RGB')
    # S_g_pred_.save("./S_g_pred_.png")    

    ra4_feat_ = F.interpolate(ra4_feat, scale_factor=32, mode='bilinear')   # Sup-4 (bs, 1, 44, 44) -> (bs, 1, 352, 352)
    ra4_feat_=tensor2PIL(ra4_feat_).convert('RGB')
    # ra4_feat_.save("./ra4_feat_.png")   

    S_5_pred_=tensor2PIL(S_5_pred).convert('RGB')
    # S_5_pred_.save("./S_5_pred_.png")   

    ra3_feat_ = F.interpolate(ra3_feat, scale_factor=16, mode='bilinear')   # Sup-4 (bs, 1, 44, 44) -> (bs, 1, 352, 352)
    ra3_feat_=tensor2PIL(ra3_feat_).convert('RGB')
    # ra3_feat_.save("./ra3_feat_.png")   

    S_4_pred_=tensor2PIL(S_4_pred).convert('RGB')
    # S_4_pred_.save("./S_4_pred_.png")   

    ra2_feat_ = F.interpolate(ra2_feat, scale_factor=8, mode='bilinear')   # Sup-4 (bs, 1, 44, 44) -> (bs, 1, 352, 352)
    ra2_feat_=tensor2PIL(ra2_feat_).convert('RGB')
    # ra2_feat_.save("./ra2_feat_.png")   

    S_3_pred_=tensor2PIL(S_3_pred).convert('RGB')
    # S_3_pred_.save("./S_3_pred_.png")   

    pred=[S_3_pred_,S_4_pred_,S_5_pred_,S_g_pred_]
    width,height=pred[0].size
    result1 = Image.new(pred[0].mode, (width* len(pred), height)) 
    for i, im in enumerate(pred):
        result1.paste(im, box=(i*width,0))

    feat=[imgs,ra2_feat_,ra3_feat_,ra4_feat_]
    width,height=feat[0].size
    result2 = Image.new(feat[0].mode, (width* len(pred), height)) 
    for i, im in enumerate(feat):
        result2.paste(im, box=(i*width,0))
    
    result = Image.new(feat[0].mode, (width* len(pred), height * 2)) 

    result.paste(result1, box=(0,0*height))
    result.paste(result2, box=(0, 1*height))
    result.save('perd_feat.png')
    return 0

class DecoderBlock(nn.Module):
    def __init__(self, in_channels, n_filters):
        super(DecoderBlock,self).__init__()

        self.conv1 = nn.Conv2d(in_channels, in_channels // 4, 1)
        self.norm1 = nn.BatchNorm2d(in_channels // 4)
        self.relu1 = partial(F.relu,inplace=True)

        self.deconv2 = nn.ConvTranspose2d(in_channels // 4, in_channels // 4, 3, stride=2, padding=1, output_padding=1)
        self.norm2 = nn.BatchNorm2d(in_channels // 4)
        self.relu2 = partial(F.relu,inplace=True)

        self.conv3 = nn.Conv2d(in_channels // 4, n_filters, 1)
        self.norm3 = nn.BatchNorm2d(n_filters)
        self.relu3 = partial(F.relu,inplace=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)
        x = self.deconv2(x)
        x = self.norm2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.norm3(x)
        x = self.relu3(x)
        return x

class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x


class RFB_modified(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(RFB_modified, self).__init__()
        self.relu = nn.ReLU(True)
        self.branch0 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
        )
        self.branch1 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 3), padding=(0, 1)),
            BasicConv2d(out_channel, out_channel, kernel_size=(3, 1), padding=(1, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=3, dilation=3)
        )
        self.branch2 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 5), padding=(0, 2)),
            BasicConv2d(out_channel, out_channel, kernel_size=(5, 1), padding=(2, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=5, dilation=5)
        )
        self.branch3 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 7), padding=(0, 3)),
            BasicConv2d(out_channel, out_channel, kernel_size=(7, 1), padding=(3, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=7, dilation=7)
        )
        self.conv_cat = BasicConv2d(4*out_channel, out_channel, 3, padding=1)
        self.conv_res = BasicConv2d(in_channel, out_channel, 1)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        x_cat = self.conv_cat(torch.cat((x0, x1, x2, x3), 1))

        x = self.relu(x_cat + self.conv_res(x))
        return x


class NeighborConnectionDecoder(nn.Module):
    def __init__(self, channel):
        super(NeighborConnectionDecoder, self).__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv_upsample1 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample2 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample3 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample4 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample5 = BasicConv2d(2*channel, 2*channel, 3, padding=1)

        self.conv_concat2 = BasicConv2d(2*channel, 2*channel, 3, padding=1)
        self.conv_concat3 = BasicConv2d(3*channel, 3*channel, 3, padding=1)
        self.conv4 = BasicConv2d(3*channel, 3*channel, 3, padding=1)
        self.conv5 = nn.Conv2d(3*channel, 1, 1)

    def forward(self, x1, x2, x3):
        x1_1 = x1
        x2_1 = self.conv_upsample1(self.upsample(x1)) * x2
        x3_1 = self.conv_upsample2(self.upsample(x2_1)) * self.conv_upsample3(self.upsample(x2)) * x3

        x2_2 = torch.cat((x2_1, self.conv_upsample4(self.upsample(x1_1))), 1)
        x2_2 = self.conv_concat2(x2_2)

        x3_2 = torch.cat((x3_1, self.conv_upsample5(self.upsample(x2_2))), 1)
        x3_2 = self.conv_concat3(x3_2)

        x = self.conv4(x3_2)
        x = self.conv5(x)

        return x

from functools import partial

class SuperpixelDecoder(nn.Module):
    def __init__(self, channel):
        super(SuperpixelDecoder, self).__init__()
        self.decoder4 = DecoderBlock(2048, 1024)
        self.decoder3 = DecoderBlock(1024, 512)
        self.decoder2 = DecoderBlock(512, 256)
        self.decoder1 = DecoderBlock(256, 256)

        self.finaldeconv1 = nn.ConvTranspose2d(256, 32, 4, 2, 1)
        self.finalrelu1 = partial(F.relu,inplace=True)
        self.finalconv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.finalrelu2 = partial(F.relu,inplace=True)
        self.finalconv3 = nn.Conv2d(32, 16, 3, padding=1)
        self.pred_mask = nn.Conv2d(16, 9, kernel_size=3, stride=1, padding=1, bias=True)
        self.softmax = nn.Softmax(1)
    def forward(self, x4, x3, x2, x1): #18,2048,11,11 18,1024,22,22 18,512,44,44 18,256,88,88

        d4 = self.decoder4(x4) + x3  #18,1024,22,22
        d3 = self.decoder3(d4) + x2  #18,512,44,44
        d2 = self.decoder2(d3) + x1  #18,256,88,88
        d1 = self.decoder1(d2)       #18，256，176，176
        out = self.finaldeconv1(d1)  #18,32,352,352
        out = self.finalrelu1(out)   #18,32,352,352
        out = self.finalconv2(out)   #18,32,352,352
        out = self.finalrelu2(out)   #18,32,352,352
        out = self.finalconv3(out) #bt,16,352,352
        mask = self.pred_mask(out)
        prob = self.softmax(mask)
        return prob

# Group-Reversal Attention (GRA) Block
class GRA(nn.Module):
    def __init__(self, channel, subchannel):
        super(GRA, self).__init__()
        self.group = channel//subchannel
        self.conv = nn.Sequential(
            nn.Conv2d(channel + self.group, channel, 3, padding=1), nn.ReLU(True),
        )
        self.score = nn.Conv2d(channel, 1, 3, padding=1)

    def forward(self, x, y):
        if self.group == 1:
            x_cat = torch.cat((x, y), 1)
        elif self.group == 2:
            xs = torch.chunk(x, 2, dim=1)
            x_cat = torch.cat((xs[0], y, xs[1], y), 1)
        elif self.group == 4:
            xs = torch.chunk(x, 4, dim=1)
            x_cat = torch.cat((xs[0], y, xs[1], y, xs[2], y, xs[3], y), 1)
        elif self.group == 8:
            xs = torch.chunk(x, 8, dim=1)
            x_cat = torch.cat((xs[0], y, xs[1], y, xs[2], y, xs[3], y, xs[4], y, xs[5], y, xs[6], y, xs[7], y), 1)
        elif self.group == 16:
            xs = torch.chunk(x, 16, dim=1)
            x_cat = torch.cat((xs[0], y, xs[1], y, xs[2], y, xs[3], y, xs[4], y, xs[5], y, xs[6], y, xs[7], y,
            xs[8], y, xs[9], y, xs[10], y, xs[11], y, xs[12], y, xs[13], y, xs[14], y, xs[15], y), 1)
        elif self.group == 32:
            xs = torch.chunk(x, 32, dim=1)
            x_cat = torch.cat((xs[0], y, xs[1], y, xs[2], y, xs[3], y, xs[4], y, xs[5], y, xs[6], y, xs[7], y,
            xs[8], y, xs[9], y, xs[10], y, xs[11], y, xs[12], y, xs[13], y, xs[14], y, xs[15], y,
            xs[16], y, xs[17], y, xs[18], y, xs[19], y, xs[20], y, xs[21], y, xs[22], y, xs[23], y,
            xs[24], y, xs[25], y, xs[26], y, xs[27], y, xs[28], y, xs[29], y, xs[30], y, xs[31], y), 1)
        else:
            raise Exception("Invalid Channel")

        x = x + self.conv(x_cat)
        y = y + self.score(x)

        return x, y


class ReverseStage(nn.Module):
    def __init__(self, channel):
        super(ReverseStage, self).__init__()
        self.weak_gra = GRA(channel, channel)
        self.medium_gra = GRA(channel, 8)
        self.strong_gra = GRA(channel, 1)

    def forward(self, x, y):
        # reverse guided block
        y = -1 * (torch.sigmoid(y)) + 1

        # three group-reversal attention blocks
        x, y = self.weak_gra(x, y)
        x, y = self.medium_gra(x, y)
        _, y = self.strong_gra(x, y)

        return y

class Network(nn.Module):
    # res2net based encoder decoder
    def __init__(self, channel=32, imagenet_pretrained=True):
        super(Network, self).__init__()
        # ---- ResNet Backbone ----
        self.resnet = res2net50_v1b_26w_4s(pretrained=imagenet_pretrained)
        # ---- Receptive Field Block like module ----
        self.rfb2_1 = RFB_modified(512, channel)
        self.rfb3_1 = RFB_modified(1024, channel)
        self.rfb4_1 = RFB_modified(2048, channel)
        # ---- Partial Decoder ----
        self.NCD = NeighborConnectionDecoder(channel)
        self.SUP = SuperpixelDecoder(channel)

        # # ---- reverse stage ----
        self.RS5 = ReverseStage(channel)
        self.RS4 = ReverseStage(channel)
        self.RS3 = ReverseStage(channel)

    def forward(self, x,spixeIds):
        # Feature Extraction
        # xx=x
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)      # bs, 64, 88, 88
        x1 = self.resnet.layer1(x)      # bs, 256, 88, 88
        x2 = self.resnet.layer2(x1)     # bs, 512, 44, 44
        x3 = self.resnet.layer3(x2)     # bs, 1024, 22, 22
        x4 = self.resnet.layer4(x3)     # bs, 2048, 11, 11

        # Receptive Field Block (enhanced)
        x2_rfb = self.rfb2_1(x2)        # channel -> 32 (18,32,44,44)
        x3_rfb = self.rfb3_1(x3)        # channel -> 32 (18,32,22,22)
        x4_rfb = self.rfb4_1(x4)        # channel -> 32 (18,32,11,11)

        # Neighbourhood Connected Decoder 
        # 如不变架构前提下，需修改的模块是NCD,设计一个更好的guidence
        # 上面错误，除NCD,每一层的guidence都不一样，额外自监督信息应该来自于每一层的特征图
        S_g = self.NCD(x4_rfb, x3_rfb, x2_rfb)

        S_sup = self.SUP(x4, x3, x2, x1)

        S_g_pred = F.interpolate(S_g, scale_factor=8, mode='bilinear')    # Sup-1 (bs, 1, 44, 44) -> (bs, 1, 352, 352)   

        # ---- reverse stage 5 ----
        guidance_g = F.interpolate(S_g, scale_factor=0.25, mode='bilinear')
        ra4_feat = self.RS5(x4_rfb, guidance_g)
        S_5 = ra4_feat + guidance_g
        S_5_pred = F.interpolate(S_5, scale_factor=32, mode='bilinear')  # Sup-2 (bs, 1, 11, 11) -> (bs, 1, 352, 352)

        # ---- reverse stage 4 ----
        guidance_5 = F.interpolate(S_5, scale_factor=2, mode='bilinear')
        ra3_feat = self.RS4(x3_rfb, guidance_5)
        S_4 = ra3_feat + guidance_5
        S_4_pred = F.interpolate(S_4, scale_factor=16, mode='bilinear')  # Sup-3 (bs, 1, 22, 22) -> (bs, 1, 352, 352)

        # ---- reverse stage 3 ----
        guidance_4 = F.interpolate(S_4, scale_factor=2, mode='bilinear')
        ra2_feat = self.RS3(x2_rfb, guidance_4)
        S_3 = ra2_feat + guidance_4
        if self.train:
            S_3_pred = F.interpolate(S_3, scale_factor=8, mode='bilinear')   # Sup-4 (bs, 1, 44, 44) -> (bs, 1, 352, 352)
        else:
            curr_spixl_map = update_spixl_map(spixeIds, S_sup)  #(bs, 1, 44, 44)
            # ori_sz_spixel_map = F.interpolate(curr_spixl_map.type(torch.float),scale_factor=8, mode='nearest').type(torch.int) #(bs, 1, 352, 352)
            curr_spixl_map = F.interpolate(curr_spixl_map.type(torch.float),scale_factor=1/8, mode='nearest').type(torch.int)
            b,_, h, w = curr_spixl_map.shape
            spixnum=curr_spixl_map[0,0].max()+1
            one_hot = torch.zeros(b, spixnum, h, w, dtype=torch.long).cuda() #超像素个数
            onehot_spixl_map = one_hot.scatter_(1, curr_spixl_map.type(torch.long).data, 1).type(torch.float32) #require long type
            S_3_=S_3.repeat(1,spixnum,1,1)
            Score_super=onehot_spixl_map*S_3_
            
            for bt in range(Score_super.shape[0]):
                for i in range(spixnum):
                    Score_super_=Score_super[bt,i]
                    Score_super_[Score_super[bt,i].nonzero()]=(Score_super[bt,i].sum())/Score_super[bt,i].nonzero().shape[0]
                    Score_super[bt,i]=Score_super_
            Score_super=Score_super.sum(1).unsqueeze(1)
            S_3_pred = F.interpolate(Score_super, scale_factor=8, mode='bilinear')   # Sup-4 (bs, 1, 44, 44) -> (bs, 1, 352, 352)      
        # S_sup = F.interpolate(S_sup, scale_factor=8, mode='bilinear')   # Sup-4 (bs, 1, 44, 44) -> (bs, 1, 352, 352)

    #将上采样的结果与GT比合理吗？=>loss必不可能为0=>最优解不存在=>会导致网络产生无意义的震荡=>?
    #RS模块的思想是去掉发现的伪装区域后(显著区域)，进一步挖掘剩下特征中的伪装区域，并引导粗标签修正
    #问题在于,去掉粗标签区域后隐藏目标边界信息更加难以区分，提升性能十分有限
        # diyplot(xx,S_g_pred,ra4_feat,S_5_pred,ra3_feat,S_4_pred,ra2_feat,S_3_pred)
        return S_g_pred, S_5_pred, S_4_pred, S_3_pred, S_sup


if __name__ == '__main__':
    import numpy as np
    from time import time
    net = Network(imagenet_pretrained=False)
    net.eval()

    dump_x = torch.randn(1, 3, 352, 352)
    frame_rate = np.zeros((1000, 1))
    for i in range(1000):
        start = time()
        y = net(dump_x)
        end = time()
        running_frame_rate = 1 * float(1 / (end - start))
        print(i, '->', running_frame_rate)
        frame_rate[i] = running_frame_rate
    print(np.mean(frame_rate))
    print(y.shape)