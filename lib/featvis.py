import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import torch.nn.functional as F


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

def diyplot2(xx,S_g_pred,S_edge_pred,ra4_feat,S_5_pred,ra3_feat,S_4_pred,ra2_feat,S_3_pred,ra1_feat,S_2_pred):
    imgs=xx[0].cpu().numpy()
    imgs=255*imgs
    imgs = imgs.astype(np.uint8).copy()        
    imgs = xx[0].cpu().clone()
    imgs=UnNormalize(imgs)
    imgs=transforms.ToPILImage()(imgs).convert('RGB')
    # imgs.save("./ori.png")    

    S_g_pred_=tensor2PIL(S_g_pred).convert('RGB')
    # S_g_pred_.save("./S_g_pred_.png")    

    S_edge_pred_=tensor2PIL(S_edge_pred).convert('RGB')
    # S_edge_pred_.save("./S_edge_pred_.png")    

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

    ra1_feat_ = F.interpolate(ra1_feat, scale_factor=4, mode='bilinear')   # Sup-4 (bs, 1, 44, 44) -> (bs, 1, 352, 352)
    ra1_feat_=tensor2PIL(ra1_feat_).convert('RGB')
    # ra1_feat_.save("./ra1_feat_.png")   

    S_2_pred_=tensor2PIL(S_2_pred).convert('RGB')
    # S_2_pred_.save("./S_2_pred_.png")   

    pred=[S_2_pred_,S_3_pred_,S_4_pred_,S_5_pred_,S_g_pred_,S_edge_pred_]
    width,height=pred[0].size
    result1 = Image.new(pred[0].mode, (width* len(pred), height)) 
    for i, im in enumerate(pred):
        result1.paste(im, box=(i*width,0))

    feat=[imgs,ra1_feat_,ra2_feat_,ra3_feat_,ra4_feat_]
    width,height=feat[0].size
    result2 = Image.new(feat[0].mode, (width* len(pred), height)) 
    for i, im in enumerate(feat):
        result2.paste(im, box=(i*width,0))
    
    result = Image.new(feat[0].mode, (width* len(pred), height * 2)) 

    result.paste(result1, box=(0,0*height))
    result.paste(result2, box=(0, 1*height))
    result.save('perd_feat.png')
    return 0


def diyplot2_0(xx,S_g_pred,ra4_feat,S_5_pred,ra3_feat,S_4_pred,ra2_feat,S_3_pred,ra1_feat,S_2_pred):
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

    ra1_feat_ = F.interpolate(ra1_feat, scale_factor=4, mode='bilinear')   # Sup-4 (bs, 1, 44, 44) -> (bs, 1, 352, 352)
    ra1_feat_=tensor2PIL(ra1_feat_).convert('RGB')
    # ra1_feat_.save("./ra1_feat_.png")   

    S_2_pred_=tensor2PIL(S_2_pred).convert('RGB')
    # S_2_pred_.save("./S_2_pred_.png")   

    pred=[S_2_pred_,S_3_pred_,S_4_pred_,S_5_pred_,S_g_pred_,S_g_pred_]
    width,height=pred[0].size
    result1 = Image.new(pred[0].mode, (width* len(pred), height)) 
    for i, im in enumerate(pred):
        result1.paste(im, box=(i*width,0))

    feat=[imgs,ra1_feat_,ra2_feat_,ra3_feat_,ra4_feat_]
    width,height=feat[0].size
    result2 = Image.new(feat[0].mode, (width* len(pred), height)) 
    for i, im in enumerate(feat):
        result2.paste(im, box=(i*width,0))
    
    result = Image.new(feat[0].mode, (width* len(pred), height * 2)) 

    result.paste(result1, box=(0,0*height))
    result.paste(result2, box=(0, 1*height))
    result.save('perd_feat.png')
    return 0

def diyplot3(xx,S_g_pred,S_edge_pred,ra4_feat,S_5_pred,ra3_feat,S_4_pred,ra2_feat,S_3_pred,ra1_feat,S_2_pred,S_sur_pred):
    imgs=xx[0].cpu().numpy()
    imgs=255*imgs
    imgs = imgs.astype(np.uint8).copy()        
    imgs = xx[0].cpu().clone()
    imgs=UnNormalize(imgs)
    imgs=transforms.ToPILImage()(imgs).convert('RGB')
    # imgs.save("./ori.png")    

    S_g_pred_=tensor2PIL(S_g_pred).convert('RGB')
    # S_g_pred_.save("./S_g_pred_.png")    

    S_edge_pred_=tensor2PIL(S_edge_pred).convert('RGB')
    # S_edge_pred_.save("./S_edge_pred_.png")    

    S_sur_pred_=tensor2PIL(S_sur_pred).convert('RGB')
    # S_sur_pred_.save("./S_sur_pred_.png")  

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

    ra1_feat_ = F.interpolate(ra1_feat, scale_factor=4, mode='bilinear')   # Sup-4 (bs, 1, 44, 44) -> (bs, 1, 352, 352)
    ra1_feat_=tensor2PIL(ra1_feat_).convert('RGB')
    # ra1_feat_.save("./ra1_feat_.png")   

    S_2_pred_=tensor2PIL(S_2_pred).convert('RGB')
    # S_2_pred_.save("./S_2_pred_.png")   

    pred=[S_2_pred_,S_3_pred_,S_4_pred_,S_5_pred_,S_g_pred_,S_edge_pred_]
    width,height=pred[0].size
    result1 = Image.new(pred[0].mode, (width* len(pred), height)) 
    for i, im in enumerate(pred):
        result1.paste(im, box=(i*width,0))

    feat=[imgs,ra1_feat_,ra2_feat_,ra3_feat_,ra4_feat_,S_sur_pred_]
    width,height=feat[0].size
    result2 = Image.new(feat[0].mode, (width* len(pred), height)) 
    for i, im in enumerate(feat):
        result2.paste(im, box=(i*width,0))
    
    result = Image.new(feat[0].mode, (width* len(pred), height * 2)) 

    result.paste(result1, box=(0,0*height))
    result.paste(result2, box=(0, 1*height))
    result.save('perd_feat.png')
    return 0


def diyplot_gs(xx,S_g, S_sur):
    import torch
    guidance_g = F.interpolate(S_g, scale_factor=4, mode='bilinear')
    guidance_s = F.interpolate(S_sur, scale_factor=4, mode='bilinear')

    guidance_s_ = -1 * (torch.sigmoid(guidance_s)) + 1
    guidance_g_= -1 *(torch.sigmoid(guidance_g)) + 1

    guidance_s_2 = torch.sigmoid(guidance_s)
    # fusion=guidance_g_+guidance_s_
    fusion=guidance_s_2

    imgs=xx[0].cpu().numpy()
    imgs=255*imgs
    imgs = imgs.astype(np.uint8).copy()        
    imgs = xx[0].cpu().clone()
    imgs=UnNormalize(imgs)
    imgs=transforms.ToPILImage()(imgs).convert('RGB')

    guidance_s=tensor2PIL(guidance_s).convert('RGB')
    guidance_s_=tensor2PIL(guidance_s_).convert('RGB')
    guidance_g=tensor2PIL(guidance_g).convert('RGB')
    guidance_g_=tensor2PIL(guidance_g_).convert('RGB')
    fusion=tensor2PIL(fusion).convert('RGB')


    pred=[fusion,guidance_s,guidance_s_]
    width,height=pred[0].size
    result1 = Image.new(pred[0].mode, (width* len(pred), height)) 
    for i, im in enumerate(pred):
        result1.paste(im, box=(i*width,0))

    feat=[imgs,guidance_g,guidance_g_]
    width,height=feat[0].size
    result2 = Image.new(feat[0].mode, (width* len(pred), height)) 
    for i, im in enumerate(feat):
        result2.paste(im, box=(i*width,0))
    
    result = Image.new(feat[0].mode, (width* len(pred), height * 2)) 

    result.paste(result1, box=(0,0*height))
    result.paste(result2, box=(0, 1*height))
    result.save('vis_g_s_rs.png')
    return 0

def diyplot_db(img,S_g_pred,S_edge_pred,S_sur_pred,                            # (bs, 1, 352, 352)
                guidance_g,guidance_s,ra4_feat_g,ra4_feat_s,S_5_pred,          # (bs, 1, 11/11/11/11/352, 11/11/11/11/352)
                guidance_5,g3_sur,ra3_feat_g,ra3_feat_s,S_4_pred,              # (bs, 1, 22/22/22/22/352, 22/22/22/22/352)
                guidance_4,g2_sur,ra2_feat_g,ra2_feat_s,S_3_pred,              # (bs, 1, 44/44/44/44/352, 44/44/44/44/352)
                guidance_3,S_sur,ra1_feat_g,ra1_feat_s,S_2_pred):               # (bs, 1, 88/88/88/88/352, 88/88/88/88/352)
    import torch

    imgs=img[0].cpu().numpy()
    imgs=255*imgs
    imgs = imgs.astype(np.uint8).copy()        
    imgs = img[0].cpu().clone()
    imgs=UnNormalize(imgs)
    imgs=transforms.ToPILImage()(imgs).convert('RGB')

    S_g_pred=tensor2PIL(S_g_pred).convert('RGB')
    S_edge_pred=tensor2PIL(S_edge_pred).convert('RGB')
    S_sur_pred=tensor2PIL(S_sur_pred).convert('RGB')

    guidance_g = F.interpolate(guidance_g, scale_factor=32, mode='bilinear')    # Sup-4 (bs, 1, 11, 11) -> (bs, 1, 352, 352)
    guidance_g_sigmoid= -1 *(torch.sigmoid(guidance_g)) + 1
    guidance_g_sigmoid=tensor2PIL(guidance_g_sigmoid).convert('RGB')
    guidance_g=tensor2PIL(guidance_g).convert('RGB')
    guidance_s = F.interpolate(guidance_s, scale_factor=32, mode='bilinear')
    guidance_s_sigmoid = torch.sigmoid(guidance_s)
    guidance_s_sigmoid=tensor2PIL(guidance_s_sigmoid).convert('RGB')    
    guidance_s=tensor2PIL(guidance_s).convert('RGB')    
    ra4_feat_g = F.interpolate(ra4_feat_g, scale_factor=32, mode='bilinear')   
    ra4_feat_g=tensor2PIL(ra4_feat_g).convert('RGB')
    ra4_feat_s =-1* F.interpolate(ra4_feat_s, scale_factor=32, mode='bilinear')   
    ra4_feat_s=tensor2PIL(ra4_feat_s).convert('RGB')
    S_5_pred=tensor2PIL(S_5_pred).convert('RGB')

    guidance_5 = F.interpolate(guidance_5, scale_factor=16, mode='bilinear')    # Sup-4 (bs, 1, 22, 22) -> (bs, 1, 352, 352)
    guidance_5_sigmoid= -1 *(torch.sigmoid(guidance_5)) + 1
    guidance_5_sigmoid=tensor2PIL(guidance_5_sigmoid).convert('RGB')
    guidance_5=tensor2PIL(guidance_5).convert('RGB')
    g3_sur = F.interpolate(g3_sur, scale_factor=16, mode='bilinear')
    g3_sur_sigmoid = torch.sigmoid(g3_sur)
    g3_sur_sigmoid=tensor2PIL(g3_sur_sigmoid).convert('RGB')  
    g3_sur=tensor2PIL(g3_sur).convert('RGB')      
    ra3_feat_g = F.interpolate(ra3_feat_g, scale_factor=16, mode='bilinear')   
    ra3_feat_g=tensor2PIL(ra3_feat_g).convert('RGB')
    ra3_feat_s =-1* F.interpolate(ra3_feat_s, scale_factor=16, mode='bilinear')   
    ra3_feat_s=tensor2PIL(ra3_feat_s).convert('RGB')
    S_4_pred=tensor2PIL(S_4_pred).convert('RGB')

    guidance_4 = F.interpolate(guidance_4, scale_factor=8, mode='bilinear')    # Sup-4 (bs, 1, 44, 44) -> (bs, 1, 352, 352)
    guidance_4_sigmoid= -1 *(torch.sigmoid(guidance_4)) + 1
    guidance_4_sigmoid=tensor2PIL(guidance_4_sigmoid).convert('RGB')
    guidance_4=tensor2PIL(guidance_4).convert('RGB')
    g2_sur = F.interpolate(g2_sur, scale_factor=8, mode='bilinear')
    g2_sur_sigmoid = torch.sigmoid(g2_sur)
    g2_sur_sigmoid=tensor2PIL(g2_sur_sigmoid).convert('RGB')  
    g2_sur=tensor2PIL(g2_sur).convert('RGB')    
    ra2_feat_g = F.interpolate(ra2_feat_g, scale_factor=8, mode='bilinear')   
    ra2_feat_g=tensor2PIL(ra2_feat_g).convert('RGB')
    ra2_feat_s =-1* F.interpolate(ra2_feat_s, scale_factor=8, mode='bilinear')   
    ra2_feat_s=tensor2PIL(ra2_feat_s).convert('RGB')
    S_3_pred=tensor2PIL(S_3_pred).convert('RGB')

    guidance_3 = F.interpolate(guidance_3, scale_factor=4, mode='bilinear')    # Sup-4 (bs, 1, 22, 22) -> (bs, 1, 352, 352)
    guidance_3_sigmoid= -1 *(torch.sigmoid(guidance_3)) + 1
    guidance_3_sigmoid=tensor2PIL(guidance_3_sigmoid).convert('RGB')
    guidance_3=tensor2PIL(guidance_3).convert('RGB')
    S_sur = F.interpolate(S_sur, scale_factor=4, mode='bilinear')
    S_sur_sigmoid = torch.sigmoid(S_sur)
    S_sur_sigmoid=tensor2PIL(S_sur_sigmoid).convert('RGB')  
    S_sur=tensor2PIL(S_sur).convert('RGB')    
    ra1_feat_g = F.interpolate(ra1_feat_g, scale_factor=4, mode='bilinear')   
    ra1_feat_g=tensor2PIL(ra1_feat_g).convert('RGB')
    ra1_feat_s = -1* F.interpolate(ra1_feat_s, scale_factor=4, mode='bilinear')   
    ra1_feat_s=tensor2PIL(ra1_feat_s).convert('RGB')
    res= torch.sigmoid(S_2_pred)
    res=(res - res.min()) / (res.max() - res.min() + 1e-8)
    res=tensor2PIL(res).convert('RGB')
    S_2_pred=tensor2PIL(S_2_pred).convert('RGB')

    Guide_G=[S_edge_pred,guidance_3,guidance_4,guidance_5,guidance_g]
    width,height=Guide_G[0].size
    result1 = Image.new(Guide_G[0].mode, (width* len(Guide_G), height)) 
    for i, im in enumerate(Guide_G):
        result1.paste(im, box=(i*width,0))

    Guide_G_Sigmoid=[S_g_pred,guidance_3_sigmoid,guidance_4_sigmoid,guidance_5_sigmoid,guidance_g_sigmoid]
    width,height=Guide_G_Sigmoid[0].size
    result2 = Image.new(Guide_G_Sigmoid[0].mode, (width* len(Guide_G_Sigmoid), height)) 
    for i, im in enumerate(Guide_G_Sigmoid):
        result2.paste(im, box=(i*width,0))

    RA_Feat_G=[imgs,ra1_feat_g,ra2_feat_g,ra3_feat_g,ra4_feat_g]
    width,height=RA_Feat_G[0].size
    result3 = Image.new(RA_Feat_G[0].mode, (width* len(RA_Feat_G), height)) 
    for i, im in enumerate(RA_Feat_G):
        result3.paste(im, box=(i*width,0))

    Pred=[res,S_2_pred,S_3_pred,S_4_pred,S_5_pred]
    width,height=Pred[0].size
    result4 = Image.new(Pred[0].mode, (width* len(Pred), height)) 
    for i, im in enumerate(Pred):
        result4.paste(im, box=(i*width,0))

    RA_Feat_S=[imgs,ra1_feat_s,ra2_feat_s,ra3_feat_s,ra4_feat_s]
    width,height=RA_Feat_S[0].size
    result5 = Image.new(RA_Feat_S[0].mode, (width* len(RA_Feat_S), height)) 
    for i, im in enumerate(RA_Feat_S):
        result5.paste(im, box=(i*width,0))

    Guide_S_Sigmoid=[S_sur_pred,S_sur_sigmoid,g2_sur_sigmoid,g3_sur_sigmoid,guidance_s_sigmoid]
    width,height=Guide_S_Sigmoid[0].size
    result6 = Image.new(Guide_S_Sigmoid[0].mode, (width* len(Guide_S_Sigmoid), height)) 
    for i, im in enumerate(Guide_S_Sigmoid):
        result6.paste(im, box=(i*width,0))

    Guide_S=[S_edge_pred,S_sur,g2_sur,g3_sur,guidance_s]
    width,height=Guide_S[0].size
    result7 = Image.new(Guide_S[0].mode, (width* len(Guide_S), height)) 
    for i, im in enumerate(Guide_S):
        result7.paste(im, box=(i*width,0))

    result = Image.new(Pred[0].mode, (width* len(Pred), height * 7)) 

    result.paste(result1, box=(0,0*height))
    result.paste(result2, box=(0, 1*height))
    result.paste(result3, box=(0, 2*height))
    result.paste(result4, box=(0, 3*height))
    result.paste(result5, box=(0, 4*height))
    result.paste(result6, box=(0, 5*height))
    result.paste(result7, box=(0, 6*height))
    result.save('perd_feat.png')
    return 0