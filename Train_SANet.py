# author: Daniel-Ji (e-mail: gepengai.ji@gmail.com)
# data: 2021-01-16
import os
import torch
import torch.nn.functional as F
import numpy as np
from datetime import datetime
from torchvision.utils import make_grid
# from lib.Network_Res2Net_GRA_NCD import Network
from lib.Network_Convnext_doublefusion_sursup_compare_detachRFB import Network
# from utils.data_val import get_loader, test_dataset
# from utils.data_val_edge_surround import get_loader, test_dataset
from utils.data_val_edge_surround_tri import get_loader, test_dataset
from utils.utils import clip_gradient, adjust_lr
from tensorboardX import SummaryWriter
import logging
import torch.backends.cudnn as cudnn
import torch.nn as nn
import random

def inplace_relu(m):
    classname = m.__class__.__name__
    if classname.find('ReLU') != -1:
        m.inplace=True


def EuclideanDistances(a,b):
    eps=1e-16
    sq_a = a**2
    sum_sq_a = torch.sum(sq_a,dim=1).unsqueeze(1)  # m->[m, 1]
    sq_b = b**2
    sum_sq_b = torch.sum(sq_b,dim=1).unsqueeze(0)  # n->[1, n]
    bt = b.t()
    L2=sum_sq_a+sum_sq_b-2*a.mm(bt)
    zeros=torch.zeros_like(L2)
    L2=torch.max(L2,zeros)
    return torch.sqrt(L2+eps)

def _get_triplet_mask(labels):
    '''
       得到一个3D的mask [a, p, n], 对应triplet（a, p, n）是valid的位置是True
       ----------------------------------
       Args:
          labels: 对应训练数据的labels, shape = (batch_size,)
       
       Returns:
          mask: 3D,shape = (batch_size, batch_size, batch_size)
    
    '''
    
    # 初始化一个二维矩阵，坐标(i, j)不相等置为1，得到indices_not_equal
    indices_equal = torch.eye(labels.shape[0]).type(torch.bool).cuda()
    indices_not_equal = torch.logical_not(indices_equal)
    # 因为最后得到一个3D的mask矩阵(i, j, k)，增加一个维度，则 i_not_equal_j 在第三个维度增加一个即，(batch_size, batch_size, 1), 其他同理
    i_not_equal_j = torch.unsqueeze(indices_not_equal, 2) 
    i_not_equal_k = torch.unsqueeze(indices_not_equal, 1)
    j_not_equal_k = torch.unsqueeze(indices_not_equal, 0)
    # 想得到i!=j!=k, 三个不等取and即可, 最后可以得到当下标（i, j, k）不相等时才取True
    distinct_indices = torch.logical_and(torch.logical_and(i_not_equal_j, i_not_equal_k), j_not_equal_k)

    # 同样根据labels得到对应i=j, i!=k
    label_equal = torch.eq(torch.unsqueeze(labels, 0), torch.unsqueeze(labels, 1))
    i_equal_j = torch.unsqueeze(label_equal, 2)
    i_equal_k = torch.unsqueeze(label_equal, 1)
    valid_labels = torch.logical_and(i_equal_j, torch.logical_not(i_equal_k))
    # mask即为满足上面两个约束，所以两个3D取and
    mask = torch.logical_and(distinct_indices, valid_labels)
    return mask


def batch_all_triplet_loss(labels, embeddings, margin=0.2):
    '''
       triplet loss of a batch
       -------------------------------
       Args:
          labels:     标签数据，shape = （batch_size,）
          embeddings: 提取的特征向量， shape = (batch_size, vector_size)
          margin:     margin大小， scalar
          
       Returns:
          triplet_loss: scalar, 一个batch的损失值
          fraction_postive_triplets : valid的triplets占的比例
    '''
    
    # 得到每两两embeddings的距离，然后增加一个维度，一维需要得到（batch_size, batch_size, batch_size）大小的3D矩阵
    # 然后再点乘上valid 的 mask即可
    pairwise_dis = EuclideanDistances(embeddings.t(),embeddings.t())
    anchor_positive_dist = torch.unsqueeze(pairwise_dis, 2)
    assert anchor_positive_dist.shape[2] == 1, "{}".format(anchor_positive_dist.shape)
    anchor_negative_dist = torch.unsqueeze(pairwise_dis, 1)
    assert anchor_negative_dist.shape[1] == 1, "{}".format(anchor_negative_dist.shape)
    triplet_loss = anchor_positive_dist - anchor_negative_dist + margin
    
    mask = _get_triplet_mask(labels)
    mask = mask.float()
    triplet_loss = torch.mul(mask, triplet_loss)
    zeros = torch.zeros_like (triplet_loss)
    triplet_loss = torch.max(triplet_loss, zeros)
    
    # 计算valid的triplet的个数，然后对所有的triplet loss求平均
    valid_triplets = torch.gt(triplet_loss, 1e-16).float()
    num_positive_triplets = torch.sum(valid_triplets)
    num_valid_triplets = torch.sum(mask)
    fraction_postive_triplets = num_positive_triplets / (num_valid_triplets + 1e-16)
    
    triplet_loss = torch.sum(triplet_loss) / (num_positive_triplets + 1e-16)
    return triplet_loss, fraction_postive_triplets

def triplet_loss_temp(anchor, positive, negative, margin = 0.2):
    anchor,positive,negative=anchor.permute(1,0),positive.permute(1,0),negative.permute(1,0)
    pos_dist = EuclideanDistances(anchor,positive)
    pos_dist_=pos_dist.mean()
    pos_dist_flat=torch.flatten(pos_dist, start_dim=0, end_dim=-1)
    neg_dist = EuclideanDistances(anchor,negative)
    neg_dist_flat=torch.flatten(neg_dist, start_dim=0, end_dim=-1)
    neg_dist_=neg_dist.mean()
    loss = F.relu(pos_dist_ - neg_dist_ + margin)
    return loss.mean()

def triplet_loss(anchor, positive, negative, margin = 0.2):
    anchor,positive,negative=anchor.permute(1,0),positive.permute(1,0),negative.permute(1,0)
    pos_dist = EuclideanDistances(anchor,positive)
    neg_dist = EuclideanDistances(anchor,negative)
    anchor_positive_dist = torch.unsqueeze(pos_dist, 2)
    assert anchor_positive_dist.shape[2] == 1, "{}".format(anchor_positive_dist.shape)
    anchor_negative_dist = torch.unsqueeze(neg_dist, 1)
    assert anchor_negative_dist.shape[1] == 1, "{}".format(anchor_negative_dist.shape)
    triplet_loss = anchor_positive_dist - anchor_negative_dist + margin

    zeros = torch.zeros_like (triplet_loss)
    triplet_loss = torch.max(triplet_loss, zeros)

    valid_triplets = torch.gt(triplet_loss, 1e-16).float()
    num_positive_triplets = torch.sum(valid_triplets)
    num_valid_triplets = triplet_loss.shape[0]*triplet_loss.shape[1]*triplet_loss.shape[2]
    # fraction_postive_triplets = num_positive_triplets / (num_valid_triplets + 1e-16)
    
    triplet_loss = torch.sum(triplet_loss) / (num_positive_triplets + 1e-16)
    # return triplet_loss, fraction_postive_triplets
    return triplet_loss

def structure_loss(pred, mask):
    """
    loss function (ref: F3Net-AAAI-2020)
    """
    weit = 1 + 5 * torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduce='none')
    wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

    pred = torch.sigmoid(pred)
    inter = ((pred * mask) * weit).sum(dim=(2, 3))
    union = ((pred + mask) * weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1) / (union - inter + 1)
    return (wbce + wiou).mean()


def train(train_loader, model, optimizer, epoch, save_path, writer):
    """
    train function
    """
    global step
    model.train()
    loss_all = 0
    epoch_step = 0
    try:
        for i, (images, gts, edges,surround,tri,trimask) in enumerate(train_loader, start=1):
            # torch.cuda.empty_cache() #so sad
            optimizer.zero_grad()
            images = images.cuda()
            gts = gts.cuda()
            edges = edges.cuda()
            surround = surround.cuda()
            trimask = trimask.cuda()
            preds = model(images)
            loss_init = structure_loss(preds[0], gts) + structure_loss(preds[1], gts) + structure_loss(preds[2], gts) + structure_loss(preds[3], gts)
            loss_final = structure_loss(preds[4], gts)
            loss_edge =  structure_loss(preds[5], edges)
            loss_surround =  structure_loss(preds[6], surround) + structure_loss(preds[7], surround) + structure_loss(preds[8], surround)

            loss_tri=torch.zeros(1).cuda()
            for item in range(3):
                x_rfb_=preds[9+item]
                # import torchvision.transforms as transforms
                # gts=transforms.ToPILImage()(gts[0].cpu()).convert('L')
                # gts.save('gts.png')
                # tri_0=transforms.ToPILImage()(tri[0,0].cpu()).convert('L')
                # tri_0.save('tri_0.png')
                # tri_1=transforms.ToPILImage()(tri[0,1].cpu()).convert('L')
                # tri_1.save('tri_1.png')
                # tri_2=transforms.ToPILImage()(tri[0,2].cpu()).convert('L')
                # tri_2.save('tri_2.png')

                tri_b = F.interpolate(tri, scale_factor=(2**item)/16, mode='bilinear').bool()  # (bs, 1, 352, 352) -> (bs, 1, 22/44/, 22/44/)
                if item == 1:
                    # nn=random.randint(0,1)
                    # mm=random.randint(0,1)
                    # tri_b = tri_b[:,:,nn::2,mm::2]
                    # x_rfb_ = x_rfb_[:,:,nn::2,mm::2]
                    tri_b = tri_b[:,:,::2,::2]
                    x_rfb_=torch.cat((x_rfb_[:,:,::2,::2],x_rfb_[:,:,1::2,::2],x_rfb_[:,:,::2,1::2],x_rfb_[:,:,1::2,1::2]), 1)
                if item == 2:
                    # nn=random.randint(0,3)
                    # mm=random.randint(0,3)
                    # tri_b = tri_b[:,:,nn::4,mm::4]
                    # x_rfb_ = x_rfb_[:,:,nn::4,mm::4]
                    tri_b = tri_b[:,:,::4,::4]
                    x_rfb_=torch.cat((x_rfb_[:,:,::4,::4],x_rfb_[:,:,1::4,::4],x_rfb_[:,:,2::4,::4],x_rfb_[:,:,3::4,::4],
                                      x_rfb_[:,:,::4,1::4],x_rfb_[:,:,1::4,1::4],x_rfb_[:,:,2::4,1::4],x_rfb_[:,:,3::4,1::4],
                                      x_rfb_[:,:,::4,2::4],x_rfb_[:,:,1::4,2::4],x_rfb_[:,:,2::4,2::4],x_rfb_[:,:,3::4,2::4],
                                      x_rfb_[:,:,::4,3::4],x_rfb_[:,:,1::4,3::4],x_rfb_[:,:,2::4,3::4],x_rfb_[:,:,3::4,3::4]), 1)

                tri_b=torch.flatten(tri_b, start_dim=2, end_dim=-1)
                x_rfb_=torch.flatten(x_rfb_, start_dim=2, end_dim=-1)
                
                for bs in range(tri_b.shape[0]):
                    bg_bs=x_rfb_[bs,:,tri_b[bs,0]]
                    fg_bs=x_rfb_[bs,:,tri_b[bs,1]]
                    su_bs=x_rfb_[bs,:,tri_b[bs,2]]
                    if bg_bs.numel()!=0 and fg_bs.numel()!=0 and su_bs.numel()!=0:
                        loss_tri+=triplet_loss(su_bs,bg_bs,fg_bs)
                                        
            loss_tri=0.1*loss_tri[0]
            loss = loss_init + loss_final + loss_edge + loss_surround + loss_tri

            loss.backward()

            clip_gradient(optimizer, opt.clip)
            optimizer.step()

            step += 1
            epoch_step += 1
            loss_all += loss.data

            if i % 20 == 0 or i == total_step or i == 1:
                print('{} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], Total_loss: {:.4f} Loss1: {:.4f} Loss2: {:0.4f} Loss3: {:0.4f} Loss4: {:0.4f} Loss5: {:0.4f}'.
                      format(datetime.now(), epoch, opt.epoch, i, total_step, loss.data, loss_init.data, loss_final.data, loss_edge.data, loss_surround.data, loss_tri.data))
                logging.info(
                    '[Train Info]:Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], Total_loss: {:.4f} Loss1: {:.4f} '
                    'Loss2: {:0.4f} Loss3: {:0.4f} Loss4: {:0.4f} Loss5: {:0.4f}'.
                    format(epoch, opt.epoch, i, total_step, loss.data, loss_init.data, loss_final.data, loss_edge.data, loss_surround.data, loss_tri.data))
                # TensorboardX-Loss
                writer.add_scalars('Loss_Statistics',
                                   {'Loss_init': loss_init.data, 'Loss_final': loss_final.data, 'Loss_edge': loss_edge.data, 'Loss_surround':loss_surround.data, 'Loss_tri':loss_tri.data, 
                                    'Loss_total': loss.data},
                                   global_step=step)
                # TensorboardX-Training Data
                grid_image = make_grid(images[0].clone().cpu().data, 1, normalize=True)
                writer.add_image('RGB', grid_image, step)
                grid_image = make_grid(gts[0].clone().cpu().data, 1, normalize=True)
                writer.add_image('GT', grid_image, step)

                # TensorboardX-Outputs
                res = preds[0][0].clone()
                res = res.sigmoid().data.cpu().numpy().squeeze()
                res = (res - res.min()) / (res.max() - res.min() + 1e-8)
                writer.add_image('Pred_init', torch.tensor(res), step, dataformats='HW')
                res = preds[3][0].clone()
                res = res.sigmoid().data.cpu().numpy().squeeze()
                res = (res - res.min()) / (res.max() - res.min() + 1e-8)
                writer.add_image('Pred_final', torch.tensor(res), step, dataformats='HW')

        loss_all /= epoch_step
        logging.info('[Train Info]: Epoch [{:03d}/{:03d}], Loss_AVG: {:.4f}'.format(epoch, opt.epoch, loss_all))
        writer.add_scalar('Loss-epoch', loss_all, global_step=epoch)
        if epoch % 50 == 0:
            torch.save(model.state_dict(), save_path + 'Net_epoch_{}.pth'.format(epoch))
    except KeyboardInterrupt:
        print('Keyboard Interrupt: save model and exit.')
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        torch.save(model.state_dict(), save_path + 'Net_epoch_{}.pth'.format(epoch + 1))
        print('Save checkpoints successfully!')
        raise


def val(test_loader, model, epoch, save_path, writer):
    """
    validation function
    """
    global best_mae, best_epoch
    model.eval()
    with torch.no_grad():
        mae_sum = 0
        for i in range(test_loader.size):
            image, gt, name, img_for_post = test_loader.load_data()
            gt = np.asarray(gt, np.float32)
            gt /= (gt.max() + 1e-8)
            image = image.cuda()

            res = model(image)

            res = F.upsample(res[4], size=gt.shape, mode='bilinear', align_corners=False)
            res = res.sigmoid().data.cpu().numpy().squeeze()
            res = (res - res.min()) / (res.max() - res.min() + 1e-8)
            mae_sum += np.sum(np.abs(res - gt)) * 1.0 / (gt.shape[0] * gt.shape[1])
        mae = mae_sum / test_loader.size
        writer.add_scalar('MAE', torch.tensor(mae), global_step=epoch)
        print('Epoch: {}, MAE: {}, bestMAE: {}, bestEpoch: {}.'.format(epoch, mae, best_mae, best_epoch))
        if epoch == 1:
            best_mae = mae
        else:
            if mae < best_mae:
                best_mae = mae
                best_epoch = epoch
                torch.save(model.state_dict(), save_path + 'Net_epoch_best.pth')
                print('Save state_dict successfully! Best epoch:{}.'.format(epoch))
        logging.info(
            '[Val Info]:Epoch:{} MAE:{} bestEpoch:{} bestMAE:{}'.format(epoch, mae, best_epoch, best_mae))


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', type=int, default=100, help='epoch number')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--batchsize', type=int, default=18, help='training batch size')
    parser.add_argument('--trainsize', type=int, default=352, help='training dataset size')
    parser.add_argument('--clip', type=float, default=0.5, help='gradient clipping margin')
    parser.add_argument('--decay_rate', type=float, default=0.1, help='decay rate of learning rate')
    parser.add_argument('--decay_epoch', type=int, default=50, help='every n epochs decay learning rate')
    parser.add_argument('--load', type=str, default="./snapshot/SINet_V2_convnext_doublefusion_sursup_compare_Stack_DetachRFB/Net_epoch_best.pth", help='train from checkpoints')
    # parser.add_argument('--load', type=str, default=None, help='train from checkpoints')
    # parser.add_argument('--gpu_id', type=str, default='0', help='train use gpu')
    parser.add_argument('--train_root', type=str, default='./Dataset/TrainValDataset/',
                        help='the training rgb images root')
    parser.add_argument('--val_root', type=str, default='./Dataset/TestDataset/CAMO/',
                        help='the test rgb images root')
    parser.add_argument('--save_path', type=str,
                        default='./snapshot/SINet_V2_convnext_doublefusion_sursup_compare_Stack_DetachRFB/',
                        help='the path to save model and log')
    group_gpus = parser.add_mutually_exclusive_group()
    group_gpus.add_argument(
        '--gpu-ids',
        type=str,
        default='0,1',
        nargs='+',
        help='(Deprecated, please use --gpu-id) ids of gpus to use '
        '(only applicable to non-distributed training)')
    group_gpus.add_argument(
        '--gpu-id',
        type=str,
        # default='0',
        help='id of gpu to use '
        '(only applicable to non-distributed training)')



    opt = parser.parse_args()

    # set the device for training
    if opt.gpu_id is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_id
        print('USE GPU',opt.gpu_id)
    if opt.gpu_ids is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_ids
        print('USE GPU',opt.gpu_ids)
    cudnn.benchmark = True

    # os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'

    # build the model
    # model = Network_(channel=32).cuda()
    model = Network(channel=32).cuda()
    if opt.gpu_ids is not None:
        model = torch.nn.DataParallel(model)
        model = model.cuda()

    if opt.load is not None:
        model.load_state_dict(torch.load(opt.load))
        print('load model from ', opt.load)

    optimizer = torch.optim.Adam(model.parameters(), opt.lr)

    model.apply(inplace_relu)   #节省显存
    save_path = opt.save_path
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # load data
    print('load data...')
    train_loader = get_loader(image_root=opt.train_root + 'Imgs/',
                              gt_root=opt.train_root + 'GT/',
                              edge_root=opt.train_root + 'Edge/',
                              batchsize=opt.batchsize,
                              trainsize=opt.trainsize,
                              num_workers=4)
    val_loader = test_dataset(image_root=opt.val_root + 'Imgs/',
                              gt_root=opt.val_root + 'GT/',
                              testsize=opt.trainsize)
    total_step = len(train_loader)

    # logging
    logging.basicConfig(filename=save_path + 'log.log',
                        format='[%(asctime)s-%(filename)s-%(levelname)s:%(message)s]',
                        level=logging.INFO, filemode='a', datefmt='%Y-%m-%d %I:%M:%S %p')
    logging.info("Network-Train")
    logging.info('Config: epoch: {}; lr: {}; batchsize: {}; trainsize: {}; clip: {}; decay_rate: {}; load: {}; '
                 'save_path: {}; decay_epoch: {}'.format(opt.epoch, opt.lr, opt.batchsize, opt.trainsize, opt.clip,
                                                         opt.decay_rate, opt.load, save_path, opt.decay_epoch))

    step = 0
    writer = SummaryWriter(save_path + 'summary')
    best_mae = 1
    best_epoch = 0

    print("Start train...")
    for epoch in range(1, opt.epoch):
        cur_lr = adjust_lr(optimizer, opt.lr, epoch, opt.decay_rate, opt.decay_epoch)
        writer.add_scalar('learning_rate', cur_lr, global_step=epoch)
        train(train_loader, model, optimizer, epoch, save_path, writer)
        val(val_loader, model, epoch, save_path, writer)
