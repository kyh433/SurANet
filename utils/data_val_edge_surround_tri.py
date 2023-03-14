import os
from PIL import Image
import torch.utils.data as data
import torchvision.transforms as transforms
import random
import numpy as np
from PIL import ImageEnhance, ImageFilter

import scipy as sp
import scipy.ndimage
import torch


def fill(test_array,h_max=0):
    input_array = np.copy(test_array) 
    el = sp.ndimage.generate_binary_structure(2,2).astype(np.int)
    inside_mask = sp.ndimage.binary_erosion(~np.isnan(input_array), structure=el)
    output_array = np.copy(input_array)
    output_array[inside_mask]=h_max
    output_old_array = np.copy(input_array)
    output_old_array.fill(0)   
    el = sp.ndimage.generate_binary_structure(2,1).astype(np.int)
    while not np.array_equal(output_old_array, output_array):
        output_old_array = np.copy(output_array)
        output_array = np.maximum(input_array,sp.ndimage.grey_erosion(output_array,footprint=el))
    return output_array

# several data augumentation strategies
def cv_random_flip(img, label,edge,surround):
    # left right flip
    flip_flag = random.randint(0, 1)
    if flip_flag == 1:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
        label = label.transpose(Image.FLIP_LEFT_RIGHT)
        edge = edge.transpose(Image.FLIP_LEFT_RIGHT)
        surround = surround.transpose(Image.FLIP_LEFT_RIGHT)
    return img, label, edge,surround


def randomCrop(image, label, edge,surround):
    border = 30
    image_width = image.size[0]
    image_height = image.size[1]
    crop_win_width = np.random.randint(image_width - border, image_width)
    crop_win_height = np.random.randint(image_height - border, image_height)
    random_region = (
        (image_width - crop_win_width) >> 1, (image_height - crop_win_height) >> 1, (image_width + crop_win_width) >> 1,
        (image_height + crop_win_height) >> 1)
    return image.crop(random_region), label.crop(random_region), edge.crop(random_region),surround.crop(random_region)

def randomRotation(image, label, edge,surround):
    mode = Image.BICUBIC
    if random.random() > 0.8:
        random_angle = np.random.randint(-15, 15)
        image = image.rotate(random_angle, mode)
        label = label.rotate(random_angle, mode)
        edge = edge.rotate(random_angle, mode)
        surround = surround.rotate(random_angle, mode)
    return image, label, edge,surround


def colorEnhance(image):
    bright_intensity = random.randint(5, 15) / 10.0
    image = ImageEnhance.Brightness(image).enhance(bright_intensity)
    contrast_intensity = random.randint(5, 15) / 10.0
    image = ImageEnhance.Contrast(image).enhance(contrast_intensity)
    color_intensity = random.randint(0, 20) / 10.0
    image = ImageEnhance.Color(image).enhance(color_intensity)
    sharp_intensity = random.randint(0, 30) / 10.0
    image = ImageEnhance.Sharpness(image).enhance(sharp_intensity)
    return image


def randomGaussian(image, mean=0.1, sigma=0.35):
    def gaussianNoisy(im, mean=mean, sigma=sigma):
        for _i in range(len(im)):
            im[_i] += random.gauss(mean, sigma)
        return im

    img = np.asarray(image)
    width, height = img.shape
    img = gaussianNoisy(img[:].flatten(), mean, sigma)
    img = img.reshape([width, height])
    return Image.fromarray(np.uint8(img))


def randomPeper(img):
    img = np.array(img)
    noiseNum = int(0.0015 * img.shape[0] * img.shape[1])
    for i in range(noiseNum):

        randX = random.randint(0, img.shape[0] - 1)

        randY = random.randint(0, img.shape[1] - 1)

        if random.randint(0, 1) == 0:

            img[randX, randY] = 0

        else:

            img[randX, randY] = 255
    return Image.fromarray(img)

def getsurround(gt,radius=50):
    filtered = gt.filter(ImageFilter.GaussianBlur(radius=radius))
    filtered = np.array(filtered.getdata()).reshape(filtered.size[1], filtered.size[0])
    gt = np.array(gt.getdata()).reshape(gt.size[1], gt.size[0])
    zeroshape=np.zeros_like(gt)
    surround=np.maximum(zeroshape,filtered-gt)
    surround = Image.fromarray(np.uint8(surround))
    return surround

# dataset for training
class PolypObjDataset(data.Dataset):
    def __init__(self, image_root, gt_root,edge_root, trainsize):
        self.trainsize = trainsize
        # get filenames
        self.images = [image_root + f for f in os.listdir(image_root) if f.endswith('.jpg')]
        self.gts = [gt_root + f for f in os.listdir(gt_root) if f.endswith('.jpg')
                    or f.endswith('.png')]
        self.edges = [edge_root + f for f in os.listdir(edge_root) if f.endswith('.jpg')
                    or f.endswith('.png')]
        # self.grads = [grad_root + f for f in os.listdir(grad_root) if f.endswith('.jpg')
        #               or f.endswith('.png')]
        # self.depths = [depth_root + f for f in os.listdir(depth_root) if f.endswith('.bmp')
        #                or f.endswith('.png')]
        # sorted files
        self.images = sorted(self.images)
        self.gts = sorted(self.gts)
        self.edges = sorted(self.edges)
        # self.grads = sorted(self.grads)
        # self.depths = sorted(self.depths)
        # filter mathcing degrees of files
        self.filter_files()
        # transforms
        self.img_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        self.gt_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize)),
            transforms.ToTensor()])
        self.edge_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize)),
            transforms.ToTensor()])
        self.surround_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize)),
            transforms.ToTensor()])
        # get size of dataset
        self.size = len(self.images)
    def __getitem__(self, index):
        # read imgs/gts/grads/depths
        image = self.rgb_loader(self.images[index])
        gt = self.binary_loader(self.gts[index])
        edge = self.binary_loader(self.edges[index])
        surround=getsurround(gt,radius=50)
        # data augumentation
        image, gt, edge,surround = cv_random_flip(image, gt, edge,surround)
        image, gt, edge,surround = randomCrop(image, gt, edge,surround)
        image, gt, edge,surround = randomRotation(image, gt, edge,surround)
        image = colorEnhance(image)

        surround_arr = np.array(surround)
        surround_arr = np.where(surround_arr[...,:] < surround_arr.mean(), 0, 255) 
        
        gt_arr = np.array(gt)
        background_arr_=-1 *(surround_arr+gt_arr) + 255
        background_arr=fill(background_arr_)  #有些图有点bug，中间有缝需要填充一下

        surround_arr = Image.fromarray(np.uint8(surround_arr))
        gt_arr = Image.fromarray(np.uint8(gt_arr))
        background_arr = Image.fromarray(np.uint8(background_arr))

        # surround_arr.save('surround.png')
        # gt_arr.save('gt.png')
        # background_arr.save('background.png')

        gt_ = self.gt_transform(gt_arr)
        surround_ = self.surround_transform(surround_arr)
        background_ = self.surround_transform(background_arr)
        tri=torch.cat((background_,gt_,surround_),0)
        trimask= torch.clone(background_)
        trimask[gt_.bool()]=2
        trimask[surround_.bool()]=3
        # trimask=torch.squeeze(trimask,0) 
        # gt_=transforms.ToPILImage()(gt_).convert('L')
        # gt_.save('gt_.png')
        # surround_=transforms.ToPILImage()(surround_).convert('L')
        # surround_.save('surround_.png')
        # background_=transforms.ToPILImage()(background_).convert('L')
        # background_.save('background_.png')

        gt = randomPeper(gt)
        edge = randomPeper(edge)
        surround = randomPeper(surround)
        image = self.img_transform(image)
        gt = self.gt_transform(gt)
        edge = self.edge_transform(edge)
        surround = self.surround_transform(surround)

        return image, gt,edge,surround,tri,trimask

    def filter_files(self):
        assert len(self.images) == len(self.gts) and len(self.gts) == len(self.images) and len(self.gts) == len(self.edges)
        images = []
        gts = []
        edges=[]
        for img_path, gt_path,edge_path in zip(self.images, self.gts,self.edges):
            img = Image.open(img_path)
            gt = Image.open(gt_path)
            edge = Image.open(edge_path)
            if img.size == gt.size and gt.size == edge.size:
                images.append(img_path)
                gts.append(gt_path)
                edges.append(edge_path)
        self.images = images
        self.gts = gts
        self.edges = edges

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')

    def __len__(self):
        return self.size


# dataloader for training
def get_loader(image_root, gt_root,edge_root, batchsize, trainsize,
               shuffle=True, num_workers=12, pin_memory=True):
    dataset = PolypObjDataset(image_root, gt_root,edge_root, trainsize)
    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batchsize,
                                  shuffle=shuffle,
                                  num_workers=num_workers,
                                  pin_memory=pin_memory)
    return data_loader


# test dataset and loader
class test_dataset:
    def __init__(self, image_root, gt_root, testsize):
        self.testsize = testsize
        self.images = [image_root + f for f in os.listdir(image_root) if f.endswith('.jpg') or f.endswith('.png')]
        self.gts = [gt_root + f for f in os.listdir(gt_root) if f.endswith('.tif') or f.endswith('.png')]
        self.images = sorted(self.images)
        self.gts = sorted(self.gts)
        self.transform = transforms.Compose([
            transforms.Resize((self.testsize, self.testsize)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        self.gt_transform = transforms.ToTensor()
        self.size = len(self.images)
        self.index = 0

    def load_data(self):
        image = self.rgb_loader(self.images[self.index])
        image = self.transform(image).unsqueeze(0)

        gt = self.binary_loader(self.gts[self.index])

        name = self.images[self.index].split('/')[-1]

        image_for_post = self.rgb_loader(self.images[self.index])
        image_for_post = image_for_post.resize(gt.size)

        if name.endswith('.jpg'):
            name = name.split('.jpg')[0] + '.png'

        self.index += 1
        self.index = self.index % self.size

        return image, gt, name, np.array(image_for_post)

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')

    def __len__(self):
        return self.size
