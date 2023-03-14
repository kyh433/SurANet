import scipy.io as scio
import os
import pandas as pd
import json
import numpy as np
import sys
from shutil import copyfile
class Logger(object):
    def __init__(self, filename='default.log', stream=sys.stdout):
        self.terminal = stream
        self.log = open(filename, 'w')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass

def mkdir(path):
	folder = os.path.exists(path)
	if not folder:                   #判断是否存在文件夹如果不存在则创建为文件夹
		os.makedirs(path)            #makedirs 创建文件时如果路径不存在会创建这个路径

def MaxMinNormalization(x):
    """[0,1] normaliaztion"""
    x = (x - np.min(x)) / (np.max(x) - np.min(x))
    return x


Model="SINet_V2_convnext_L2_edgetask_fusion"
# Model="SINet_V2_convnext_small"
DataPath = './Dataset/TestDataset/'
Datasets = ['CAMO', 'CHAMELEON','COD10K']
mkdir(f"./Statistics/{Model}/")
sys.stdout = Logger(f"./Statistics/{Model}/{Model}_statistic.log", sys.stdout)
print("************************************************************")
dataFile = f'./EvaluationResults/{Model}/COD10K-mat/{Model}_img.mat'
data = scio.loadmat(dataFile)

caminfo=pd.read_excel(io='/media/perry/E/DataSets/COD10K-v2/COD10K-v2/Info/Statistics-CAM.xlsx')
caminfo=json.loads(caminfo.to_json(orient="index",force_ascii=False))
for i in range(len(caminfo)):
    ImgName=caminfo[str(i)]["ImgName"]
    caminfo[ImgName] = caminfo.pop(str(i))
    del caminfo[ImgName]["ImgName"]
gtPath = DataPath + Datasets[2] + '/GT/'
imgFiles=[]  
imgFiles_temp = os.listdir(gtPath)
for filename in imgFiles_temp:
    if os.path.splitext(filename)[1] == '.png':
        imgFiles.append(filename)

#建立字典库
img_dict=[]
category_dict=[]
subcategory_dict=[]
#添加指标和标识符等特征
for i,imgfile in enumerate(imgFiles):
    category=imgfile.split("-")[3]
    if category not in category_dict:
        category_dict.append(category)

    subcategory=imgfile.split("-")[5]
    if subcategory not in subcategory_dict:
        subcategory_dict.append(subcategory)

    CAMinfo=caminfo[imgfile[:-4]]
    CAMkey=list(CAMinfo.keys())
    CAM=np.zeros(7)
    for j in range(len(CAMinfo)):
        if CAMinfo[CAMkey[j]] == None:
            CAM[j]=0
        else:
            CAM[j]=1
        Smeasure=data["Smeasure"][0,i]
        wFmeasure=data["wFmeasure"][0,i]
        MAE=data["MAE"][0,i]
        Mean_Emeasure=np.mean(data["threshold_Emeasure"],1)[i]
    img_dict.append({"imgfile":imgfile,"category":category,"subcategory":subcategory,"CAM":CAM,\
                     "Smeasure":Smeasure,"wFmeasure":wFmeasure,"MAE":MAE,"Mean_Emeasure":Mean_Emeasure})

category_dict.sort()
subcategory_dict.sort()
for img_item in img_dict:
    category=np.zeros(5)
    subcategory=np.zeros(69)

    cat_ind=category_dict.index(img_item["category"])
    scat_ind=subcategory_dict.index(img_item["subcategory"])
    category[cat_ind]=1
    subcategory[scat_ind]=1

    img_item["category"]=category
    img_item["subcategory"]=subcategory

#多元线性回归
df = pd.DataFrame(img_dict)
x=[np.array(df["category"].tolist()),np.array(df["subcategory"].tolist()),np.array(df["CAM"].tolist())]
y=[np.array(df["Smeasure"].tolist()),np.array(df["Mean_Emeasure"].tolist()),np.array(df["wFmeasure"].tolist()),np.array(df["MAE"].tolist())]
x_lable=["category","subcategory","CAM"]
y_lable=["Smeasure","Mean_Emeasure","wFmeasure","MAE"]
# Res=[]
for i in range(len(x)):
    # res=[]
    for j in range(len(y)):
        # 标准化
        y[j]=MaxMinNormalization(y[j])
        data_x=pd.DataFrame(x[i])
        data_y=pd.Series(y[j])
        # 查看是否符合高斯分布 画散点图和直方图(非高斯分布不适用z-score标准化)
        # import matplotlib.pyplot as plt
        # plt.switch_backend('agg')
        # fig = plt.figure(figsize = (10,6))
        # ax1 = fig.add_subplot(2,1,1)  # 创建子图1
        # ax1.scatter(data_y.index, data_y.values)
        # plt.grid()
        # ax2 = fig.add_subplot(2,1,2)  # 创建子图2
        # data_y.hist(bins=30,alpha = 0.5,ax = ax2)
        # data_y.plot(kind = 'kde', secondary_y=True,ax = ax2)
        # plt.grid()
        # plt.savefig(f"./Statistics/{Model}/{Model}_"+y_lable[j]+"_"+ str(j) +".jpg")

        
    #     res.append(np.linalg.lstsq(data_x,data_y,rcond=None))
    # Res.append(res)
        theta = np.dot(np.dot(np.linalg.inv(np.dot(data_x.T, data_x)), data_x.T), data_y)
        pred_y = np.dot(theta, data_x.T)
        print('x_lable',x_lable[i],'y_lable',y_lable[j])
        print('权重',theta) # 权重
        loss = np.mean(np.array(data_y-pred_y))  # 残差
        print('残差',loss)            
        print('')

#四种指标排序并进行基础统计

x_lable=["category","subcategory","CAM"]
y_lable=["Smeasure","Mean_Emeasure","wFmeasure","MAE"]
# 对于四种指标
for i in range(len(y_lable)):
    df=df.sort_values(by=[y_lable[i]])
    stastic_data=[df["imgfile"].tolist(),np.array(df[y_lable[i]].tolist()),np.array(df["category"].tolist()),np.array(df["subcategory"].tolist()),np.array(df["CAM"].tolist())]
    category_num=np.zeros(5)
    subcategory_num=np.zeros(69)
    cam_num=np.zeros(7)
    errfile_list=[]
    if y_lable[i]!="MAE": #MAE越小越好，取最大10%统计错误样本特征
        # 逐个累加标签信息
        for k in range(df.shape[0]):
            if k < np.round(df.shape[0]*0.1):
                errfile_list.append([stastic_data[0][k],k])
                category_num +=stastic_data[2][k]
                subcategory_num +=stastic_data[3][k]
                cam_num +=stastic_data[4][k]
            else:
                break
    else:
        for k in range(df.shape[0]):
            if k > np.round(df.shape[0]*0.9):
                errfile_list.append([stastic_data[0][k],k])
                category_num +=stastic_data[2][k]
                subcategory_num +=stastic_data[3][k]
                cam_num +=stastic_data[4][k]

    print('y_lable',y_lable[i])
    print('category_num',category_num)
    print('subcategory_num',subcategory_num)
    print('cam_num',cam_num)
    print('')
    # 写入错误图片
    for camind,camtype in enumerate(CAMkey):
        errorimg_path=f"./Statistics/errorimg/{Model}/{Model}_10par_{y_lable[i]}/"+str(camind)+"_"+camtype+"/"
        mkdir(errorimg_path)
        for errfile,k in errfile_list:
                if stastic_data[4][k][camind]==1:
                    predimgpath=f"res/{Model}/COD10K/"+errfile
                    copyfile(predimgpath, errorimg_path+errfile)



print("************************************************************")














# img_dict.sort(key=lambda k: k["A_MAE"])
# for i in sorted (img_dict) : 
#         print ((i, key_value[i]), end =" ") 
