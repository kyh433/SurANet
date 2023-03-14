# import cv2
# gtpath="/media/perry/E/Model_e/SINet-V2-main/Dataset/TestDataset/Customize/GT/COD10K-CAM-1-Aquatic-3-Crab-31.png"
# gt=cv2.imread(gtpath)
# # gt1 = cv2.blur(gt,(11,11))
# gt1 = cv2.blur(gt,(500,500))
# gt2=gt1-gt
# cv2.imwrite("gt.png", gt1)

from PIL import Image, ImageFilter
import numpy as np
gtpath="/media/perry/E/Model_e/SINet-V2-main/Dataset/TestDataset/Customize/GT/COD10K-CAM-1-Aquatic-3-Crab-31.png"
gt=Image.open(gtpath) 
# image = Image.fromarray(a)
filtered = gt.filter(ImageFilter.GaussianBlur(radius=50))
filtered = np.array(filtered.getdata()).reshape(filtered.size[1], filtered.size[0])
gt = np.array(gt.getdata()).reshape(gt.size[1], gt.size[0])
zeroshape=np.zeros_like(gt)
newgt=np.maximum(zeroshape,filtered-gt)
newgt = Image.fromarray(np.uint8(newgt))
newgt.save("./gt.png")   
