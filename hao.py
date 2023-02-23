from PIL import Image,ImageDraw
import json
import os
import pandas as pd
import numpy as np

yuan = Image.open('./A0001_15/masks/A0001_15_2.jpg').convert("1")
yuan_array = np.array(yuan)

img = Image.open('./A0001_15/masks/A0001_15_1.jpg').convert("1")
img_array = np.array(img)
# color = [255, 0, 0]  # 类型  颜色 type_info
# location = np.all((img_array == color), axis=2) + 0  # 在第二个维度上取与
# # print(location)
pos = np.argwhere(img_array == 1)
print(len(pos))
for xy in pos:
    # print(xy)
    # print(new_img_array[xy[0],xy[1]])
    yuan_array[xy[0], xy[1]] = 255
    # print(new_img_array[xy[0], xy[1]])
img_final = Image.fromarray(yuan_array)
print(img_final.mode)
img_final.save('hhh.jpg')