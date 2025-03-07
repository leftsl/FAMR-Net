import shutil
import pandas as pd
import os
from tqdm import tqdm
import time
# 读取CSV文件，这里假设CSV文件是逗号分隔的
metadatafile = 'EN\HAM10000_metadata'  # 请确保文件名和路径正确
df = pd.read_csv(metadatafile)

# 图片原始路径和目标路径
originalpath = 'EN\HAM10000\images'
targetpath = 'EN\HAM10000\organizedimages'

# 确保目标路径存在，如果不存在，则创建它
if not os.path.exists(targetpath):
    os.makedirs(targetpath)

# 遍历CSV文件中的每一行
for index, row in tqdm(df.iterrows(),total=df.shape[0]):
    # 获取图片名称和标签
    imagename = row['image_id']  # 假设CSV文件中有一个名为'imageid'的列
    label = row['dx']  # 假设CSV文件中有一个名为'dx'的列
    # 构建原始图片的完整路径
    originalimagepath = os.path.join(originalpath, imagename+'.jpg')
    
    # 检查文件是否存在，如果不存在，则跳过
    if not os.path.exists(originalimagepath):
        print(f"不存在的文件: {originalimagepath}")
        continue
    
    # 确保标签文件夹在目标路径中存在，如果不存在，则创建它
    labelfolder = os.path.join(targetpath, label)
    if not os.path.exists(labelfolder):
        os.makedirs(labelfolder)
    
    # 构建目标图片的完整路径
    targetimagepath = os.path.join(labelfolder, imagename+'.jpg')
    
    # 移动图片
    shutil.copy(originalimagepath, targetimagepath)

print("图片分类完成。")
