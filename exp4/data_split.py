# rainbow_yu exp4.data_split 🐋✨

import os
import shutil

def data_split(srcfile,dstfile):
    if not os.path.isfile(srcfile):
        print("src not exist!")
    else:
        fpath,fname=os.path.split(dstfile)    #分离文件名和路径
        if not os.path.exists(fpath):
            os.makedirs(fpath)                #创建路径
        shutil.move(srcfile,dstfile)          #移动文件
test_rate=0.2
img_num=1250
test_num=int(img_num*test_rate)

import random
test_index = random.sample(range(0, img_num), test_num)
file_path=r"E:\python_project\datasets\cats_and_dogs\PetImages"
tr="train"
te="test"
cat="Cat"
dog="Dog"

if __name__ == "__main__":
    srcfile=os.path.join(file_path,tr)
    dstfile=os.path.join(file_path,te)
    #将上述index中的文件都移动到/test/Cat/和/test/Dog/下面去。
    for i in range(len(test_index)):
        #移动猫
        srcfile=os.path.join(file_path,tr,cat,str(test_index[i])+".jpg")
        dstfile=os.path.join(file_path,te,cat,str(test_index[i])+".jpg")
        data_split(srcfile,dstfile)
        #移动狗
        srcfile=os.path.join(file_path,tr,dog,str(test_index[i])+".jpg")
        dstfile=os.path.join(file_path,te,dog,str(test_index[i])+".jpg")
        data_split(srcfile,dstfile)