import os
import lmdb
import caffe
import numpy as np
import cv2
import random
from tqdm import tqdm
import shutil

new_width=256
new_height=256

def image2lmdb(split="train",dataroot="../data/train"):
    if not os.path.exists("lmdb"):
        os.makedirs("lmdb")
    lmdb_dir = "lmdb/"+split+"_lmdb"
    if os.path.exists(lmdb_dir):
        shutil.rmtree(lmdb_dir)
    txtpath="util/"+split+".txt"
    map_size = 10000000000
    db = lmdb.open(lmdb_dir,map_size=map_size)
    with db.begin(write=True) as txn:
        with open(txtpath) as f:
            lines = f.readlines()
            if split=="train":
                random.shuffle(lines)
            for i,line in enumerate(tqdm(lines)):
                items = line.split(" ")
                if len(items)==2:
                    filename = items[0]
                    label = items[1]
                    datum = caffe.proto.caffe_pb2.Datum()
                    img = cv2.imread(dataroot+"/"+filename)
                    if img is None:
                        print(filename + ' cannot read')
                        continue
                    img = cv2.resize(img,(new_width,new_height))
                    datum.channels = img.shape[2]
                    datum.height = img.shape[0]
                    datum.width = img.shape[1]
                    datum.data = img.tobytes()
                    datum.label = int(label)
                    str_id = '{:08}'.format(i)
                    txn.put(str_id.encode('ascii'),datum.SerializeToString())

def lmdb2image(split="train",show=True,togt=False):
    db = lmdb.open("lmdb/"+split+"_lmdb")
    txn = db.begin()
    cursor = txn.cursor()
    for key, value in tqdm(cursor):
        datum = caffe.proto.caffe_pb2.Datum()
        datum.ParseFromString(value)
        img = np.fromstring(datum.data,dtype=np.uint8)
        img = img.reshape(datum.channels,datum.height,datum.width).transpose(1,2,0).copy()
        label = datum.label 
        cv2.putText(img,key,(0,20),1,1,(255,0,0))
        cv2.putText(img,str(label),(0,40),3,1,(0,0,255))
        if show:
            cv2.imshow("img",img)
            cv2.waitKey()
        if togt:
            cv2.imwrite("gt/"+key,img)

def main():
    image2lmdb()
    image2lmdb("val")
    #lmdb2image()

if __name__=="__main__":
    main()