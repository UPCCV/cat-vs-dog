import os
import tensorflow as tf
from PIL import Image
from tqdm import tqdm

def _int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def convert2tfrecords(imgdir="train",args=None):
    recordfilename=imgdir+".tfrecords"
    writer= tf.python_io.TFRecordWriter(recordfilename)
    files=os.listdir(imgdir)
    labels={}
    for file in tqdm(files):
        filepath=imgdir+"/"+file
        cls=file.split(".")[0]
        if cls in labels:
            label=labels[cls]
        else:
            labels[cls]=len(labels)
        label=labels[cls]
        img=Image.open(filepath)
        #img= img.resize(args.resize)
        example=tf.train.Example(features=tf.train.Features(feature={
            "filepath":_bytes_feature(bytes(filepath, encoding = "utf8")),
            "height":_int64_feature(img.height),
            "width":_int64_feature(img.width),
            "label":_int64_feature(label),
            "image_raw":_bytes_feature(img.tobytes())
            }))
        writer.write(example.SerializeToString())
    writer.close()

if __name__=="__main__":
    convert2tfrecords()