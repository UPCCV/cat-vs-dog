#coding=utf-8
import os,shutil
import tensorflow as tf

tf.app.flags.DEFINE_string('train_dir', '../data/train2', 'The directory of train dataset.')

FLAGS = tf.app.flags.FLAGS

dst_dir="errors"
# 无法识别的四种类型，需要从训练集中删去
no_cat_or_dog = ["dog.1773", "cat.11184", "cat.4338", "cat.10712", "dog.10747", 
                 "dog.10237", "dog.10801", "cat.5418", "cat.5351", "dog.2614", 
                 "dog.4367", "dog.5604", "dog.8736", "dog.9517", "dog.11299"]
both_cat_and_dog = ["cat.5583", "cat.3822", "cat.9250", "cat.10863", "cat.4688",
                    "cat.11724", "cat.11222", "cat.10266", "cat.9444", "cat.7920", 
                    "cat.7194", "cat.5355", "cat.724", "dog.2461", "dog.8507"]
hard_to_recognition = ["cat.6402", "cat.6987", "dog.11083", "cat.12499", "cat.2753", 
                       "dog.669", "cat.2150", "dog.5490", "cat.12493", "cat.7703", 
                       "dog.3430", "cat.2433", "cat.3250", "dog.4386", "dog.12223", 
                       "cat.9770", "cat.9626", "cat.6649", "cat.5324", "cat.335",
                       "cat.10029", "dog.1835", "dog.3322", "dog.3524", "dog.6921",
                       "dog.7413", "dog.10939", "dog.11248"]
too_abstract_image = ["dog.8898", "dog.1895", "dog.4690", "dog.1308", "dog.10190",
                      "dog.10161"]

# 标注反了，需要修改标注
label_reverse = ["cat.4085", "cat.12272", "dog.2877", "dog.4334", "dog.10401", "dog.10797", 
                 "dog.11731"]

cat_index = 12500
dog_index = 12500

def mklink():
    train_dir="../data/train"
    files=os.listdir(train_dir)
    train_cat=filter(lambda x:x[:3]=="cat",files)
    train_dog=filter(lambda x:x[:3]=="dog",files)
    for file in train_cat:
        #os.syslink(train_dir+"/"+file,"train2/cat/"+file)
        shutil.copyfile(train_dir+"/"+file,"../data/train2/cat/"+file)
    for file in train_dog:
        #os.syslink(train_dir+"/"+file,"train2/dog/"+file)
        shutil.copyfile(train_dir+"/"+file,"../data/train2/dog/"+file)

def mv_image_in_list(name_list):
    for name in name_list:
        cls=name.split(".")[0]
        path = os.path.join(FLAGS.train_dir, cls,name + ".jpg")
        dst_path=dst_dir+"/"+name + ".jpg"
        if os.path.exists(path):
            shutil.move(path,dst_path)

def change_label_in_list(name_list):
    global cat_index
    global dog_index
    for name in name_list:
        cls=name.split(".")[0]
        path = os.path.join(FLAGS.train_dir,cls, name + ".jpg")
        if os.path.exists(path):
            if "cat" in name:
                new_name = "dog." + str(dog_index)
                dog_index += 1
            else:
                new_name = "cat." + str(cat_index)
                cat_index += 1
            new_path = os.path.join(FLAGS.train_dir,new_name.split(".")[0],new_name + ".jpg")
            os.rename(path, new_path)

if __name__=="__main__":
    mklink()
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)
    #mv_image_in_list(no_cat_or_dog)
    #mv_image_in_list(both_cat_and_dog)
    #mv_image_in_list(hard_to_recognition)
    #mv_image_in_list(too_abstract_image)
    change_label_in_list(label_reverse)