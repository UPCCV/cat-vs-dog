import tensorflow as tf
import tensorflow.contrib.slim as slim
import tensorflow.contrib.slim.nets as nets
from tensorflow.python.framework import graph_util
import os
import numpy as np

VGG_19_MODEL_DIR = 'models/vgg_19.ckpt'
BATCH_SIZE = 32
IMAGE_SIZE=224
CLASSES=2
EPOCHES=10

MODEL_PREFIX="models/vgg19.ckpt"

def get_files(dir="../data/train"):
    files=os.listdir(dir)
    paths=[]
    labels=[]
    for file in files:
        paths.append(dir+"/"+file)
        items=file.split(".")
        if items[0]=="cat":
            labels.append(0)
        else:
            labels.append(1)
    return paths,labels

def load_batch():
    images,labels=get_files()
    images_tensor=tf.convert_to_tensor(images,dtype=tf.string)
    labels_tensor=tf.convert_to_tensor(labels,dtype=tf.int64)
    img_path,label=tf.train.slice_input_producer([images_tensor,labels_tensor],num_epochs=EPOCHES)
    image_content=tf.read_file(img_path)
    image=tf.image.decode_jpeg(image_content,channels=3)
    image=tf.image.resize_images(image,(IMAGE_SIZE,IMAGE_SIZE))
    image = image * 1.0 / 127.5 - 1.0
    label = tf.one_hot(label, CLASSES)

    #image=tf.image.convert_image_dtype(image,tf.float32)
    #image = tf.image.resize_image_with_crop_or_pad(image,227,227)
    # image=tf.cast(image,tf.float32)
    # image = tf.image.per_image_standardization(image)
    #image_batch,label_batch=tf.train.shuffle_batch([image,label],batch_size=BATCH_SIZE,capacity=100,min_after_dequeue=60)
    image_batch, label_batch = tf.train.batch([image, label], batch_size=BATCH_SIZE, capacity=100,num_threads=4)
    return image_batch,label_batch

def load_test_batch():
    images,labels=get_files()
    images_tensor=tf.convert_to_tensor(images,dtype=tf.string)
    labels_tensor=tf.convert_to_tensor(labels,dtype=tf.int64)
    img_path,label=tf.train.slice_input_producer([images_tensor,labels_tensor],num_epochs=1)
    image_content=tf.read_file(img_path)
    image=tf.image.decode_jpeg(image_content,channels=3)
    image=tf.image.resize_images(image,(IMAGE_SIZE,IMAGE_SIZE))
    image = image * 1.0 / 127.5 - 1.0
    label = tf.one_hot(label, CLASSES)
    image_batch, label_batch = tf.train.batch([image, label], batch_size=BATCH_SIZE, capacity=100,num_threads=4)
    return image_batch,label_batch


def AlexNet(inputs,n_classes=2):
    with slim.arg_scope([slim.conv2d,slim.fully_connected],
                            activation_fn=tf.nn.relu,
                            weights_initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.01),
                            weights_regularizer=slim.l2_regularizer(0.0005)):
        net=slim.conv2d(inputs,96,[11,11],padding="VALID",stride=4,scope="conv_1")
        net=slim.max_pool2d(net,[3,3],2,scope="pool_1")
        net=slim.conv2d(net,256,[5,5],scope="conv_2")
        net=slim.max_pool2d(net,[3,3],2,scope="pool_2")
        net=slim.repeat(net,2,slim.conv2d,384,[3,3],scope="conv_3_4")
        net=slim.conv2d(net,256,[3,3],scope="conv_5")
        net=slim.max_pool2d(net,[3,3],2,scope="pool_3")
        net=slim.flatten(net,scope="flatten")
        net=slim.fully_connected(net,4096,scope="fc_6")
        net=slim.dropout(net,keep_prob=0.5)
        net=slim.fully_connected(net,n_classes,scope="prob",activation_fn=None)
        return net

def get_accuracy(logits, labels):
    correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    return accuracy

def train():
    #images = tf.placeholder(tf.float32, [BATCH_SIZE, 224, 224, 3])
    #labels = tf.placeholder(tf.float32, [BATCH_SIZE, 2])
    global_step = tf.train.get_or_create_global_step()
    image_batch, label_batch = load_batch()
    images=tf.placeholder(shape=[None,IMAGE_SIZE,IMAGE_SIZE,3],dtype=tf.float32,name="image_tensor")
    labels=tf.placeholder(shape=[None,CLASSES],dtype=tf.float32,name="labels")
    with tf.name_scope("image"):
        tf.summary.image("input_images",images)
    keep_prob = tf.placeholder(tf.float32,name="keep_prob")
    logits, _ = nets.vgg.vgg_19(inputs=images, num_classes=CLASSES, dropout_keep_prob=keep_prob, is_training=True)
    variables_to_restore = slim.get_variables_to_restore(exclude=['vgg_19/fc8'])
    restorer = tf.train.Saver(variables_to_restore)
    with tf.name_scope('cross_entropy'):
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels))

    tf.summary.scalar('cross_entropy', loss)
    learning_rate = 1e-4
    train_op = tf.train.AdamOptimizer(learning_rate).minimize(loss,global_step=global_step)
    with tf.name_scope('accuracy'):
        accuracy = get_accuracy(logits, labels)
    tf.summary.scalar('accuracy', accuracy)

    merged = tf.summary.merge_all()
    saver = tf.train.Saver()
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        if os.path.exists(VGG_19_MODEL_DIR):
            restorer.restore(sess, VGG_19_MODEL_DIR)
        writer = tf.summary.FileWriter("log", sess.graph)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        try:
            while not coord.should_stop():
                image, label = sess.run([image_batch, label_batch])
                gl,_=sess.run([global_step,train_op],feed_dict={images: image, labels: label,keep_prob: 0.5})
                if gl%10==0:
                    loss_value,acc, msp = sess.run([loss, accuracy,merged],
                           feed_dict={images: image, labels: label, keep_prob: 0.5})
                    writer.add_summary(msp, gl)
                    print(gl, loss_value, acc)
                if gl % 1000 == 0:
                    saver.save(sess, MODEL_PREFIX,global_step=global_step)
        except tf.errors.OutOfRangeError:
            print("read data end")
        finally:
            coord.request_stop()
            saver.save(sess,MODEL_PREFIX,global_step=global_step)
        coord.join(threads)

def evaluate():
    image_batch, label_batch = load_test_batch()
    saver = tf.train.import_meta_graph("models/vgg19.ckpt-1001.meta")
    labels = tf.placeholder(shape=[None, CLASSES], dtype=tf.float32, name="labels")
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        sess.run(tf.local_variables_initializer())
        sess.run(tf.global_variables_initializer())
        ckpt = tf.train.latest_checkpoint("models")
        if ckpt:
            saver.restore(sess, ckpt)
            print("Restroing form " + ckpt)
            images = sess.graph.get_tensor_by_name("image_tensor:0")
            keep_prob = sess.graph.get_tensor_by_name("keep_prob:0")
            logits = sess.graph.get_tensor_by_name("vgg_19/fc8/squeezed:0")
            with tf.name_scope('accuracy'):
                accuracy = get_accuracy(logits, labels)
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)
            try:
                all_acc=[]
                index=0
                while not coord.should_stop():
                    image, label = sess.run([image_batch, label_batch])
                    acc=sess.run([accuracy],feed_dict={images: image, labels: label,keep_prob: 0.5})
                    print(index,acc)
                    all_acc.append(acc)
                    index+=1
            except tf.errors.OutOfRangeError:
                print("read data end")
            finally:
                coord.request_stop()
            print("mean:"+str(np.mean(all_acc)))
            coord.join(threads)

def getnodenames():
    for op in tf.get_default_graph().get_operations():
        print(op.name)

def convert_img_to_tensor(sess,imgpath):
    image_content = tf.read_file(imgpath)
    image = tf.image.decode_jpeg(image_content, channels=3)
    image = tf.image.resize_images(image, (IMAGE_SIZE, IMAGE_SIZE))
    image = image * 1.0 / 127.5 - 1.0
    image = tf.expand_dims(image, 0)
    img=sess.run(image)
    return img

def test():
    imgpath="../data/train/cat.0.jpg"
    saver=tf.train.import_meta_graph("models/vgg19.ckpt-1001.meta")
    #getnodenames()
    with tf.Session() as sess:
        #saver.restore(sess,"models/vgg19.ckpt-1001")
        ckpt = tf.train.latest_checkpoint("models")
        if ckpt:
            saver.restore(sess, ckpt)
            print("Restroing form " + ckpt)
        writer=tf.summary.FileWriter("log",sess.graph)
        images =sess.graph.get_tensor_by_name("image_tensor:0")
        keep_prob=sess.graph.get_tensor_by_name("keep_prob:0")
        logits=sess.graph.get_tensor_by_name("vgg_19/fc8/squeezed:0")
        img=convert_img_to_tensor(sess,imgpath)
        l=sess.run([logits],feed_dict={images:img,keep_prob:1.0})
        print(l)

def freeze_graph():
    saver = tf.train.import_meta_graph("models/vgg19.ckpt-1001.meta", clear_devices=True)
    with tf.Session() as sess:
        ckpt = tf.train.latest_checkpoint("models")
        if ckpt:
            saver.restore(sess, ckpt)
            print("Restroing form " + ckpt)
            output_graph = "models/frozen_model.pb"
            output_node_names="vgg_19/fc8/squeezed"
            output_graph_def = graph_util.convert_variables_to_constants(sess, tf.get_default_graph().as_graph_def(),
                                                 output_node_names.split(","))
            with tf.gfile.GFile(output_graph, 'wb')as f:
                f.write(output_graph_def.SerializeToString())

def load_graph(model_file):
    graph = tf.Graph()
    graph_def = tf.GraphDef()

    with open(model_file, "rb") as f:
        graph_def.ParseFromString(f.read())
    with graph.as_default():
        tf.import_graph_def(graph_def)
    return graph

def load_pb_test():
    imgpath = "../data/train/cat.0.jpg"
    graph=load_graph("models/frozen_model.pb")
    images = graph.get_tensor_by_name("import/image_tensor:0")
    keep_prob = graph.get_tensor_by_name("import/keep_prob:0")
    logits = graph.get_tensor_by_name("import/vgg_19/fc8/squeezed:0")
    with tf.Session(graph=graph)as sess:
        img=convert_img_to_tensor(sess,imgpath)
        p=sess.run([logits],feed_dict={images:img,keep_prob:1.0})
        print(p)

if __name__=="__main__":
    #train()
    evaluate()
    #test()
    #freeze_graph()
    #load_pb_test()