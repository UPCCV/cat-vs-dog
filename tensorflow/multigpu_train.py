import tensorflow as tf
from tensorflow.contrib import slim
from tensorflow.python.framework import graph_util
import matplotlib.pyplot as plt
import numpy as np
import os

NUM_GPUS=2
BATCH_SIZE=32
NUM_EPOCH=100

def get_files(dir="../data/train"):
    files=os.listdir(dir)
    paths=[]
    labels=[]
    for file in files:
    #for i in range(10):
        #file=files[i]
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
    input_queue=tf.train.slice_input_producer([images_tensor,labels_tensor])
    image_content=tf.read_file(input_queue[0])
    image=tf.image.decode_jpeg(image_content,channels=3)
    #image=tf.image.convert_image_dtype(image,tf.float32)
    #image=tf.image.resize_images(image,(256,256))
    image = tf.image.resize_image_with_crop_or_pad(image,227,227)
    #image=tf.cast(image,tf.float32)
    #image = tf.image.per_image_standardization(image)
    label=input_queue[1]
    label=tf.one_hot(label,2)
    image_batch,label_batch=tf.train.shuffle_batch([image,label],batch_size=BATCH_SIZE*NUM_GPUS,capacity=200,min_after_dequeue=100)
    return image_batch,label_batch

def test_loaddatasets():
    image_batch,label_batch=load_batch()
    with tf.Session() as sess:
        coord=tf.train.Coordinator()
        threads=tf.train.start_queue_runners(sess=sess,coord=coord)
        try:
            while not coord.should_stop():
                image,label=sess.run([image_batch,label_batch])
                for j in np.arange(BATCH_SIZE):
                    plt.imshow(image[j,:,:,:])
                    plt.show()
        except tf.errors.OutOfRangeError:
            print("Date load end")
        finally:
            coord.request_stop()
        threads.join(coord)

def AlexNet_by_layers(inputs):
    with slim.arg_scope([slim.conv2d,slim.fully_connected],     activation_fn=tf.nn.relu,weights_initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.01), weights_regularizer=slim.l2_regularizer(0.0005)):
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
        net=slim.fully_connected(net,2,scope="prob",activation_fn=None)
        return net

def AlexNet(inputs):
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
        net=slim.fully_connected(net,2,scope="prob",activation_fn=None)
        return net

def train():
    global_step=tf.train.get_or_create_global_step()
    image_batch,label_batch=load_batch()
    with tf.name_scope("input"):
        images=tf.placeholder(shape=[None,227,227,3],dtype=tf.float32)
        labels=tf.placeholder(shape=[None,2],dtype=tf.float32)
    with tf.name_scope("image"):
        tf.summary.image("input",images)
    with tf.name_scope("Predicts"):
        predicts=AlexNet(images)
    with tf.name_scope("Loss"):
        loss=tf.losses.softmax_cross_entropy(labels,predicts)
        tf.summary.scalar("loss",loss)
    with tf.name_scope("Optimizer"):
        optimizer=tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(loss,global_step=global_step)
    with tf.name_scope("acc"):
        correct_prediction=tf.equal(tf.argmax(predicts,1),tf.argmax(labels,1))
        accuracy=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
        tf.summary.scalar("accuracy",accuracy)
    merged_summary_op=tf.summary.merge_all()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver=tf.train.Saver()
        ckpt = tf.train.latest_checkpoint("models")
        if ckpt:
            saver.restore(sess, ckpt)
            print(ckpt)
        writer=tf.summary.FileWriter("log",sess.graph)
        saver=tf.train.Saver()
        coord=tf.train.Coordinator()
        threads=tf.train.start_queue_runners(sess=sess,coord=coord)
        try:
            while not coord.should_stop():
                image,label=sess.run([image_batch,label_batch])
                gl,loss_value,_,msp=sess.run([global_step,loss,optimizer,merged_summary_op],feed_dict={images:image,labels:label})
                writer.add_summary(msp,gl)
                print(gl,loss_value)
                if gl%100==0:
                    saver.save(sess,"models/alxenet.ckpt")        
        except tf.errors.OutOfRangeError:
            print("train end")
        finally:
            coord.request_stop()
            #saver.save(sess,"models/alexnet.ckpt")
        coord.join(threads)

def get_one_image(image_paths):
    n=len(image_paths)
    image=image.open(image_paths[0])
    plt.imshow(image)
    plt.show()
    image = image.resize([227, 227])
    image = np.array(image)
    return image
   
def test_one_image():
    image_batch,_batch=get_files()
    image=tf.placeholder(shape=[None,227,227,3],dtype=tf.float32)
    predicts=AlexNet(image)
    saver=tf.train.Saver()
    ckpt = tf.train.latest_checkpoint("models")
    if ckpt:
        saver.restore(sess, ckpt)
    with tf.Session() as sess:
        sess.run(tf.global_initialization())
        image,label=sess.run([image_batch,label_batch])
        predict=sess.run(predicts,feed_dict={images:images})

def average_gradients(tower_grads):
  average_grads = []
  for grad_and_vars in zip(*tower_grads):
    grads = []
    for g, _ in grad_and_vars:
      expanded_g = tf.expand_dims(g, 0)
      grads.append(expanded_g)
    grad = tf.concat(grads, 0)
    grad = tf.reduce_mean(grad, 0)
    v = grad_and_vars[0][1]
    grad_and_var = (grad, v)
    average_grads.append(grad_and_var)
  return average_grads

def tower_loss(images,labels,scope):
    with tf.variable_scope("AlexNet",reuse=tf.AUTO_REUSE):
        logits=AlexNet(images)
        losses=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=labels,logits=logits))
        tf.summary.scalar("loss",losses)
        return losses

def train_multigpu():
    with tf.Graph().as_default(),tf.device("/cpu:0"):
        global_step=tf.train.get_or_create_global_step()
        opt=tf.train.GradientDescentOptimizer(0.01)
        tower_grads=[]
        X=tf.placeholder(shape=[None,227,227,3],dtype=tf.float32)
        Y=tf.placeholder(shape=[None,2],dtype=tf.float32)
        tf.summary.image("image",X)
        merged_summary=tf.summary.merge_all()
        for i in range(NUM_GPUS):
            with tf.device("/gpu:%d"%i):
                with tf.name_scope("%s_%d"%("tower",i))as scope:
                    _x=X[i*BATCH_SIZE:(i+1)*BATCH_SIZE]
                    _y=Y[i*BATCH_SIZE:(i+1)*BATCH_SIZE]
                    loss=tower_loss(_x,_y,scope)
                    grads=opt.compute_gradients(loss)
                    tower_grads.append(grads)
        grads=average_gradients(tower_grads)
        train_op=opt.apply_gradients(grads)
        image_batch,label_batch=load_batch()
        with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
            sess.run(tf.global_variables_initializer())
            writer=tf.summary.FileWriter("log",sess.graph)
            saver=tf.train.Saver()
            coord=tf.train.Coordinator()
            threads=tf.train.start_queue_runners(sess=sess,coord=coord)
            try:
                while not coord.should_stop():
                    image,label=sess.run([image_batch,label_batch])
                    sess.run([merged_summary,train_op],feed_dict={X:image,Y:label})
            except tf.errors.OutOfRangeError:
                print("train end")
            finally:
                coord.request_stop()
            #saver.save(sess,"models/alexnet.ckpt")
            coord.join(threads)

if __name__=="__main__":
    #train()
    train_multigpu()