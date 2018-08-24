#coding=utf-8
import keras,os
import tensorflow as tf
import numpy as np
from keras import backend as K
from keras.models import load_model
from keras.preprocessing import image
from tensorflow.python.framework import graph_io
from keras.backend.tensorflow_backend import set_session

config = tf.ConfigProto()
config.gpu_options.allow_growth=True
set_session(tf.Session(config=config))

cls_list = ['cats', 'dogs']

#h5_model_path="model.h5"
#inputnode_name="conv2d_1_input:0"
#outputnode_name="activation_4/Softmax:0"

h5_model_path="model97.18.h5"#99.98.h5
inputnode_name="input_1:0"
outputnode_name="dense_2/Softmax:0"

pb_model_name="model.pb"

tf_model_path="./model/ckpt"

def freeze_session(session, keep_var_names=None, output_names=None, clear_devices=True):
    from tensorflow.python.framework.graph_util import convert_variables_to_constants
    graph = session.graph
    with graph.as_default():
        freeze_var_names = list(set(v.op.name for v in tf.global_variables()).difference(keep_var_names or []))
        output_names = output_names or []
        output_names += [v.op.name for v in tf.global_variables()]
        input_graph_def = graph.as_graph_def()
        if clear_devices:
            for node in input_graph_def.node:
                node.device = ""
        frozen_graph = convert_variables_to_constants(session, input_graph_def,
                                                      output_names, freeze_var_names)
        return frozen_graph


def keras2pb():
    K.set_learning_phase(0)
    net_model = load_model(h5_model_path)
    sess = K.get_session()
    frozen_graph = freeze_session(K.get_session(), output_names=[net_model.output.op.name])
    graph_io.write_graph(frozen_graph,"./", pb_model_name, as_text=False)

def keras_to_tf():
    K.set_learning_phase(0)
    net_model = load_model(h5_model_path)
    saver = tf.train.Saver()
    with K.get_session() as sess:
        K.set_learning_phase(0)
        saver.save(sess, tf_model_path)
    return True

def tf_to_graph(tf_model_path, model_in, model_out, graph_path):
    os.system('mvNCCompile {0}.meta -in {1} -on {2} -o {3}'.format(tf_model_path, model_in, model_out, graph_path))
    return True

def keras_to_graph():
    keras_to_tf()
    tf_to_graph(tf_model_path, "input_1", "dense_2/Softmax", "graph")


def loadpb_test(imgpath="../data/train/cat.0.jpg"):
    with tf.Graph().as_default():
        output_graph_def = tf.GraphDef()

        # 打开.pb模型
        with open(pb_model_name, "rb") as f:
            output_graph_def.ParseFromString(f.read())
            tensors = tf.import_graph_def(output_graph_def, name="")
            print("tensors:",tensors)

        # 在一个session中去run一个前向
        with tf.Session() as sess:
            init = tf.global_variables_initializer()
            sess.run(init)

            op = sess.graph.get_operations()

            # 打印图中有的操作
            for i,m in enumerate(op):
                print('op{}:'.format(i),m.values())

            #input_x = sess.graph.get_tensor_by_name("")  # 具体名称看上一段代码的input.name
            input_x = sess.graph.get_tensor_by_name(inputnode_name)
            #print("input_X:",input_x)

            #out_softmax = sess.graph.get_tensor_by_name("")  # 具体名称看上一段代码的output.name
            out_softmax = sess.graph.get_tensor_by_name(outputnode_name)
            #print("Output:",out_softmax)
            img=image.load_img(imgpath, target_size=(224,224))
            x = image.img_to_array(img)
            x = keras.applications.resnet50.preprocess_input(x)
            x = np.expand_dims(x, axis=0)
            pred= sess.run(out_softmax,feed_dict={input_x: x})
            print(pred)
            for i in range(len(pred[0])):
                print('    {:.3f}  {}'.format(pred[0][i], cls_list[i]))

if __name__=="__main__":
    keras2pb()
    loadpb_test()
    #keras_to_graph()