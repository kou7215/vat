import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import glob
import os
import argparse
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

parser = argparse.ArgumentParser()
# data args
parser.add_argument("--load_model", default=False, type=bool,help="train is False, test is True")
parser.add_argument("--load_model_path", default='./logs/model', help="path for checkpoint")
parser.add_argument("--save_dir", default='./logs', help="path for save the model and logs")
# train conditions args
parser.add_argument("--batch_size", type=int, default=32, help="batch size")
parser.add_argument("--Ip", type=int, default=1, help="Power iteration num")
parser.add_argument("--eps", type=float, default=8.0, help="Power iteration num")
parser.add_argument("--xi", type=float, default=1e-6, help="hyper parameter scale of initialised d")
parser.add_argument("--epoch", type=int, default=500, help="epoch")
parser.add_argument("--print_loss_freq", type=int, default=500, help="print loss EPOCH frequency")
parser.add_argument("--labeled_sample_num", type=int, default=100, help="number of labeled samples")
parser.add_argument("--gpu_config", default=-1, help="0: gpu0, 1: gpu1, -1: both gpu")
a = parser.parse_args()

for k,v in a._get_kwargs():
    print(k, '=', v)

config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))
if a.gpu_config == '0':
    config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True, visible_device_list='0'))
elif a.gpu_config == '1':
    config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True, visible_device_list='1'))

def Accuracy(y,label):
    correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(label,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    return accuracy

def preprocess(image):
    with tf.name_scope("preprocess"):
        # [0, 1] => [-1, 1]
        return image * 2 - 1

def deprocess(image):
    with tf.name_scope("deprocess"):
        # [-1, 1] => [0, 1]
        return (image + 1) / 2

def batchnorm(input):
    with tf.variable_scope('batchnorm',reuse=tf.AUTO_REUSE):
        input = tf.identity(input)
        channels = input.get_shape()[-1]    # output channel size
        print(input.get_shape())
        offset = tf.get_variable("offset_{}".format(channels), [channels], dtype=tf.float32, initializer=tf.zeros_initializer())
        scale = tf.get_variable("scale_{}".format(channels), [channels], dtype=tf.float32, initializer=tf.random_normal_initializer(1.0, 0.02))
        mean, variance = 0., 1.
        # MLP
        if len(input.get_shape()) == 2: # rank 2 tensor
            mean, variance = tf.nn.moments(input, axes=[0], keep_dims=False)
        # CNN
        if len(input.get_shape()) == 4: # rank 4 tensor
            mean, variance = tf.nn.moments(input, axes=[0, 1, 2], keep_dims=False)
        variance_epsilon = 1e-5
        normalized = tf.nn.batch_normalization(input, mean, variance, offset, scale, variance_epsilon=variance_epsilon)
        return normalized

def Get_normalized_vector(d):
    with tf.name_scope('get_normalized_vec'):
        d /= (1e-12 + tf.reduce_sum(tf.abs(d), axis=[1,2,3], keep_dims=True))
        d /= tf.sqrt(1e-6 + tf.reduce_sum(tf.pow(d, 2.), axis=[1,2,3], keep_dims=True))
        return d

def KL_divergence(p, q):
    # KL = ∫p(x)log(p(x)/q(x))dx
    #    = ∫p(x)(log(p(x)) - log(q(x)))dx
    kld = tf.reduce_mean(tf.reduce_sum(p * (tf.log(p + 1e-14) - tf.log(q + 1e-14)), axis=[1]))
    return kld
    

def CNN(x):
    stride = 2
    # dropout_rate
    dropout_rate = 1.0
    if a.load_model is not True:
        dropout_rate = 0.5
    with tf.variable_scope('CNN',reuse=tf.AUTO_REUSE):
        with tf.variable_scope('layer1'):
            # [n,28,28,1] -> [n,14,14,64]
            padded = tf.pad(x, [[0, 0], [1, 1], [1, 1], [0, 0]], mode="CONSTANT")
            w = tf.get_variable(name='w1',shape=[4,4,1,64], dtype=tf.float32, initializer=tf.random_normal_initializer(0,0.02))
            b = tf.get_variable(name='b1',shape=[64], dtype=tf.float32, initializer=tf.constant_initializer(0.0))
            out = tf.nn.leaky_relu(batchnorm(tf.nn.conv2d(padded, w, [1,stride,stride,1], padding='VALID') + b), 0.2)
        with tf.variable_scope('layer2'):
            # [n,14,14,64] -> [n,7,7,128] ([n,7x7x128])
            padded = tf.pad(out, [[0, 0], [1, 1], [1, 1], [0, 0]], mode="CONSTANT")
            w = tf.get_variable(name='w2',shape=[4,4,64,128], dtype=tf.float32, initializer=tf.random_normal_initializer(0,0.02))
            b = tf.get_variable(name='b2',shape=[128], dtype=tf.float32, initializer=tf.constant_initializer(0.0))
            out = tf.nn.leaky_relu(batchnorm(tf.nn.conv2d(padded, w, [1,stride,stride,1], padding='VALID') + b), 0.2)
            out = tf.reshape(out, [-1,7*7*128])
        with tf.variable_scope('layer3'):
            # [n,7*7*128] -> [n,1024]
            w = tf.get_variable(name='w3',shape=[7*7*128,1024], dtype=tf.float32, initializer=tf.random_normal_initializer(0,0.02))
            b = tf.get_variable(name='b3',shape=[1024], dtype=tf.float32, initializer=tf.constant_initializer(0.0))
            out = tf.nn.leaky_relu(batchnorm(tf.matmul(out, w) + b), 0.2)
            # dropout
            out = tf.nn.dropout(out, dropout_rate)
        with tf.variable_scope('layer4'):
            # [n,1024] -> [n,10]
            w = tf.get_variable(name='w4',shape=[1024,10], dtype=tf.float32, initializer=tf.random_normal_initializer(0,0.02))
            b = tf.get_variable(name='b4',shape=[10], dtype=tf.float32, initializer=tf.constant_initializer(0.0))
            out = tf.nn.softmax(batchnorm(tf.matmul(out, w) + b))
            return out

def Generate_perturbation(x):
    d = tf.random_normal(shape=tf.shape(x))
    for i in range(a.Ip):
        d = a.xi * Get_normalized_vector(d)
        p = CNN(x)
        q = CNN(x + d)
        d_kl = KL_divergence(p,q)
        grad = tf.gradients(d_kl, [d], aggregation_method=2)[0] # d(d_kl)/d(d)
        d = tf.stop_gradient(grad)
    return a.eps * Get_normalized_vector(d)
    
def Get_VAT_loss(x,r):
    with tf.name_scope('Get_VAT_loss'):
        p = tf.stop_gradient(CNN(x))
        q = CNN(x + r)
        loss = KL_divergence(p,q)
        return tf.identity(loss, name='vat_loss')

# ----------------- Define Model -----------------
with tf.name_scope('Model'):
    with tf.name_scope('Inputs'):
        # load inputs
        # for train
        labeled_x_vec = tf.placeholder(shape=[a.batch_size,784], dtype=tf.float32, name='labeled_x')
        unlabeled_x_vec = tf.placeholder(shape=[a.batch_size,784], dtype=tf.float32, name='unlabeled_x')
        label = tf.placeholder(shape=[a.batch_size,10], dtype=tf.float32, name='label')
        # for validation
        val_x_vec = tf.placeholder(shape=[None,784], dtype=tf.float32, name='val_x')
        val_l = tf.placeholder(shape=[None,10], dtype=tf.float32, name='val_l')
        # preprocess
        labeled_x = preprocess(tf.reshape(labeled_x_vec, [-1,28,28,1]))
        unlabeled_x = preprocess(tf.reshape(unlabeled_x_vec, [-1,28,28,1]))
        val_x = preprocess(tf.reshape(val_x_vec, [-1,28,28,1]))
    
    with tf.name_scope('Generate_perturbation'):
        # generate perturbation
        r_adv_labeled_x = Generate_perturbation(labeled_x)
        r_adv_unlabeled_x = Generate_perturbation(unlabeled_x)
        # add perturbation onto x
        labeled_x_r   = labeled_x + r_adv_labeled_x
        unlabeled_x_r = unlabeled_x + r_adv_unlabeled_x

    with tf.name_scope('CNN_outputs'):
        labeled_out = CNN(labeled_x)
        unlabeled_out = CNN(unlabeled_x)
        val_out = CNN(val_x)


# -------------------- Define Loss & ACC --------------------
with tf.name_scope('loss'):
    with tf.name_scope('cross_entropy_loss'):
        cross_entropy_loss = -tf.reduce_mean(tf.reduce_sum(label*tf.log(labeled_out), axis=[1]))
    
    with tf.name_scope('conditional_entropy_loss'):
        with tf.name_scope('labeled_c_entropy'):
            labeled_c_entropy_loss = -tf.reduce_mean(tf.reduce_sum(labeled_out*tf.log(labeled_out), axis=[1]))
            unlabeled_c_entropy_loss = -tf.reduce_mean(tf.reduce_sum(unlabeled_out*tf.log(unlabeled_out), axis=[1]))
        c_entropy_loss = labeled_c_entropy_loss + unlabeled_c_entropy_loss
    
    with tf.name_scope('vat_loss'):
        vat_loss = Get_VAT_loss(labeled_x,r_adv_labeled_x) + Get_VAT_loss(unlabeled_x,r_adv_unlabeled_x)

    losses = cross_entropy_loss + c_entropy_loss + vat_loss

with tf.name_scope('Accuracy'):
    with tf.name_scope('train_acc'):
        train_acc = Accuracy(labeled_out, label)

    with tf.name_scope('val_acc'):
        val_acc = Accuracy(val_out, val_l)


# -------------------- Define Optimizers --------------
with tf.name_scope('Optimizer'):
    trainable_vars_list = [var for var in tf.trainable_variables()]
    print(trainable_vars_list)
    adam = tf.train.AdamOptimizer(0.0002,0.5)
    gradients_vars = adam.compute_gradients(losses, var_list=trainable_vars_list)
    train_op = adam.apply_gradients(gradients_vars)

# ---------------------------- Define Summaries ------------------------- #
with tf.name_scope('summary'):
    with tf.name_scope('Input_summaies'):
        tf.summary.image('labeled_x', tf.image.convert_image_dtype(deprocess(labeled_x), dtype=tf.uint8, saturate=True))
        tf.summary.image('unlabeled_x', tf.image.convert_image_dtype(deprocess(unlabeled_x), dtype=tf.uint8, saturate=True))
        tf.summary.image('labeled_x_r', tf.image.convert_image_dtype(deprocess(labeled_x_r), dtype=tf.uint8, saturate=True))
        tf.summary.image('unlabeled_x_r', tf.image.convert_image_dtype(deprocess(unlabeled_x_r), dtype=tf.uint8, saturate=True))

    with tf.name_scope('Loss_summary'):
        tf.summary.scalar('cross_entropy_loss', cross_entropy_loss)
        tf.summary.scalar('conditional_entropy_loss', c_entropy_loss)
        tf.summary.scalar('vat_loss', vat_loss)
        tf.summary.scalar('total_loss', losses)

    with tf.name_scope('Accuracy_summary'):
        tf.summary.scalar('train_acc', train_acc)
        tf.summary.scalar('val_acc', val_acc)
    
    for var in tf.trainable_variables():
        tf.summary.histogram(var.op.name + '/Variable_histogram', var)

    for grad, var in gradients_vars:
        tf.summary.histogram(var.op.name + '/Gradients', grad)

parameter_count = tf.reduce_sum([tf.reduce_prod(tf.shape(v)) for v in tf.trainable_variables()])

global_step = tf.get_variable('global_step',shape=[],initializer=tf.zeros_initializer(),trainable=False)
global_step = tf.assign(global_step, global_step+1)

# ------------------ Session -------------------
init = tf.global_variables_initializer()
saver = tf.train.Saver(tf.global_variables())
with tf.Session(config=config) as sess:
    if a.load_model is not True:
        sess.run(init)
    
        # mkdir if not exist directory
        if not os.path.exists(a.save_dir): # NOT CHANGE
            os.mkdir(a.save_dir)
            os.mkdir(os.path.join(a.save_dir,'summary'))
            os.mkdir(os.path.join(a.save_dir,'variables'))
            os.mkdir(os.path.join(a.save_dir,'model'))
        
        # remove old summary if already exist
        if tf.gfile.Exists(os.path.join(a.save_dir,'summary')):    # NOT CHANGE
            tf.gfile.DeleteRecursively(os.path.join(a.save_dir,'summary'))
        
        # merging summary & set summary writer
        merged = tf.summary.merge_all()
        summary_writer = tf.summary.FileWriter(os.path.join(a.save_dir,'summary'), graph=sess.graph)
        
        # load MNIST
        x_train = mnist.train.images
        y_train = mnist.train.labels
        x_val   = mnist.validation.images
        y_val   = mnist.validation.labels
        # split labeled data and unlabeled data
        all_idx = np.random.RandomState(seed=114514).permutation(len(x_train))
        labeled_idx = all_idx[:a.labeled_sample_num]
        unlabeled_idx = all_idx[a.labeled_sample_num:]
        x_train_labeled, y_train_labeled = x_train[labeled_idx], y_train[labeled_idx]
        x_train_unlabeled, y_train_unlabeled = x_train[unlabeled_idx], y_train[unlabeled_idx]

        # train
        iteration_per_epoch = int(len(x_train) / a.batch_size)
        iteration_num = iteration_per_epoch * a.epoch
        for eph in range(a.epoch):
            for itr in range(iteration_per_epoch):
                # batch indice
                batch_labeled_idx = np.random.choice(len(x_train[labeled_idx]),a.batch_size,replace=False)
                batch_unlabeled_idx = np.random.choice(len(x_train[unlabeled_idx]),a.batch_size,replace=False)
                # sample batch
                x_train_labeled_bch, y_train_labeled_bch = x_train_labeled[batch_labeled_idx], y_train_labeled[batch_labeled_idx]
                x_train_unlabeled_bch = x_train_unlabeled[batch_unlabeled_idx]
                # train operation
                sess.run(train_op, feed_dict={labeled_x_vec:x_train_labeled_bch,
                                              label:y_train_labeled_bch,
                                              unlabeled_x_vec:x_train_unlabeled_bch})
                step = sess.run(global_step)
                if step % a.print_loss_freq == 0:
                    print('step', step)
                    print('loss: ', sess.run(losses, feed_dict={labeled_x_vec:x_train_labeled_bch,
                                                                 label:y_train_labeled_bch,
                                                                 unlabeled_x_vec:x_train_unlabeled_bch}))
                    print()
                    summary_writer.add_summary(sess.run(merged,feed_dict={labeled_x_vec:x_train_labeled_bch,
                                                                          label:y_train_labeled_bch,
                                                                          unlabeled_x_vec:x_train_unlabeled_bch,
                                                                          val_x_vec:x_val,
                                                                          val_l:y_val}), step)

            if eph % (a.epoch/5) == 0: 
                saver.save(sess, a.save_dir + "/model/model.ckpt")    

        saver.save(sess, a.save_dir + "/model/model.ckpt")    
    # test
    else:
        checkpoint = tf.train.latest_checkpoint(a.load_model_path + '/model')
        saver.restore(sess, checkpoint)
        print('Loaded model from {}'.format(a.load_model_path))

