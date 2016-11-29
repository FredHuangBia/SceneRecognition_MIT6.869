
import read_batch as rb
import tensorflow as tf
from newpara import *

learning_rate = 0.001
training_iters = 100000
batch_size = 40
display_step = 20
val_step = 100
save_step = 10000


n_input = 126*126*3
n_classes = 100

x = tf.placeholder(tf.float32, [batch_size,126,126,3])
y = tf.placeholder(tf.int32, [batch_size])

def top_k_error(predictions, labels, k):
    in_top1 = tf.to_float(tf.nn.in_top_k(predictions, labels, k))
    num_correct = tf.reduce_sum(in_top1)
    return (batch_size - num_correct) / float(batch_size)

def conv2d(name, l_input, w):
    #tf.nn.conv2d([patch,hight,width,channel], [filter_height, filter_width, in_channels, out_channels], strides=[patch,hight,width,channel])
    return tf.nn.conv2d(l_input, w, strides=[1, 1, 1, 1], padding='SAME', name=name)

def conv2dp(name, l_input, w):
    #tf.nn.conv2d([patch,hight,width,channel], [filter_height, filter_width, in_channels, out_channels], strides=[patch,hight,width,channel])
    return tf.nn.conv2d(l_input, w, strides=[1, 2, 2, 1], padding='SAME', name=name)

def bias(name, l_input,b):
    return tf.nn.bias_add(l_input,b,name=name)

def relu(name, l_input):
    return tf.nn.relu(l_input,name=name)

def max_pool(name, l_input, k=2,s=2):
    # ksize[patch,height,width,channel], strides[patch,height,width,channel]
    return tf.nn.max_pool(l_input, ksize=[1, k, k, 1], strides=[1, s, s, 1], padding='SAME', name=name)

def avg_pool(name, l_input, k,s=1):
    # ksize[patch,height,width,channel], strides[patch,height,width,channel]
    return tf.nn.avg_pool(l_input, ksize=[1, k, k, 1], strides=[1, s, s, 1], padding='SAME', name=name)

def cbr(name, l_input, w, b):
    return relu(name,bias(name,conv2d(name, l_input,w),b))

def cpbr(name, l_input, w, b):
    return relu(name,bias(name,conv2dp(name, l_input,w),b))

def res_net(_X, _weights, _biases):

    conv1 = cpbr('conv1', _X, _weights['wc1'], _biases['bc1'])
    #[path,63,63,64]

    conv2a_2a = cbr('conv2a_2a', conv1, _weights['wc2a_2a'], _biases['bc2a_2a'])
    conv2a_2b = cbr('conv2a_2b', conv2a_2a, _weights['wc2a_2b'], _biases['bc2a_2b'])
    conv2a_2c = conv2d('conv2a_2c', conv2a_2b, _weights['wc2a_2c'])
    conv2a_2c = bias('conv2a_2c', conv2a_2c, _biases['bc2a_2c'])

    conv2a_1 = conv2d('conv2a_1', conv1, _weights['wc2a_1'])
    conv2a_1 = bias('conv2a_1', conv2a_1, _biases['bc2a_1'])

    conv2a = relu('conv2a', conv2a_1 + conv2a_2c)
    #[path,63,63,256]

    conv2b_2a = cbr('conv2b_2a', conv2a, _weights['wc2b_2a'], _biases['bc2b_2a'])
    conv2b_2b = cbr('conv2b_2b', conv2b_2a, _weights['wc2b_2b'], _biases['bc2b_2b'])
    conv2b_2c = conv2d('conv2b_2c', conv2b_2b, _weights['wc2b_2c'])
    conv2b_2c = bias('conv2b_2c', conv2b_2c, _biases['bc2b_2c'])

    conv2b = relu('conv2b', conv2a + conv2b_2c)
    #[path,63,63,256]

    conv2c_2a = cbr('conv2c_2a', conv2b, _weights['wc2c_2a'], _biases['bc2c_2a'])
    conv2c_2b = cbr('conv2c_2b', conv2c_2a, _weights['wc2c_2b'], _biases['bc2c_2b'])
    conv2c_2c = conv2d('conv2c_2c', conv2c_2b, _weights['wc2c_2c'])
    conv2c_2c = bias('conv2c_2c', conv2c_2c, _biases['bc2c_2c'])

    conv2c = relu('conv2c', conv2b + conv2c_2c)
    #[path,63,63,256]

    conv3a_2a = cpbr('conv3a_2a', conv2c, _weights['wc3a_2a'], _biases['bc3a_2a'])
    #[path,32,32,128]
    conv3a_2b = cbr('conv3a_2b', conv3a_2a, _weights['wc3a_2b'], _biases['bc3a_2b'])
    conv3a_2c = conv2d('conv3a_2c', conv3a_2b, _weights['wc3a_2c'])
    conv3a_2c = bias('conv3a_2c', conv3a_2c, _biases['bc3a_2c'])
    #[path,32,32,512]

    conv3a_1 = conv2dp('conv3a_1', conv2c, _weights['wc3a_1'])
    #[path,32,32,512]
    conv3a_1 = bias('conv3a_1', conv3a_1, _biases['bc3a_1'])

    conv3a = relu('conv3a', conv3a_1 + conv3a_2c)
    #[path,32,32,512]

    conv3b_2a = cbr('conv3b_2a', conv3a, _weights['wc3b_2a'], _biases['bc3b_2a'])
    conv3b_2b = cbr('conv3b_2b', conv3b_2a, _weights['wc3b_2b'], _biases['bc3b_2b'])
    conv3b_2c = conv2d('conv3b_2c', conv3b_2b, _weights['wc3b_2c'])
    conv3b_2c = bias('conv3b_2c', conv3b_2c, _biases['bc3b_2c'])

    conv3b = relu('conv3b', conv3a + conv3b_2c)
    #[path,32,32,512]

    conv3c_2a = cbr('conv3c_2a', conv3b, _weights['wc3c_2a'], _biases['bc3c_2a'])
    conv3c_2b = cbr('conv3c_2b', conv3c_2a, _weights['wc3c_2b'], _biases['bc3c_2b'])
    conv3c_2c = conv2d('conv3c_2c', conv3c_2b, _weights['wc3c_2c'])
    conv3c_2c = bias('conv3c_2c', conv3c_2c, _biases['bc3c_2c'])

    conv3c = relu('conv3c', conv3b + conv3c_2c)
    #[path,32,32,512]

    conv3d_2a = cbr('conv3d_2a', conv3c, _weights['wc3d_2a'], _biases['bc3d_2a'])
    conv3d_2b = cbr('conv3d_2b', conv3d_2a, _weights['wc3d_2b'], _biases['bc3d_2b'])
    conv3d_2c = conv2d('conv3d_2c', conv3d_2b, _weights['wc3d_2c'])
    conv3d_2c = bias('conv3d_2c', conv3d_2c, _biases['bc3d_2c'])

    conv3d = relu('conv3c', conv3c + conv3d_2c)
    #[path,32,32,512]

#######################################################
    conv4a_2a = cpbr('conv4a_2a', conv3d, _weights['wc4a_2a'], _biases['bc4a_2a'])
    #[path,16,16,256]
    conv4a_2b = cbr('conv4a_2b', conv4a_2a, _weights['wc4a_2b'], _biases['bc4a_2b'])
    conv4a_2c = conv2d('conv4a_2c', conv4a_2b, _weights['wc4a_2c'])
    conv4a_2c = bias('conv4a_2c', conv4a_2c, _biases['bc4a_2c'])
    #[path,16,16,1024]

    conv4a_1 = conv2dp('conv4a_1', conv3d, _weights['wc4a_1'])
    #[path,16,16,1024]
    conv4a_1 = bias('conv4a_1', conv4a_1, _biases['bc4a_1'])

    conv4a = relu('conv4a', conv4a_1 + conv4a_2c)
    #[path,16,16,1024]

    conv4b_2a = cbr('conv4b_2a', conv4a, _weights['wc4b_2a'], _biases['bc4b_2a'])
    conv4b_2b = cbr('conv4b_2b', conv4b_2a, _weights['wc4b_2b'], _biases['bc4b_2b'])
    conv4b_2c = conv2d('conv4b_2c', conv4b_2b, _weights['wc4b_2c'])
    conv4b_2c = bias('conv4b_2c', conv4b_2c, _biases['bc4b_2c'])

    conv4b = relu('conv4b', conv4a + conv4b_2c)
    #[path,16,16,1024]

    conv4c_2a = cbr('conv4c_2a', conv4b, _weights['wc4c_2a'], _biases['bc4c_2a'])
    conv4c_2b = cbr('conv4c_2b', conv4c_2a, _weights['wc4c_2b'], _biases['bc4c_2b'])
    conv4c_2c = conv2d('conv4c_2c', conv4c_2b, _weights['wc4c_2c'])
    conv4c_2c = bias('conv4c_2c', conv4c_2c, _biases['bc4c_2c'])

    conv4c = relu('conv4c', conv4b + conv4c_2c)
    #[path,16,16,1024]

    conv4d_2a = cbr('conv4d_2a', conv4c, _weights['wc4d_2a'], _biases['bc4d_2a'])
    conv4d_2b = cbr('conv4d_2b', conv4d_2a, _weights['wc4d_2b'], _biases['bc4d_2b'])
    conv4d_2c = conv2d('conv4d_2c', conv4d_2b, _weights['wc4d_2c'])
    conv4d_2c = bias('conv4d_2c', conv4d_2c, _biases['bc4d_2c'])

    conv4d = relu('conv4c', conv4c + conv4d_2c)
    #[path,16,16,1024]

    conv4e_2a = cbr('conv4e_2a', conv4d, _weights['wc4e_2a'], _biases['bc4e_2a'])
    conv4e_2b = cbr('conv4e_2b', conv4e_2a, _weights['wc4e_2b'], _biases['bc4e_2b'])
    conv4e_2c = conv2d('conv4e_2c', conv4e_2b, _weights['wc4e_2c'])
    conv4e_2c = bias('conv4e_2c', conv4e_2c, _biases['bc4e_2c'])

    conv4e = relu('conv4c', conv4d + conv4e_2c)
    #[path,16,16,1024]

    conv4f_2a = cbr('conv4f_2a', conv4e, _weights['wc4f_2a'], _biases['bc4f_2a'])
    conv4f_2b = cbr('conv4f_2b', conv4f_2a, _weights['wc4f_2b'], _biases['bc4f_2b'])
    conv4f_2c = conv2d('conv4f_2c', conv4f_2b, _weights['wc4f_2c'])
    conv4f_2c = bias('conv4f_2c', conv4f_2c, _biases['bc4f_2c'])

    conv4f = relu('conv4c', conv4e + conv4f_2c)
    #[path,16,16,1024]
####################################################
    conv5a_2a = cpbr('conv5a_2a', conv4f, _weights['wc5a_2a'], _biases['bc5a_2a'])
    #[path,8,8,512]
    conv5a_2b = cbr('conv5a_2b', conv5a_2a, _weights['wc5a_2b'], _biases['bc5a_2b'])
    conv5a_2c = conv2d('conv5a_2c', conv5a_2b, _weights['wc5a_2c'])
    conv5a_2c = bias('conv5a_2c', conv5a_2c, _biases['bc5a_2c'])
    #[path,8,8,2048]

    conv5a_1 = conv2dp('conv5a_1', conv4f, _weights['wc5a_1'])
    #[path,8,8,2048]
    conv5a_1 = bias('conv5a_1', conv5a_1, _biases['bc5a_1'])

    conv5a = relu('conv5a', conv5a_1 + conv5a_2c)
    #[path,8,8,2048]

    conv5b_2a = cbr('conv5b_2a', conv5a, _weights['wc5b_2a'], _biases['bc5b_2a'])
    conv5b_2b = cbr('conv5b_2b', conv5b_2a, _weights['wc5b_2b'], _biases['bc5b_2b'])
    conv5b_2c = conv2d('conv5b_2c', conv5b_2b, _weights['wc5b_2c'])
    conv5b_2c = bias('conv5b_2c', conv5b_2c, _biases['bc5b_2c'])

    conv5b = relu('conv5b', conv5a + conv5b_2c)
    #[path,8,8,2048]

    conv5c_2a = cbr('conv5c_2a', conv5b, _weights['wc5c_2a'], _biases['bc5c_2a'])
    conv5c_2b = cbr('conv5c_2b', conv5c_2a, _weights['wc5c_2b'], _biases['bc5c_2b'])
    conv5c_2c = conv2d('conv5c_2c', conv5c_2b, _weights['wc5c_2c'])
    conv5c_2c = bias('conv5c_2c', conv5c_2c, _biases['bc5c_2c'])

    conv5c = relu('conv5c', conv5b + conv5c_2c)
    #[path,8,8,2048]

    pool5 = avg_pool('pool5', conv5c, k=5)
    #[path,8,8,2048]


    dense1 = tf.reshape(pool5, [-1, _weights['out'].get_shape().as_list()[0]])
    #[patch,hight*width*channel]
    out = tf.matmul(dense1, _weights['out'])
    return out


pred = res_net(x, weights, biases)

cost = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(pred, y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate,epsilon=0.1).minimize(cost)

top1=top_k_error(pred,y,1)
answer=tf.argmax(pred,1)
top5=top_k_error(pred,y,5)

saver = tf.train.Saver(tf.all_variables())
init = tf.initialize_all_variables()


with tf.Session() as sess:
    sess.run(init)
    step = 1
    train = True
    if train:
        #saver.restore(sess,"resnet-27000")
        #step=27001
        while step  <= training_iters:
            batch = rb.generate_samples(batch_size)
            batch_xs = batch[0]
            batch_ys = batch[1]
            print('Start train step ',str(step))
            sess.run(optimizer, feed_dict={x: batch_xs, y: batch_ys})
            print('Finish step', str(step))
            if step % display_step == 0:

                acc1 = sess.run(top1, feed_dict={x: batch_xs, y: batch_ys})
                acc5 = sess.run(top5, feed_dict={x: batch_xs, y: batch_ys})

                loss = sess.run(cost, feed_dict={x: batch_xs, y: batch_ys})
                print( "Iter " + str(step) + ", Minibatch Loss= " + "{:.6f}".format(loss) + ", Top1= " + "{:.5f}".format(acc1)+ ", Top5= " + "{:.5f}".format(acc5))
            if step % save_step == 0:
                saver.save(sess,'newres',global_step=step)
            # if step % val_step == 0:
            #     batch2 = rb.generate_samples(batch_size)
            #     batch_xs2 = batch2[0]
            #     batch_ys2 = batch2[1]
            #     acc5 = sess.run(top5, feed_dict={x: batch_xs2, y: batch_ys2})
            #     print( "Evaluation accuracy: "+ "{:.5f}".format(acc5))
            step += 1
        print ("Optimization Finished!")

    else:
        evl_size=32
        while 1==1:
            saver.restore(sess,"resnet-30000")
            batch = rb.generate_val_samples(evl_size)
            batch_xs = batch[0]
            batch_ys = batch[1]
            # acc1 = sess.run(top1, feed_dict={x: batch_xs, y: batch_ys})
            # acc5 = sess.run(top5, feed_dict={x: batch_xs, y: batch_ys})
            # loss = sess.run(cost, feed_dict={x: batch_xs, y: batch_ys})
            # print("Top5 Error= " + "{:.5f}".format(acc5)+"    Top1 Error= " + "{:.5f}".format(acc1))
            # print("Minibatch Loss= " + "{:.6f}".format(loss))

            #acc5 = sess.run(top5, feed_dict={x: batch_xs, y: batch_ys})
            ans = sess.run(pred, feed_dict={x: batch_xs, y: batch_ys})
            print(ans)
            print("Top5 Error= " + "{:.5f}".format(acc5))
