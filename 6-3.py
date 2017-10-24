import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data
import time
import matplotlib.pyplot as plt
import numpy as np

start_time = time.time()

mnist = input_data.read_data_sets("./mnist/data/", one_hot=True)

X = tf.placeholder(tf.float32, [None, 784])
Y = tf.placeholder(tf.float32, [None, 10])

keep_prob = tf.placeholder(tf.float32)

W1 = tf.Variable(tf.random_normal([784, 256], stddev=0.01))
L1 = tf.nn.relu(tf.matmul(X, W1))
L1 = tf.nn.dropout(L1, keep_prob)
# tf.layers.batch_normalization(L1, training=False)

W2 = tf.Variable(tf.random_normal([256, 256], stddev=0.01))
L2 = tf.nn.relu(tf.matmul(L1, W2))
L2 = tf.nn.dropout(L2, keep_prob)
# tf.layers.batch_normalization(L2, training=False)

W3 = tf.Variable(tf.random_normal([256, 10], stddev=0.01))
model = tf.matmul(L2, W3)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
    logits=model, labels=Y
))
optimizer = tf.train.AdamOptimizer(0.001).minimize(cost)

sess = tf.Session()
saver = tf.train.Saver(tf.global_variables())

ckpt = tf.train.get_checkpoint_state('./model')
if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
    saver.restore(sess, ckpt.model_checkpoint_path)
else:
    sess.run(tf.global_variables_initializer())

    batch_size = 100
    total_batch = int(mnist.train.num_examples / batch_size)

    for epoch in range(30):
        total_cost = 0

        for i in range(total_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)

            _, cost_val = sess.run([optimizer, cost],
                                   feed_dict={X: batch_xs, Y: batch_ys, keep_prob: 0.8})
            total_cost += cost_val

        print('Epoch:', '%04d' % (epoch+1),
              'Avg. cost =', '{:.3f}'.format(total_cost / total_batch))

    print('최적화 완료!')

    saver.save(sess, './model/mnist.ckpt')

is_correct = tf.equal(tf.argmax(model, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))
print('정확도:', sess.run(accuracy,
                       feed_dict={X: mnist.test.images,
                                  Y: mnist.test.labels,
                                  keep_prob: 1}))

labels = sess.run(model,
                  feed_dict={X: mnist.test.images,
                             Y: mnist.test.labels,
                             keep_prob: 1})

fig = plt.figure()

for i in range(10):
    subplot = fig.add_subplot(2, 5, i+1)
    subplot.set_xticks([])
    subplot.set_yticks([])
    subplot.set_title('%d' % np.argmax(labels[i+20]))
    subplot.imshow(mnist.test.images[i+20].reshape((28, 28)),
                   cmap=plt.cm.gray_r)

plt.show()
print('--- %s seconds ---' % (time.time() - start_time))

