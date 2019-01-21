from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf

mnist = input_data.read_data_sets('MNIST_data', one_hot = True)

sess = tf.InteractiveSession()

x = tf.placeholder(tf.float32, shape=[None, 784]) #784 = 28 X 28
y_ = tf.placeholder(tf.float32, shape=[None, 10])

W = tf.Variable(tf.zeros([784, 10])) # 784 * 10 행
b = tf.Variable(tf.zeros([10])) # 10차원 벡터

sess.run(tf.global_variables_initializer())

y = tf.nn.softmax(tf.matmul(x, W) + b)

cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1])) #최소 비용 함


train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)


for i in range(1000):
    batch = mnist.train.next_batch(50) ##각 각의 훈련 단계에서 50개의 훈련 샘플 추출
    train_step.run(feed_dict={x:batch[0], y_:batch[1]})


correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
# argmax(y, 1) : 모델이 입력을 받고 가장 그럴듯하다고 생각되는 레이블
# argmax(y_, 1) : 실제 레이블
# 일치하는지 확인

accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
# cast : 자료형 변환

print(accuracy.eval(feed_dict={x: mnist.test.images, y_:mnist.test.labels}))
