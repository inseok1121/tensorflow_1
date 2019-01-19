import tensorflow as tf

sess = tf.InteractiveSession()

x = tf.Variable([1., 2.])
y = tf.constant([[1.,2.]])
a = tf.constant([3., 3.])
b = tf.constant([[3.],[3.]])
x.initializer.run()

sub = tf.sub(x,a)
print(sub.eval())

mul = tf.matmul(y,b)
print(mul.eval())

sess.close()
