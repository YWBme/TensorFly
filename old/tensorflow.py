import tensorflow as tf
matrix1 = tf.constant([[3., 3.]])
matrix2 = tf.constant([[2.], [2.]])
product = tf.matmul(matrix1, matrix2)

sess = tf.Session()
result = sess.run(product)
print(result)
sess.close()



'''第二种方法'''
# Enter an interactive TensorFlow Session.
import tensorflow as tf

sess = tf.InteractiveSession()
x = tf.Variable([1.0, 2.0])
a = tf.constant([3.0, 3.0])
# Initialize 'x' using the run() method of its initializer op.
x.initializer.run()
# Add an op to subtract 'a' from 'x'. Run it and print the result
sub = tf.sub(x, a)

print(sub.eval())  # ==> [−2. −1.]
# Close the Session when we're done.
sess.close()


state = tf.Variable(0, name="counter")
# Create an Op to add one to `state`.
one = tf.constant(1)

new_value = tf.add(state, one)
update = tf.assign(state, new_value)
# Variables must be initialized by running an `init` Op after having # launched the graph. We first have to add the `init` Op to the
init_op = tf.initialize_all_variables()
# Launch the graph and run the ops.
with tf.Session() as sess:

  sess.run(init_op)
  print(sess.run(state))
  for _ in range(3):
    sess.run(update)
    print(sess.run(state))


'''   '''
input1 = tf.placeholder(tf.float32)
input2 = tf.placeholder(tf.float32)
output = tf.mul(input1, input2)
with tf.Session() as sess:
  print(sess.run([output], feed_dict={input1: [7.], input2: [2.]}))

  # output:
  # [array([ 14.], dtype=float32)]

import tensorflow