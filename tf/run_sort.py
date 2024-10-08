import tensorflow as tf
import numpy as np
import mnist_input
import multi_mnist_cnn
from sinkhorn import gumbel_sinkhorn, sinkhorn_operator
from PIL import Image
from statistics import median
import time

import util
import random
import os

tf.compat.v1.set_random_seed(94305)
tf.compat.v1.disable_eager_execution()

random.seed(94305)

flags = tf.compat.v1.app.flags
flags.DEFINE_integer('M', 1, 'batch size')
flags.DEFINE_integer('n', 3, 'number of elements to compare at a time')
flags.DEFINE_integer('l', 4, 'number of digits')
flags.DEFINE_integer('tau', 5, 'temperature (dependent meaning)')
flags.DEFINE_string('method', 'deterministic_neuralsort',
                    'which method to use?')
flags.DEFINE_integer('n_s', 5, 'number of samples')
flags.DEFINE_integer('num_epochs', 200, 'number of epochs to train')
flags.DEFINE_float('lr', 1e-4, 'initial learning rate')

FLAGS = flags.FLAGS

n_s = FLAGS.n_s
NUM_EPOCHS = FLAGS.num_epochs
M = FLAGS.M
n = FLAGS.n
l = FLAGS.l
tau = FLAGS.tau
method = FLAGS.method
initial_rate = FLAGS.lr

experiment_id = 'sort-%s-M%d-n%d-l%d-t%d' % (method, M, n, l, tau * 10)
checkpoint_path = 'checkpoints/%s/' % experiment_id
volume_experiment_path = '/arc/%s/' % experiment_id

logfile = open('./logs/%s.log' % experiment_id, 'w')

def prnt(*args):
    print(*args)
    print(*args, file=logfile)


def digit_image_loader(image_path):
    image = Image.open(image_path)
    image = image.convert('L').resize((28, 28))
    image_array = np.array(image)
    # features = {
    #     'image': image_tensor,
    #     'label': tf.constant(0, dtype=tf.int64)  # Replace with actual label
    # }
    return image_array

# TODO: Batch sort ops
# TODO: Compare to n(n/2) stochastic to mergesort ARC training/eval
def input_generator():
    open_set_dir_path = "/arc/mnist_sort"
    open_set_files = os.listdir(open_set_dir_path)
    open_set = [np.load(os.path.join(open_set_dir_path, node)) for node in open_set_files]
    for a in range(len(open_set)):
        for b in range(a+1, len(open_set)):
            sort_inputs = [open_set[a], open_set[b]]
            sort_tensor_input = np.stack(sort_inputs)
            sort_tensor_input = np.reshape(sort_tensor_input, (1, 2, 28, 28))
            values = np.array([9, 1])
            med_val = int(median(values))
            values = np.reshape(values, (1, 2))
            med = np.array([med_val])
            arg_med = np.equal(values, med).astype('float32')
            arg_med = np.reshape(arg_med, (1, 2))
            sort_files = [open_set_files[a], open_set_files[b]]
            sort_files = np.reshape(sort_files, (1, 2))
            ret = (sort_tensor_input, med, arg_med, values, sort_files)
            yield ret

# For learned mergesort, we assume these values 
# l = 1
# n = 2
def get_sort_iterator():
    l = 1
    n = 2
    mm_data = tf.data.Dataset.from_generator(
        input_generator,
        output_signature = (
            tf.TensorSpec(shape=((1, n, l * 28, 28)), dtype=tf.float32),
            tf.TensorSpec(shape=(1), dtype=tf.float32),
            tf.TensorSpec(shape=(1, n), dtype=tf.float32),
            tf.TensorSpec(shape=(1, n), dtype=tf.float32),
            tf.TensorSpec(shape=(1, n), dtype=tf.string),
        )
    )
    mm_data.batch(1)
    mm_data = mm_data.prefetch(1)
    return tf.compat.v1.data.make_one_shot_iterator(mm_data)

train_iterator, val_iterator, test_iterator = mnist_input.get_iterators(
    l, n, 10 ** l - 1, minibatch_size=M)

false_tensor = tf.convert_to_tensor(False)
evaluation = tf.compat.v1.placeholder_with_default(false_tensor, ())
temperature = tf.cond(evaluation,
                      false_fn=lambda: tf.convert_to_tensor(
                          tau, dtype=tf.float32),
                      true_fn=lambda: tf.convert_to_tensor(
                          1e-10, dtype=tf.float32)  # simulate hard sort
                      )

volume_model_path = tf.train.latest_checkpoint(volume_experiment_path)
if volume_model_path is not None:
    prnt("Model with same parameters found in volume. Loading instead of training.")
    should_load_model_from_volume = True
    M = 1 # assume run sort
else:
    should_load_model_from_volume = False

handle = tf.compat.v1.placeholder(tf.string, ())

if should_load_model_from_volume:
    X_iterator = tf.compat.v1.data.Iterator.from_string_handle(
        handle,
        (tf.float32, tf.float32, tf.float32, tf.float32, tf.string),
        ((M, n, l * 28, 28), (M,), (M, n), (M, n), (M, n))
    )
    X, y, median_scores, true_scores, sort_files= X_iterator.get_next()
else:
    X_iterator = tf.compat.v1.data.Iterator.from_string_handle(
        handle,
        (tf.float32, tf.float32, tf.float32, tf.float32),
        ((M, n, l * 28, 28), (M,), (M, n), (M, n))
    )
    X, y, median_scores, true_scores = X_iterator.get_next()

true_scores = tf.expand_dims(true_scores, 2)
P_true = util.neuralsort(true_scores, 1e-10)

if method == 'vanilla':
    representations = multi_mnist_cnn.deepnn(l, X, n)
    concat_reps = tf.reshape(representations, [M, n * n])
    fc1 = tf.compat.v1.layers.dense(concat_reps, n * n)
    fc2 = tf.compat.v1.layers.dense(fc1, n * n)
    P_hat_raw = tf.compat.v1.layers.dense(fc2, n * n)
    P_hat_raw_square = tf.reshape(P_hat_raw, [M, n, n])

    P_hat = tf.nn.softmax(P_hat_raw_square, axis=-1)  # row-stochastic!

    losses = tf.nn.softmax_cross_entropy_with_logits(
        labels=P_true, logits=P_hat_raw_square, axis=2)
    losses = tf.reduce_mean(losses, axis=-1)
    loss = tf.reduce_mean(losses)

elif method == 'sinkhorn':
    representations = multi_mnist_cnn.deepnn(l, X, n)
    pre_sinkhorn = tf.reshape(representations, [M, n, n])
    P_hat = sinkhorn_operator(pre_sinkhorn, temp=temperature)
    P_hat_logit = tf.math.log(P_hat)

    losses = tf.nn.softmax_cross_entropy_with_logits(
        labels=P_true, logits=P_hat_logit, axis=2)
    losses = tf.reduce_mean(losses, axis=-1)
    loss = tf.reduce_mean(losses)

elif method == 'gumbel_sinkhorn':
    representations = multi_mnist_cnn.deepnn(l, X, n)
    pre_sinkhorn = tf.reshape(representations, [M, n, n])
    P_hat = sinkhorn_operator(pre_sinkhorn, temp=temperature)

    P_hat_sample, _ = gumbel_sinkhorn(
        pre_sinkhorn, temp=temperature, n_samples=n_s)
    P_hat_sample_logit = tf.math.log(P_hat_sample)

    P_true_sample = tf.expand_dims(P_true, 1)
    P_true_sample = tf.tile(P_true_sample, [1, n_s, 1, 1])

    losses = tf.nn.softmax_cross_entropy_with_logits(
        labels=P_true_sample, logits=P_hat_sample_logit, axis=3)
    losses = tf.reduce_mean(losses, axis=-1)
    losses = tf.reshape(losses, [-1])
    loss = tf.reduce_mean(losses)

elif method == 'deterministic_neuralsort':
    scores = multi_mnist_cnn.deepnn(l, X, 1)
    scores = tf.reshape(scores, [M, n, 1])
    P_hat = util.neuralsort(scores, temperature)

    losses = tf.nn.softmax_cross_entropy_with_logits(
        labels=P_true, logits=tf.math.log(P_hat + 1e-20), axis=2)
    losses = tf.reduce_mean(losses, axis=-1)
    loss = tf.reduce_mean(losses)

elif method == 'stochastic_neuralsort':
    scores = multi_mnist_cnn.deepnn(l, X, 1)
    scores = tf.reshape(scores, [M, n, 1])
    P_hat = util.neuralsort(scores, temperature)

    scores_sample = tf.tile(scores, [n_s, 1, 1])
    scores_sample += util.sample_gumbel([M * n_s, n, 1])
    P_hat_sample = util.neuralsort(
        scores_sample, temperature)

    P_true_sample = tf.tile(P_true, [n_s, 1, 1])
    losses = tf.nn.softmax_cross_entropy_with_logits(
        labels=P_true_sample, logits=tf.math.log(P_hat_sample + 1e-20), axis=2)
    losses = tf.reduce_mean(losses, axis=-1)
    loss = tf.reduce_mean(losses)
else:
    raise ValueError("No such method.")


def vec_gradient(l):  # l is a scalar
    gradient = tf.gradients(l, tf.compat.v1.trainable_variables())
    vec_grads = [tf.reshape(grad, [-1]) for grad in gradient]  # flatten
    z = tf.concat(vec_grads, 0)  # n_params
    return z

prop_correct = util.prop_correct(P_true, P_hat)
prop_any_correct = util.prop_any_correct(P_true, P_hat)

opt = tf.compat.v1.train.AdamOptimizer(initial_rate)
train_step = opt.minimize(loss)
saver = tf.compat.v1.train.Saver()

# MAIN BEGINS

sess = tf.compat.v1.Session()

sess.run(tf.compat.v1.global_variables_initializer())
train_sh, validate_sh, test_sh = sess.run([
    train_iterator.string_handle(),
    val_iterator.string_handle(),
    test_iterator.string_handle()
])


TRAIN_PER_EPOCH = mnist_input.TRAIN_SET_SIZE // (l * M)
VAL_PER_EPOCH = mnist_input.VAL_SET_SIZE // (l * M)
TEST_PER_EPOCH = mnist_input.TEST_SET_SIZE // (l * M)
best_correct_val = 0


def save_model(epoch):
    saver.save(sess, checkpoint_path + 'checkpoint', global_step=epoch)

def save_model_to_volume(epoch):
    saver.save(sess, volume_experiment_path + 'sort', global_step=epoch)

def load_model_from_checkpoint():
    filename = tf.train.latest_checkpoint(checkpoint_path)
    if filename == None:
        raise Exception("No model found.")
    prnt("Loaded model %s." % filename)
    saver.restore(sess, filename)

def load_model_from_volume():
    filename = tf.train.latest_checkpoint(volume_experiment_path)
    if filename == None:
        raise Exception("No model found in volume.")
    prnt("Loaded model from volume: %s." % filename)
    saver.restore(sess, filename)

def train(epoch):
    loss_train = []
    for _ in range(TRAIN_PER_EPOCH):
        _, l = sess.run([train_step, loss],
                        feed_dict={handle: train_sh})
        loss_train.append(l)
    prnt('Average loss:', sum(loss_train) / len(loss_train))

def test(epoch, val=False):
    global best_correct_val
    p_cs = []
    p_acs = []
    for _ in range(VAL_PER_EPOCH if val else TEST_PER_EPOCH):
        p_c, p_ac = sess.run([prop_correct, prop_any_correct], feed_dict={
                             handle: validate_sh if val else test_sh,
                             evaluation: True})
        p_cs.append(p_c)
        p_acs.append(p_ac)

    p_c = sum(p_cs) / len(p_cs)
    p_ac = sum(p_acs) / len(p_acs)

    if val:
        prnt("Validation set: prop. all correct %f, prop. any correct %f" %
             (p_c, p_ac))
        if p_c > best_correct_val:
            best_correct_val = p_c
            prnt('Saving...')
            save_model(epoch)
    else:
        prnt("Test set: prop. all correct %f, prop. any correct %f" % (p_c, p_ac))

if should_load_model_from_volume:
    load_model_from_volume_path = volume_model_path
else:
    # Train the model
    for epoch in range(1, NUM_EPOCHS + 1):
        prnt('Epoch', epoch, '(%s)' % experiment_id)
        train(epoch)
        test(epoch, val=True)
        logfile.flush()

    # Save the trained model to the volume
    save_model_to_volume(NUM_EPOCHS)

# Load the model (either from volume or trained)
if should_load_model_from_volume:
    load_model_from_volume()
    sort_iterator = get_sort_iterator()
    sort_sh = sess.run(sort_iterator.string_handle())


    open_set_dir_path = "/arc/mnist_sort"
    open_set_files = os.listdir(open_set_dir_path)

    # n(n/2)
    sort_pairs = []
    for a in range(len(open_set_files)):
        for b in range(a+1, len(open_set_files)):
            sort_pairs.append([a, b])

    comparator_results = np.zeros(len(open_set_files))
    z_f = 0
    while True:
        try:
            start_time = time.time()
            p_h = sess.run(P_hat, feed_dict={
                handle: sort_sh,
                evaluation: True})
            sort_permutation = p_h[0]
            print(sort_permutation)
            # comparator '0' means  a < b (I think)
            comparator = sort_permutation[0].argmax()
            print(comparator)
            print(sort_files)
            # TODO: Figure out how to extract string from strided slice or upgrade to tf2 wtf...?
            # print(f'{sort_files[0][comparator]} > {sort_files[0][(comparator+1)%2]}')
            print(f'{open_set_files[sort_pairs[z_f][comparator]]} > {open_set_files[sort_pairs[z_f][(comparator+1)%2]]}')
            comparator_results[sort_pairs[z_f][comparator]] = comparator_results[sort_pairs[z_f][comparator]] + 1
            end_time = time.time()
            print(f"Search operator execution time: {(end_time - start_time) * 1000:.2f} ms")
            z_f = z_f + 1
        except tf.errors.OutOfRangeError:
            break
    
    print(comparator_results)
    print(open_set_files)
    sort_results = [(comparator_results[i], open_set_files[i]) for i in range(len(open_set_files))]
    sort_results.sort(key=lambda x:x[0])
    print(sort_results)

    # TODO: Iterator results, print values/file path
    # with open(f"/arc/sorted.txt", 'w') as f:
    #     for digit in digits:
    #         f.write(f"{digit[2]}\n")
    # TODO: Test Python custom sort comparator
else:
    load_model_from_checkpoint()
    test(NUM_EPOCHS, val=False)

sess.close()
logfile.close()
