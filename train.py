from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import tensorflow as tf
import numpy as np
import math
import time
import config
import scipy.io as sio
import re
import alexnet
from shutil import copyfile


def main(args):

    if args.run_name == '':
        args.run_name = time.strftime("%Y%m%d_%H%M")

    # Load data
    print('loading data from %s...' % args.train_input_file)
    X_train = np.tile(np.expand_dims(np.load(args.train_input_file)[:200],axis = 3),(1,1,1,3))
    y_train = np.load(args.train_labels_file)
    print('loading data from %s...' % args.train_input_file)
    X_val = np.tile(np.expand_dims(np.load(args.train_input_file)[:200],axis = 3),(1,1,1,3))
    y_val = np.load(args.train_labels_file)
    sz = X_train.shape

    # get other useful thing for running the tf graph
    n_train, n_val = X_train.shape[0], X_val.shape[0]
    n_batches_train = int(math.floor((n_train-1) / args.batch_size))
    n_batches_val = int(math.floor((n_val-1) / args.batch_size))
    ridx_train = np.random.choice(n_train, size=(n_train,), replace=False)

    # turn off tensorflow verbose
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    # tf Graph input
    x = tf.placeholder(tf.float32, [None, sz[1], sz[2], sz[3]], 'input')
    y = tf.placeholder(tf.int32, [None], 'prediction')
    is_training = tf.placeholder(tf.bool)

    # Construct model
    if args.arch == '3d':
        args.training = True
        pred = alexnet.alexnet(x)

    # Define loss and optimizer
    loss,msqerr = get_loss_classify(args,y,pred)
    optimizer = tf.train.AdamOptimizer(learning_rate=args.lr).minimize(loss)

    # Initializer for the variables
    init = tf.global_variables_initializer()

    # Create a Saver
    saver = tf.train.Saver()

    # Also, save all training and validation msq/loss scores
    msqe_train = np.zeros(args.n_epochs)
    msqe_val = np.zeros(args.n_epochs)
    loss_train = np.zeros(args.n_epochs)

    # # Force gpu to use only half of available GPU memory
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=.8)

    # Document arch and args
    save_model_settings(args)
    make_tensorboard_summaries(args)

    # Launch the graph
#    tf.initialize_all_variables()
    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
    # with tf.Session() as sess:
        merged = tf.summary.merge_all()

        # If an input model was provided, initialize using the model
        print('Initializing new model')
        sess.run(init)
        if args.input_model_ckpt != '':
            print('Loading previous model from %s' % args.input_model_ckpt)
#            saver.restore(sess, args.input_model_ckpt)

        # Train over n_epochs
        for epoch in range(args.n_epochs):

            # Train on Training Set
            print('Epoch: {0}'.format(epoch))
            for batch_num in range(n_batches_train+1):
                idx1 = batch_num * args.batch_size
                idx2 = min(idx1 + args.batch_size, n_train)
                batch_idx = ridx_train[idx1:idx2]
                X_batch = X_train[batch_idx, :]
                y_batch = y_train[batch_idx]
                msqe_, loss_, _, = sess.run([msqerr, loss, optimizer], feed_dict={x: X_batch, y: y_batch, is_training: True})
                msqe_train[epoch] += msqe_ * (idx2-idx1) / n_train
                loss_train[epoch] += loss_ * (idx2-idx1) / n_train

            print('Train: \tMSQE = %6.4f\tLoss = %6.4f' % (msqe_train[epoch]*100, loss_train[epoch]*100))

            # Evaluate on Validation Set
            for batch_num in range(n_batches_val+1):
                idx1 = batch_num * args.batch_size
                idx2 = min(idx1 + args.batch_size, n_val)
                X_batch = X_val[idx1:idx2]
                y_batch = y_val[idx1:idx2]
                msqe_, summary = sess.run([msqerr, merged], feed_dict={x: X_batch, y: y_batch, is_training: False})
                msqe_val[epoch] += msqe_ * (idx2-idx1) / n_val
            print('Val: \tMSQE = %6.4f' % (msqe_val[epoch]*100))

            # save checkpoint 
            save_ckpt(saver,args,sess,epoch)

        # Save all of the model parameters as a .mat file!
        save_network_mat(sess,args,{'msqe_train': msqe_train, 'msqe_val': msqe_val, 'loss_train': loss_train})

def save_ckpt(saver,args,sess,epoch):

    if epoch == args.n_epochs:
        save_path = saver.save(sess, os.path.join(args.save_dir, args.run_name,'model.ckpt'))
        print("Model saved in file: %s" % save_path)
    elif (epoch) % args.tb_interval == 0:
        save_path = saver.save(sess, os.path.join(args.save_dir, args.run_name,'model_{0}.ckpt'.format(epoch)))
        print("Model saved in file: %s" % save_path)

def save_network_mat(sess,args,mpdict):
    '''Takes input dict of parameters and saves these values , along with all kernals and biases of the network
    these parameters are saved in a .mat file in the run's folder under modelparams.mat
    ''' 
    mpdir = os.path.join(args.save_dir, args.run_name,'modelparams.mat')
    param_list = [v for v in tf.trainable_variables() if 'kernel' in v.name or 'bias' in v.name]
    for p in param_list:
        name = re.sub('/', '_', p.name)
        name = re.sub(':0', '', name)
        value = sess.run(p)
        mpdict.update({name: value})
    sio.savemat(mpdir, mpdict)

def make_tensorboard_summaries(args):
    # Add hyperparameters to Tensorboard summary
   with tf.name_scope('hyperparameters'):
        tf.summary.scalar('learning_rate', args.lr)
        tf.summary.scalar('reg1', args.reg1)
        tf.summary.scalar('reg2', args.reg2)
        tf.summary.scalar('batch_size', args.batch_size)
        tf.summary.scalar('dropout', args.dropout)

   """
   with tf.name_scope('loss'):
        tf.summary.scalar('loss', loss)
        tf.summary.scalar('l2_err', msqerr)
        tf.summary.scalar('l1_loss', l1_loss)
        tf.summary.scalar('l2_loss', l2_loss)
   """


def load_from_mat(dataset_path, nimgs=0):
    import h5py
    if dataset_path != '':
        if nimgs == 0:
            img = np.array(h5py.File(dataset_path)['img'])
            ref = np.array(h5py.File(dataset_path)['ref'])
        else:
            img = np.array(h5py.File(dataset_path)['img'][:nimgs])
            ref = np.array(h5py.File(dataset_path)['ref'][:nimgs])
        szi = img.shape
        szr = ref.shape
        print('(%d,%d,%d,%d,%d) tensor loaded.' % (szi[0], szi[1], szi[2], szi[3], szi[4]))
        # Normalize img, ref by their RMS
        img = img.reshape(szi[0], np.prod(szi[1:]))
        ref = ref.reshape(szr[0], np.prod(szr[1:]))
        img /= np.sqrt(np.mean(np.square(img),axis=1,keepdims=True))
        ref /= np.sqrt(np.mean(np.square(ref),axis=1,keepdims=True))
        print(np.min(img))
        print(np.max(img))
        img = img.reshape(szi)
        ref = ref.reshape(szr)
        # img /= np.linalg.norm(img, ord='fro', axis=(1,2),keepdims=True)
        # ref /= np.linalg.norm(ref, ord='fro', axis=(1,2),keepdims=True)
        return img, ref[:,:,:,:,0]
    else:
        return None


def get_loss(args, y, pred):


    # Evaluation metric
    if args.log_loss:
        msqerr = tf.losses.mean_squared_error(tf.log(y), tf.log(pred))
    else:
        msqerr = tf.losses.mean_squared_error(y,pred)

#    tvloss = tf.reduce_sum(tf.image.total_variation(pred))

    # Define loss and optimizer
    kernel_list = [v for v in tf.trainable_variables() if 'kernel' in v.name]
    l1_loss, l2_loss = 0, 0
    for v in kernel_list:
        l1_loss += tf.reduce_mean(tf.abs(v))
        l2_loss += tf.nn.l2_loss(v)

    loss = msqerr + args.reg1 * l1_loss + args.reg2 * l2_loss #+ args.tv_reg*tvloss

    return loss, msqerr

def get_loss_channel(args, y, pred):

    num_channels = 16
    y = tf.tile(tf.expand_dims(y,axis=3),[1,1,1,num_channels,1])
    print('shape of y ',y.shape)
    print('shape of pred ',pred.shape)

    # Evaluation metric

    if args.log_loss:
        msqerr = tf.losses.mean_squared_error(tf.log(y), tf.log(pred))
    else:
        msqerr = tf.losses.mean_squared_error(y, pred)
    print('shape of msqerr ',msqerr.shape)

#    tvloss = tf.reduce_sum(tf.image.total_variation(pred))

    # Define loss and optimizer
    kernel_list = [v for v in tf.trainable_variables() if 'kernel' in v.name]
    l1_loss, l2_loss = 0, 0
    for v in kernel_list:
        l1_loss += tf.reduce_mean(tf.abs(v))
        l2_loss += tf.nn.l2_loss(v)

    loss = msqerr + args.reg1 * l1_loss + args.reg2 * l2_loss #+ args.tv_reg*tvloss

    return loss, msqerr

def get_loss_classify(args, y, pred):
    
    print('yshape',y.shape)
    print('predshape',pred.shape)

    y = tf.one_hot(y,15)
    print('yshape',y.shape)

    msqerr = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = pred, labels = y))
#    print(dummy.get_shape())

#    print('make sure variable names match if reg isnt working!!')
#    kernel_list = [v for v in tf.trainable_variables() if 'kernel' in v.name]
#    l1_loss, l2_loss = 0, 0
#    for v in kernel_list:
#        l1_loss += tf.reduce_mean(tf.abs(v))
#        l2_loss += tf.nn.l2_loss(v)

#    loss = msqerr + args.reg1 * l1_loss + args.reg2 * l2_loss 
    loss = msqerr

    return loss, msqerr

def save_model_settings(args):

    try:
            os.stat(os.path.join(args.save_dir, args.run_name))
    except:
            os.mkdir(os.path.join(args.save_dir, args.run_name))

    copyfile('alexnet.py', os.path.join(args.save_dir, args.run_name,'arch.py'))
    fp = open(os.path.join(args.save_dir, args.run_name,'model_args.txt'),'w+');
    for k,a in vars(args).items():
         fp.write('{0} = {1}\n'.format(k, a));
    fp.close();
    

if __name__ == '__main__':
    a = config.parser()
    main(a)

