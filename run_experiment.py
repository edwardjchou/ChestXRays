import numpy as np
import tensorflow as tf
import os
import config
import train
import time
import copy
import random
#import infer
#import InferWrapper as IW

def main(args):

    # Network Parameters
    args.n_epochs = 51 

    # Directories
    args.train_input_file = '/data/klooby/courses/cs273b/images.npy' 
    args.train_labels_file = '/data/klooby/courses/cs273b/labels.npy' 

    # results folder will be written here
    args.save_dir = '/data/llbricks/model_checkpoints/3d/112017_testnewarch'  
    args.input_model_ckpt = ''

    args.n_train = 0  
    args.n_val = 1000
    args.log_loss = False 
    args.w_opt = False 
    args.tv_reg = 0 
    args.batch_size = 2

    # END USER INPUT! ---------------------------------

    # Make saving files and directories
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    args.tb_interval = 10
    nch = 16
    runid = 'model_' + copy.deepcopy(time.strftime("%Y%m%d_%H%M%S"))

    args.reg1 = 1E-4 
    args.reg2 = 1E-4
    args.dropout = 0.3
    
#    for i in range(1000):

    args.run_name = runid + '_{}ch_pl'.format(nch)

    args.lr = 10**(random.uniform(-6,-3))

    # Execute training
    train.main(args)

    # Reset tf graph
    tf.reset_default_graph()

#        try:
#            Infer = IW.InferWrapper(os.path.join(args.save_dir,args.run_name))
#            Infer('phantom1')
#            Infer('invivo')
#        except: 
#            pass

if __name__ == '__main__':
    a = config.parser()
    main(a)

