# import time
import os
import argparse

import cv2
import numpy as np
import tensorflow as tf
import neuralgym as ng

from inpaint_model import InpaintCAModel


parser = argparse.ArgumentParser()
parser.add_argument(
    '--flist', default='', type=str,
    help='The filenames of image to be processed: input, mask, output.')
parser.add_argument(
    '--image_height', default=64, type=int,
    help='The height of images should be defined, otherwise batch mode is not'
    ' supported.')
parser.add_argument(
    '--image_width', default=64, type=int,
    help='The width of images should be defined, otherwise batch mode is not'
    ' supported.')
parser.add_argument(
    '--checkpoint_dir', default='', type=str,
    help='The directory of tensorflow checkpoint.')
parser.add_argument(
    '--path_image', default='/data/Yang/masked_LT', type=str,
    help='The path for input image')
parser.add_argument(
    '--path_mask', default='/home/yangpan/projs/MITTEC_mask', type=str,
    help='The path of mask')
parser.add_argument(
    '--path_output', default='/data/Yang/sngan_18yrs/tmp', type=str,
    help='The path of output')
parser.add_argument(
    '--folder_yr', default=False, type=bool,
    help='Whether to put output categorized in years')
parser.add_argument(
    '--inp_type', default='igs', type=str,
    help='Whether to inpaint igstec or MIT-TEC')

if __name__ == "__main__":
    FLAGS = ng.Config('/home/yangpan/projs/gi/yml_folder/inpaint_cv_x.yml')
    # ng.get_gpus(1)
    # os.environ['CUDA_VISIBLE_DEVICES'] =''
    args = parser.parse_args()

    assert args.inp_type in ['igs', 'mit']
    sess_config = tf.compat.v1.ConfigProto()
    sess_config.gpu_options.allow_growth = True
    sess = tf.compat.v1.Session(config=sess_config)

    model = InpaintCAModel()
    input_image_ph = tf.placeholder(
        tf.float32, shape=(1, args.image_height, args.image_width*2, 3))
    output = model.build_server_graph(FLAGS, input_image_ph)
    output = (output + 1.) * 127.5
    output = tf.reverse(output, [-1])
    output = tf.saturate_cast(output, tf.float32) ### dtype 
    vars_list = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.GLOBAL_VARIABLES)
    assign_ops = []
    for var in vars_list:
        vname = var.name
        from_name = vname
        var_value = tf.contrib.framework.load_variable(
            args.checkpoint_dir, from_name)
        assign_ops.append(tf.compat.v1.assign(var, var_value))
    sess.run(assign_ops)
    print('Model loaded.')

    with open(args.flist, 'r') as f:
        lines = f.read().splitlines()
    
    # make folder for putput
    os.makedirs(args.path_output, exist_ok=True)
    # make subfolder(s) when necessary
    if args.folder_yr:
        yr_list = ['{}'.format(item[:4]) for item in lines]
        yr_list_u = (list(set(yr_list)))
        for yr in yr_list_u:
            os.makedirs(os.path.join(args.path_output, yr), exist_ok=True)

    # t = time.time()
    for line in lines:
    # for i in range(100):
        # default style, full path info in flist file
        # image, mask, out = line.split() # default separator as a space
        # base = os.path.basename(mask)

        # flist in 'yyyymmdd_hr'
        yr = line[:4]
        hr = int(line.split('_')[-1])
        
        if args.inp_type == 'igs':
            image = os.path.join(args.path_image, yr, 'masked', '{}_mask.txt'.format(line))
            mask = os.path.join(args.path_mask, 'MITTEC_mask_{}.npy'.format(hr))
        elif args.inp_type == 'mit': 
            # /data/Yang/masked_MITTEC/org_mix_brush/txt_lt_brushmasked_10per/{}.txt
            # /data/Yang/masked_MITTEC/org_mix_rm/txt_lt_rm_10per/{}.txt   /totalmask_org_rm_10per/{}.npy 
            image = os.path.join(args.path_image, '{}.txt'.format(line))
            mask = os.path.join(args.path_mask,'{}_mask.npy'.format(line))
        else:
            assert('Not supported inpainting type, which must be igs or mit')
        if args.folder_yr:
            out = os.path.join(args.path_output, yr, '{}_out.txt'.format(line))
        else:
            out = os.path.join(args.path_output, '{}_out.txt'.format(line))

        if os.path.splitext(image)[1] == '.txt':
            image = np.loadtxt(image).reshape(args.image_height, args.image_width, -1)
            image = np.repeat(image, 3, axis=2)
        else:
            image = cv2.imread(image)
        if os.path.splitext(mask)[1] == '.npy':
            mask = np.load(mask) #; import pdb; pdb.set_trace()
            mask = np.repeat(mask, 3, axis=2) 
        else:
            mask = cv2.imread(mask)
        # image = cv2.resize(image, (args.image_width, args.image_height))
        # mask = cv2.resize(mask, (args.image_width, args.image_height))
        # cv2.imwrite(out, image*(1-mask/255.) + mask)
        # # continue
        # image = np.zeros((128, 256, 3))
        # mask = np.zeros((128, 256, 3))

        assert image.shape == mask.shape

        # h, w, _ = image.shape
        # grid = 8
        # image = image[:h//grid*grid, :w//grid*grid, :]
        # mask = mask[:h//grid*grid, :w//grid*grid, :]
        # print('Shape of image: {}'.format(image.shape))

        image = np.expand_dims(image, 0)
        mask = np.expand_dims(mask, 0)
        input_image = np.concatenate([image, mask], axis=2)

        # load pretrained model
        result = sess.run(output, feed_dict={input_image_ph: input_image})
        print('Processed: {}'.format(out))
        # cv2.imwrite(out, result[0][:, :, ::-1])
        np.savetxt(out, result[0].flatten())

    # print('Time total: {}'.format(time.time() - t))
