import argparse
import itertools
import os
import sklearn.model_selection
import random
from glob import glob
from random import shuffle

parser = argparse.ArgumentParser()
parser.add_argument('--folder_path', default='/data/Yang/tec15min_128x128_bin', type=str,
                    help='The folder path')
parser.add_argument('--train_filename', default='../generative_inpainting/data_flist/igstec_15_128_lt/train.flist', type=str,
                    help='The output filename.')
parser.add_argument('--validation_filename', default='../generative_inpainting/data_flist/igstec_15_128_lt/validation.flist', type=str,
                    help='The output filename.')

if __name__ == "__main__":

    args = parser.parse_args()

    SUPPORTED_EXTENSIONS = ["png", "jpg", "jpeg", "txt", "pkl"]
    ret = []#; import pdb; pdb.set_trace()
    
    list_yr = [x for x in range(1998, 2019) if x not in [2004, 2016]]
    # ret.extend(list(itertools.chain.from_iterable(glob(os.path.join(args.folder_path, "*/*.{}".format(ext))) for ext in SUPPORTED_EXTENSIONS)))
    ret.extend(list(itertools.chain.from_iterable(glob(os.path.join(args.folder_path, "{}/*.pkl".format(_yr))) for _yr in list_yr)))

    # make 2 lists to save file paths
    shuffle(ret) #; import pdb; pdb.set_trace()
    training_file_names = ret; validation_file_names = []
    # training_file_names, validation_file_names = sklearn.model_selection.train_test_split(ret, test_size=0.2)

    # randomly select number of files in training list
    # t_f_n = random.sample(training_file_names, 63004)
    t_f_n = training_file_names

    # make output file if not existed
    if not os.path.exists(args.train_filename):
        os.mknod(args.train_filename)

    if not os.path.exists(args.validation_filename):
        os.mknod(args.validation_filename)

    # write to file
    fo = open(args.train_filename, "w")
    fo.write("\n".join(t_f_n))
    fo.close()

    fo = open(args.validation_filename, "w")
    fo.write("\n".join(validation_file_names))
    fo.close()

