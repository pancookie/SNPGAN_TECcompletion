import argparse
import itertools
import os
import sklearn.model_selection
from glob import glob
from random import shuffle

parser = argparse.ArgumentParser()
parser.add_argument('--folder_path', default='/data/Yang/places2/val_256', type=str,
                    help='The folder path')
parser.add_argument('--train_filename', default='./data_flist/train.flist', type=str,
                    help='The output filename.')
parser.add_argument('--validation_filename', default='./data_flist/validation.flist', type=str,
                    help='The output filename.')

if __name__ == "__main__":

    args = parser.parse_args()

    SUPPORTED_EXTENSIONS = ["png", "jpg", "jpeg", "txt", "pkl"]
    ret = []#; import pdb; pdb.set_trace()
    
    ret.extend(list(itertools.chain.from_iterable(glob(os.path.join(args.folder_path, "*/*.{}".format(ext))) for ext in SUPPORTED_EXTENSIONS)))

    # make 2 lists to save file paths
    shuffle(ret) #; import pdb; pdb.set_trace()
    training_file_names, validation_file_names = sklearn.model_selection.train_test_split(ret, test_size=0.2)

    # make output file if not existed
    if not os.path.exists(args.train_filename):
        os.mknod(args.train_filename)

    if not os.path.exists(args.validation_filename):
        os.mknod(args.validation_filename)

    # write to file
    fo = open(args.train_filename, "w")
    fo.write("\n".join(training_file_names))
    fo.close()

    fo = open(args.validation_filename, "w")
    fo.write("\n".join(validation_file_names))
    fo.close()

