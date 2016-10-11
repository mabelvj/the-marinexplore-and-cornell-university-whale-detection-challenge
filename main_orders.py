from load_and_generate import *
train_folders = maybe_extract(os.path.abspath( '../whale-inputs/data/train/'))
train_datasets = maybe_pickle(train_folders, 7000, force= True)
