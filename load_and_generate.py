from multiprocessing import Pool
import matplotlib.pyplot as plt
import numpy as np
import aifc
import os
import tensorflow as tf
import random
import time

from os import listdir
from os.path import isfile, join
file_path = '../whale-inputs/data/train' #change to test if desired
file_names = [f for f in listdir(file_path) if isfile(join(file_path, f))]

data_loc = '../whale-inputs/data' 
train_folder = 'train'

   
    
def get_time_data(data_loc,train_folder,file_name):
    ''' Retrieves time data from the aiff file'''
    f = aifc.open(os.path.join(data_loc,train_folder,file_name), 'r')
    str_frames = f.readframes(f.getnframes())
    Fs = f.getframerate()
    time_data = np.fromstring(str_frames, np.short).byteswap()
    f.close()
    return time_data
    
def compute_specgram(data_loc,train_folder,file_name):
	'''Retrieves time data from the aiff file and compute the spectogram for time_data'''  
	plt.figure(figsize=(18.,15.), dpi=80)
	f = aifc.open(os.path.join(data_loc,train_folder, file_name), 'r')
	str_frames = f.readframes(f.getnframes())
	Fs = f.getframerate()
	time_data = np.fromstring(str_frames, np.short).byteswap()
	f.close()
	 
	# spectrogram of file
	Pxx, freqs, bins, im = plt.specgram(time_data,Fs=Fs,noverlap=90,cmap=plt.cm.gist_heat)

	plt.savefig(data_loc + '/'+ train_folder + '/specgrams/'+ file_name.split('.')[0] + '.png', bbox_inches='tight')
	plt.close()
	
#if __name__ == '__main__':
    #pool = Pool(processses=2)
    #pool.map(compute_specgram, (data_loc,train_folder,file_names))

for i, file_name in  enumerate(file_names):
	print "Processing figure %d..." %i
	start_time = time.time()
	compute_specgram(data_loc,train_folder,file_name)
	print("--- %s seconds --- \n" % (time.time() - start_time))
