from multiprocessing import Pool
import matplotlib.pyplot as plt
import numpy as np
import aifc
import os
#import tensorflow as tf
import random
import time


try:
   import cPickle as pickle
except:
   import pickle


import os.path
from os import listdir
from os.path import isfile, join
file_path = '../whale-inputs/data/' #change to test if desired
file_names = [f for f in listdir(os.path.join(file_path, 'train')) if isfile(os.path.join(file_path, 'train', f))]

data_loc = '../whale-inputs/data' 
train_folder = 'train'
#train_folder = join(file_path, 'train')
#test_folder = join(file_path, 'test')

#def maybe_pickle(data_folders, min_num_images_per_class, force=False):
  #dataset_names = []
  #for folder in data_folders:
    #set_filename = folder + '.pickle'
    #dataset_names.append(set_filename)
    #if os.path.exists(set_filename) and not force:
      ## You may override by setting force=True.
      #print('%s already present - Skipping pickling.' % set_filename)
    #else:
      #print('Pickling %s.' % set_filename)
      #dataset = load_letter(folder, min_num_images_per_class)
      #try:
        #with open(set_filename, 'wb') as f:
          #pickle.dump(dataset, f, pickle.HIGHEST_PROTOCOL)
      #except Exception as e:
        #print('Unable to save data to', set_filename, ':', e)
  
  #return dataset_names

#train_datasets = maybe_pickle(train_folder, 100)
#test_datasets = maybe_pickle(test_folders, 50)

#def open_pickle(data_folders):
  #folder = random.sample(data_folders, 1)
  #pickle_filename = ''.join(folder) + '.pickle'
  #try:
    #with open(pickle_filename, 'rb') as f:
      #dataset = pickle.load(f)
  #except Exception as e:
    #print('Unable to read data from', pickle_filename, ':', e)
    #return
    
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
	if not os.path.isfile(data_loc + '/' + train_folder + '/specgrams/' + file_name.split('.')[0] + '.png'):
		try:
			plt.figure(figsize=(18.,15.), dpi=80)
			f = aifc.open(os.path.join(data_loc,train_folder, file_name), 'r')
			str_frames = f.readframes(f.getnframes())
			#Fs = f.getframerate()
			Fs= 4000
			time_data = np.fromstring(str_frames, np.short).byteswap()
			f.close()
			 

			# Pxx is the segments x freqs array of instantaneous power, freqs is
			# the frequency vector, bins are the centers of the time bins in which
			# the power is computed, and im is the matplotlib.image.AxesImage
			# instance

			# spectrogram of file
			
			Pxx, freqs, bins, im = plt.specgram(time_data,Fs=Fs,noverlap=90,cmap=plt.cm.gist_heat)

			plt.axis('off')
			plt.savefig(data_loc + '/'+ train_folder + '/specgrams/'+ file_name.split('.')[0] + '.png', bbox_inches='tight')
			plt.close()
		except ValueError:
			print("Error in file: "+ file_name + "...\n")
	
#if __name__ == '__main__':
    #pool = Pool(processses=2)
    #pool.map(compute_specgram, (data_loc,train_folder,file_names))

#for i, file_name in  enumerate(file_names):
number_of_samples= 100
for i, file_name in  enumerate(random.sample(file_names, number_of_samples)):
	print "Processing figure %d..." %i
	start_time = time.time()
	compute_specgram(data_loc,train_folder,file_name)
	print("--- %s seconds --- \n" % (time.time() - start_time))
