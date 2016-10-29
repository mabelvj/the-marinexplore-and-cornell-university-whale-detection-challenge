from multiprocessing import Pool
import matplotlib.pyplot as plt
import numpy as np
import aifc
import os
import random
import time
import pandas as pd

try:
   import cPickle as pickle
except:
   import pickle


import os.path
from os import listdir
from os.path import isfile, join


data_loc = '../whale-inputs/data' 
train_folder = join(data_loc, 'train')
test_folder = join(data_loc, 'test')

num_classes = 2
np.random.seed(133)

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
			plt.figure(figsize=(18.,16.), dpi=50) #900x800
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
	
	
	
def process_image(file_name, enlarge = True, image_height = 129, image_width = 23): 
	'''Retrieves time data from the aiff file and compute the spectogram for time_data'''
	'''enlarge: gives option to resize the image to new dimensions WxH by interpolation'''
	if file_name.endswith('.aiff'):
		f = aifc.open(file_name, 'r')
		str_frames = f.readframes(f.getnframes())
		Fs = f.getframerate()
		time_data = np.fromstring(str_frames, np.short).byteswap()
		f.close()
		Pxx, freqs, bins, im = plt.specgram(time_data,NFFT=256,Fs=Fs,noverlap=90,cmap=plt.cm.gist_heat)
		Pxx = Pxx[[freqs<250.]] # Right-whales call occur under 250Hz
		#print Pxx.shape
		from scipy.misc import imresize
		from sklearn import preprocessing
		
		if enlarge: #change image size
			#Pxx_prep = imresize(np.log10(Pxx),(image_height,image_width), interp= 'lanczos').astype('float32')
			Pxx_prep = imresize(Pxx,(image_height,image_width), interp= 'lanczos').astype('float32')
		else: #image size not changed
			Pxx_prep = np.log(Pxx).astype('float32')
		
		#Pxx_prep = preprocessing.MinMaxScaler().fit_transform(Pxx_prep) #rescale to 0-1
		Pxx_prep = preprocessing.StandardScaler(copy=True, with_mean=True, with_std=True).fit_transform(Pxx_prep) #rescale by std
		Pxx_ = (Pxx_prep*255.0).astype(int)
		# Returning raw values to perform operations. Used to obtain raw data for ipynb
		#Pxx_ = Pxx_prep
		#Pxx_ = Pxx
		return Pxx_
	else:
		print("Error in file: "+ file_name + "...\n")
		pass
		



def return_image_size(image_width = 129, image_height = 23):
	return np.round(np.sqrt(image_width * image_height))

pixel_depth = 255.0  # Number of levels per pixel.

def load_image(folder, min_num_images, image_height = 129, image_width = 23 ):
  """Load the data for a single letter label."""
  import time
  from scipy import ndimage
  start_time_init = time.time()
  image_files = os.listdir(folder)
  dataset = np.ndarray(shape=(len(image_files), image_height, image_width),
                         dtype=np.float32)
                         
  num_images = 0
  for i, image in enumerate(image_files):
    image_file = os.path.join(folder, image)
    #print "Image %d... \n"%i
    #print image_file
    #print '\n'
    start_time = time.time()
    try:
      if image_file.endswith('.aiff'):
          image_data =  np.array(process_image(image_file, True, image_height, image_width)) 
            #(ndimage.imread(image_file).astype(float) - pixel_depth / 2) / pixel_depth
          if image_data.shape != (image_height, image_width):
            raise Exception('Unexpected image shape: %s' % str(image_data.shape))
          dataset[num_images, :, :] = image_data
          num_images = num_images + 1
    except IOError as e:
      print('Could not read:', image_file, ':', e, '- it\'s ok, skipping.')
    #print("--- %s seconds --- \n" % (time.time() - start_time))
    
  dataset = dataset[0:num_images, :, :]
  if num_images < min_num_images:
    raise Exception('Many fewer images than expected: %d < %d' %
                    (num_images, min_num_images))
  print("Time ellapsed: %s seconds --- \n" % (time.time() - start_time_init))  
  print('Full dataset tensor:', dataset.shape)
  print('Mean:', np.mean(dataset))
  print('Standard deviation:', np.std(dataset))
  return dataset



def maybe_pickle(data_folders, min_num_images_per_class, force=False, image_height = 129, image_width = 23):
  dataset_names = []
  for folder in data_folders:
    set_filename = folder + '.pickle'
    dataset_names.append(set_filename)
    if os.path.exists(set_filename) and not force:
      # You may override by setting force=True.
      print('%s already present - Skipping pickling.' % set_filename)
    else:
      print('Pickling %s' % set_filename)
      dataset = load_image(folder, min_num_images_per_class, image_height , image_width)
      try:
        with open(set_filename, 'wb') as f:
          pickle.dump(dataset, f, pickle.HIGHEST_PROTOCOL)
      except Exception as e:
        print('Unable to save data to', set_filename, ':', e)
  
  return dataset_names


def open_pickle(data_folders):
  folder = random.sample(data_folders, 1)
  #print folder
  pickle_filename = ''.join(folder) + '.pickle'
  try:
    with open(pickle_filename, 'rb') as f:
      dataset = pickle.load(f)
      return dataset
  except Exception as e:
    print('Unable to read data from', pickle_filename, ':', e)
    pass


#train_datasets = maybe_pickle([train_folder+'/whales', train_folder+'/no_whales'], 500, force= True)
#test_datasets = maybe_pickle([test_folder], 50)#54503

#number_of_samples= 100
#for i, file_name in  enumerate(random.sample(file_names, number_of_samples)):
	#print "Processing figure %d..." %i
	#start_time = time.time()
	#compute_specgram(data_loc,train_folder,file_name)
	#print("--- %s seconds --- \n" % (time.time() - start_time))
	
def create_label_folders(data_folder, dataset_folder, file_name):
	import pandas as pd
	ground_truth = pd.read_csv(os.path.join(data_folder, file_name), index_col= 0)
	df = pd.concat([ground_truth, pd.get_dummies(ground_truth.label)], axis=1); 
	train_folders_names= ['whales', 'no_whales']
	whales = df[df[1]==1].index.values
	print whales
	print ("!!!!")
	no_whales = df[df[0]==1].index.values
	for train_folder in train_folders_names:
		if not os.path.isdir(os.path.join(data_folder, dataset_folder, train_folder )):
			os.makedirs(os.path.join(data_folder, dataset_folder, train_folder ))
		
		import shutil
		if train_folder== 'whales':
			files_names= whales
		else : 
			files_names = no_whales
		
		for file_to_copy in files_names:
			#print file_to_copy
			#print (os.path.join(data_folder, dataset_folder,file_to_copy+'.aiff'))
			if os.path.isfile(os.path.join(data_folder, dataset_folder,file_to_copy+'.aiff')):
				print data_folder + file_to_copy
				shutil.copyfile(os.path.join(data_folder, dataset_folder,file_to_copy+'.aiff') , os.path.join(data_folder,dataset_folder,train_folder,file_to_copy+'.aiff') )
	pass
	


def maybe_extract(filename, force=False):
  "Returns the absolute path of the folders for each label"
  root = os.path.splitext(os.path.splitext(filename)[0])[0]  # remove .tar.gz
  data_folders = [
    os.path.join(root, d) for d in sorted(os.listdir(root))
    if os.path.isdir(os.path.join(root, d))]
  if len(data_folders) != num_classes:
    raise Exception(
      'Expected %d folders, one per class. Found %d instead.' % (
        num_classes, len(data_folders)))
  print(data_folders)
  return data_folders
  
  

def randomize_linear(dataset, labels):
	"Randomize a dataset that has been straighted out"	
	permutation = np.random.permutation(labels.shape[0])
	shuffled_dataset = dataset[permutation,:]
	shuffled_labels = labels[permutation]
	return shuffled_dataset, shuffled_labels
  
def randomize(dataset, labels):
	"Randomize a dataset"
	permutation = np.random.permutation(labels.shape[0])
	shuffled_dataset = dataset[permutation,:, :]
	shuffled_labels = labels[permutation]
	return shuffled_dataset, shuffled_labels

    
    
	
