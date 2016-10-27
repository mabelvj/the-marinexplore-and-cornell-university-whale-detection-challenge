import sklearn
from sklearn.cross_validation import train_test_split#, StratifiedShuffleSplit
from sklearn.model_selection import StratifiedShuffleSplit
from load_and_generate import *


image_size = 128
image_width = 23
image_height = 129
#image_width = 128
#image_height = 128


#Adjusting to freqs 250Hz : height 33
image_height = 33
image_width = 23

#Rescale width
image_height = 32
image_width = 32

train_folders = maybe_extract(os.path.abspath( '../whale-inputs/data/train/'))
train_datasets = maybe_pickle(train_folders, 7000, False, image_height, image_width )




def make_arrays(nb_rows, img_size):
  if nb_rows:
    dataset = np.ndarray((nb_rows,  image_height, image_width), dtype=np.float32) #change to dif
    labels = np.ndarray(nb_rows, dtype=np.int32)
  else:
    dataset, labels = None, None
  return dataset, labels

def merge_datasets(pickle_files, train_size, valid_size=0):
  num_classes = len(pickle_files)
  valid_dataset, valid_labels = make_arrays(valid_size, image_size)
  train_dataset, train_labels = make_arrays(train_size, image_size)
  vsize_per_class = valid_size // num_classes
  tsize_per_class = train_size // num_classes
    
  start_v, start_t = 0, 0
  end_v, end_t = vsize_per_class, tsize_per_class
  end_l = vsize_per_class+tsize_per_class
  for label, pickle_file in enumerate(pickle_files):       
    try:
      with open(pickle_file, 'rb') as f:
        letter_set = pickle.load(f)
        # let's shuffle the letters to have random validation and training set
        np.random.shuffle(letter_set)
        if valid_dataset is not None:
          valid_letter = letter_set[:vsize_per_class, :, :]
          valid_dataset[start_v:end_v, :, :] = valid_letter
          valid_labels[start_v:end_v] = label
          start_v += vsize_per_class
          end_v += vsize_per_class
                    
        train_letter = letter_set[vsize_per_class:end_l, :, :]
        train_dataset[start_t:end_t, :, :] = train_letter
        train_labels[start_t:end_t] = label
        start_t += tsize_per_class
        end_t += tsize_per_class
    except Exception as e:
      print('Unable to process data from', pickle_file, ':', e)
      raise
    
  return valid_dataset, valid_labels, train_dataset, train_labels
            

# Extraction of whole dataset
v_, v_, train_dataset, train_labels = merge_datasets(
  train_datasets, 7027, 0)
  
whale_dataset = np.copy(train_dataset)        
whale_labels = np.copy(train_labels)  

#Dataset mixing    
train_size = np.int(2.*7027.)
valid_size = 0 #np.floor(14000.*.3).astype(int)
test_size = 3000

validation_dataset, validation_labels, train_dataset, train_labels = merge_datasets(
  train_datasets, train_size, valid_size) #train_datasets_ is_pickle
#_, _, test_dataset, test_labels = merge_datasets(test_datasets, test_size)
#Stratified shuffle Split to keep elements of each class
X= np.copy(train_dataset) # we do this to preserve data during splits
y = np.copy(train_labels)
# Split train_dataset into train_dataset and test_dataset 60-40
sss = StratifiedShuffleSplit(n_splits=1,  test_size=0.40, random_state=np.random.seed(19))
for train_ix, test_ix in sss.split(X,y):
    #print("TRAIN:", train_index, "TEST:", test_index)
    train_dataset, test_dataset, train_labels, test_labels = X[train_ix,:,:], X[test_ix,:,:],y[train_ix], y[test_ix]    

    
X2= np.copy(test_dataset) # we do this to preserve data during splits
y2 = np.copy(test_labels)
    
# Split remailin test_dataset into test_dataset and validation_dataset 20 -20
sss = StratifiedShuffleSplit(n_splits=1,  test_size=0.50, random_state=np.random.seed(21))
for test_ix, valid_ix in sss.split( X2, y2):
    test_dataset, valid_dataset, test_labels, valid_labels = X2[test_ix,:,:], X2[valid_ix,:,:], y2[test_ix], y2[valid_ix]    


print('Training:', train_dataset.shape, train_labels.shape)
print('Validation:', valid_dataset.shape, valid_labels.shape)
print('Testing:', test_dataset.shape, test_labels.shape)



def randomize(dataset, labels):
  permutation = np.random.permutation(labels.shape[0])
  shuffled_dataset = dataset[permutation,:,:]
  shuffled_labels = labels[permutation]
  return shuffled_dataset, shuffled_labels

train_dataset, train_labels = randomize(train_dataset, train_labels)
test_dataset, test_labels = randomize(test_dataset, test_labels)
valid_dataset, valid_labels = randomize(valid_dataset, valid_labels)

# test_generation
pickle_file = 'exploration.pickle'

try:
  f = open(pickle_file, 'wb')
  save = {
    'train_dataset': train_dataset,
    'train_labels': train_labels,
    'valid_dataset': valid_dataset,
    'valid_labels': valid_labels,
    'test_dataset': test_dataset,
    'test_labels': test_labels,
    }
  pickle.dump(save, f, pickle.HIGHEST_PROTOCOL)
  f.close()
except Exception as e:
  print('Unable to save data to', pickle_file, ':', e)
  raise


  
  
statinfo = os.stat(pickle_file)
print('Compressed pickle size:', statinfo.st_size)  
