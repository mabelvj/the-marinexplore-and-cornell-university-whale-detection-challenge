import sklearn
from sklearn.cross_validation import train_test_split#, StratifiedShuffleSplit
from sklearn.model_selection import StratifiedShuffleSplit
from load_and_generate import *



#Adjusting to freqs 250Hz : height 33
image_height = 33
image_width = 23

#Rescale width
image_height = 32
image_width = 32

image_size = image_height*image_width
#create_label_folders('../whale-inputs/data/', 'test', 'test.csv')

test_folders = maybe_extract(os.path.abspath( '../whale-inputs/data/test/'))
test_datasets = maybe_pickle(test_folders, 10000, False, image_height, image_width )




def make_arrays(nb_rows, image_height, image_width):
  if nb_rows:
    dataset = np.ndarray((nb_rows,  image_height, image_width), dtype=np.float32) #change to dif
    labels = np.ndarray(nb_rows, dtype=np.int32)
  else:
    dataset, labels = None, None
  return dataset, labels

def load_datasets(pickle_files, train_size):
  num_classes = len(pickle_files)
  train_dataset, train_labels = None, None

  for label, pickle_file in enumerate(pickle_files):       
    try:
      with open(pickle_file, 'rb') as f:
        letter_set = pickle.load(f)
        if train_dataset is not None:
            train_dataset = np.concatenate((train_dataset, letter_set), axis=0)
            train_labels = np.concatenate((train_labels,np.ones(letter_set.shape[0])* label),axis=0)
        else:
            train_dataset = letter_set
            train_labels = np.ones(letter_set.shape[0])* label
    except Exception as e:
      print('Unable to process data from', pickle_file, ':', e)
      raise 
      
  return train_dataset, train_labels
            

# Extraction of whole dataset
test_dataset, test_labels = load_datasets(test_datasets, 54503)

print('Testing:', test_dataset.shape, test_labels.shape)



def randomize(dataset, labels):
  permutation = np.random.permutation(labels.shape[0])
  shuffled_dataset = dataset[permutation,:,:]
  shuffled_labels = labels[permutation]
  return shuffled_dataset, shuffled_labels


test_dataset, test_labels = randomize(test_dataset, test_labels)


# test_generation
pickle_file = 'test.pickle'

try:
  f = open(pickle_file, 'wb')
  save = {

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
