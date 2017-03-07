# Import Modules
from __future__ import print_function
from six.moves import cPickle as pickle
from six.moves import range
from six.moves.urllib.request import urlretrieve
from scipy import ndimage
from PIL import Image
import numpy as np
import os
import sys
import tarfile
import h5py
from numpy import random
import matplotlib.pyplot as plt

# Download data
print('Downloading data...')

url = 'http://ufldl.stanford.edu/housenumbers/'

def maybe_download(filename, force=False):
  """Download a file if not present, and make sure it's the right size."""
  if force or not os.path.exists(filename):
    print('Attempting to download:', filename) 
    filename, _ = urlretrieve(url + filename, filename)
    print('Download Complete!')
  statinfo = os.stat(filename)
  return filename

train_filename = maybe_download('train.tar.gz')
test_filename = maybe_download('test.tar.gz')
extra_filename = maybe_download('extra.tar.gz')

print('Successfully downloaded data!')


# Unzip Data
print('Unzipping data...')
np.random.seed(8)

def maybe_extract(filename, force=False):
  # Remove .tar.gz
  root = os.path.splitext(os.path.splitext(filename)[0])[0]  
  if os.path.isdir(root) and not force:
    # You may override by setting force=True.
    print('%s already present - Skipping extraction of %s.' % (root, filename))
  else:
    print('Extracting data for %s. This may take a while. Please wait.' % root)
    tar = tarfile.open(filename)
    sys.stdout.flush()
    tar.extractall()
    tar.close()
  data_folders = root
  print(data_folders)
  return data_folders
  
train_folders = maybe_extract(train_filename)
test_folders = maybe_extract(test_filename)
extra_folders = maybe_extract(extra_filename)

print('Successfully unzipped data!')

train_folders = "train"
test_folders = "test"
extra_folders = "extra"

# Create dictionary for bounding boxes
print('Creating dictionary of bounding boxes...')
class DigitStructFile:
    def __init__(self, inf):
        self.inf = h5py.File(inf, 'r')
        self.digitStructName = self.inf['digitStruct']['name']
        self.digitStructBbox = self.inf['digitStruct']['bbox']
    def getName(self,n):
        return ''.join([chr(c[0]) for c in self.inf[self.digitStructName[n][0]].value])
    def bboxHelper(self,attr):
        if (len(attr) > 1):
            attr = [self.inf[attr.value[j].item()].value[0][0] for j in range(len(attr))]
        else:
            attr = [attr.value[0][0]]
        return attr
    def getBbox(self,n):
        bbox = {}
        bb = self.digitStructBbox[n].item()
        bbox['height'] = self.bboxHelper(self.inf[bb]["height"])
        bbox['label'] = self.bboxHelper(self.inf[bb]["label"])
        bbox['left'] = self.bboxHelper(self.inf[bb]["left"])
        bbox['top'] = self.bboxHelper(self.inf[bb]["top"])
        bbox['width'] = self.bboxHelper(self.inf[bb]["width"])
        return bbox
    def getDigitStructure(self,n):
        s = self.getBbox(n)
        s['name']=self.getName(n)
        return s
    def getAllDigitStructure(self):
        return [self.getDigitStructure(i) for i in range(len(self.digitStructName))]
    def getAllDigitStructure_ByDigit(self):
        pictDat = self.getAllDigitStructure()
        result = []
        structCnt = 1
        for i in range(len(pictDat)):
            item = { 'filename' : pictDat[i]["name"] }
            figures = []
            for j in range(len(pictDat[i]['height'])):
               figure = {}
               figure['height'] = pictDat[i]['height'][j]
               figure['label']  = pictDat[i]['label'][j]
               figure['left']   = max(0,pictDat[i]['left'][j])
               figure['top']    = max(0,pictDat[i]['top'][j])
               figure['width']  = pictDat[i]['width'][j]
               figures.append(figure)
            structCnt = structCnt + 1
            item['boxes'] = figures
            result.append(item)
        return result
    
print("Successfully created dictionary of bounding boxes!")


# Get Digit Structure
print('Getting digit structure for training data...')
digitFileTrain=DigitStructFile(os.path.join('train','digitStruct.mat'))
train_data=digitFileTrain.getAllDigitStructure_ByDigit()
print('Success!')

print('Getting digit structure for test data...')
digitFileTest=DigitStructFile(os.path.join('test','digitStruct.mat'))
test_data=digitFileTest.getAllDigitStructure_ByDigit()
print('Success!')


#print('Getting digit structure for extra data...')
#digitFileExtra=DigitStructFile(os.path.join('extra','digitStruct.mat'))
#extra_data=digitFileExtra.getAllDigitStructure_ByDigit()
#print('Success!')

# Crop Training Images
print('Cropping training images...')
train_imsize = np.ndarray([len(train_data),2])
for i in np.arange(len(train_data)):
    filename = train_data[i]['filename']
    fullname = os.path.join(train_folders, filename)
    im = Image.open(fullname)
    train_imsize[i, :] = im.size[:]

print('Success!')

# Crop Test Images
print('Cropping test images...')
test_imsize = np.ndarray([len(test_data),2])
for i in np.arange(len(test_data)):
    filename = test_data[i]['filename']
    fullname = os.path.join(test_folders, filename)
    im = Image.open(fullname)
    test_imsize[i, :] = im.size[:]

print('Success!')

## Crop Extra Images
#print('Cropping extra images...')
#extra_imsize = np.ndarray([len(extra_data)/20,2])
#for i in np.arange(int(len(extra_data)/20)):
#    filename = extra_data[i]['filename']
#    fullname = os.path.join(extra_folders, filename)
#    im = Image.open(fullname)
#    extra_imsize[i, :] = im.size[:]

print('Success!')
# Use extra data
def generate_dataset(data, folder):
    dataset = np.ndarray([len(data),64,64,3], dtype='float32')
    labels = np.ones([len(data),6], dtype=int) *0 
    for i in np.arange(len(data)):
     filename = data[i]['filename']
     #print ('Processing {} \n',filename )
     fullname = os.path.join(folder, filename)
     im = Image.open(fullname)
     boxes = data[i]['boxes']
     num_digit = len(boxes)
     labels[i,0] = num_digit
     top = np.ndarray([num_digit], dtype='uint64')
     left = np.ndarray([num_digit], dtype='uint64')
     height = np.ndarray([num_digit], dtype='uint64')
     width = np.ndarray([num_digit], dtype='uint64')
     for j in np.arange(num_digit):
         if j < 5: 
             labels[i,j+1] = boxes[j]['label']
             if boxes[j]['label'] == 10: labels[i,j+1] = 0
         else: print('#',i,'image has more than 5 digits.')
         top[j] = boxes[j]['top']
         left[j] = boxes[j]['left']
         height[j] = boxes[j]['height']
         width[j] = boxes[j]['width']
     img_top = np.amin(top)
     img_left = np.amin(left)
     img_height = np.amax(top) + height[np.argmax(top)] - img_top
     img_width = np.amax(left) + width[np.argmax(left)] - img_left
     box_left = int(np.floor(img_left - 0.1 * img_width))
     box_top = int(np.floor(img_top - 0.1 * img_height))
     box_right = int(np.amin([np.ceil(box_left + 1.2 * img_width), im.size[0]]))
     box_bottom = int(np.amin([np.ceil(img_top + 1.2 * img_height), im.size[1]]))
     im = im.crop((box_left, box_top, box_right, box_bottom)).resize([32,32], Image.ANTIALIAS)
     pix = np.array(im)
     norm_pix=(pix*1.0)/128 - 1
     norm_pix -= np.mean(norm_pix, axis=0)
     norm_pix=im.reshape(64,64,3)
     dataset[i] = norm_pix
    return dataset, labels

print('Generating training dataset and labels...')
train_dataset, train_labels = generate_dataset(train_data, train_folders)
print('Success! \n Training set: {} \n Training labels: {}'.format(train_dataset.shape, train_labels.shape))
del train_data

print('Generating testing dataset and labels...')
test_dataset, test_labels = generate_dataset(test_data, test_folders)
print('Success! \n Testing set: {} \n Testing labels: {}'.format(test_dataset.shape, test_labels.shape))
del test_data

for i in range(40):
 plt.imshow(train_dataset[i].reshape(32,32))
 plt.show()


#print('Generating extra dataset and labels...')
#extra_dataset, extra_labels = generate_dataset(extra_data, extra_folders)
#print('Success! \n Testing set: {} \n Testing labels: {}'.format(extra_dataset.shape, extra_labels.shape))
#del extra_data

# Clean up data by deleting digits more than 5 (very few)
print('Cleaning up training data...')
train_dataset = np.delete(train_dataset, 29929, axis=0)
train_labels = np.delete(train_labels, 29929, axis=0)
print('Success!')

from sklearn.cross_validation import train_test_split
train_dataset_new, valid_dataset, train_labels_new, valid_labels = train_test_split(train_dataset, train_labels, test_size=0.1, random_state = 42)

DATA_PATH="./"
def write_npy_file(data_array, lbl_array, data_set_name):
    np.save(os.path.join(DATA_PATH, +data_set_name+'_imgs.npy'), data_array)
    print('Saving to %s_svhn_imgs.npy file done.' % data_set_name)
    np.save(os.path.join(DATA_PATH, +data_set_name+'_labels.npy'), lbl_array)
    print('Saving to %s_svhn_labels.npy file done.' % data_set_name)


write_npy_file(train_dataset_new,train_labels_new,"svnh_seq_train" )
write_npy_file(valid_dataset,valid_labels,"svnh_seq_valid")
write_npy_file(test_dataset,test_labels,"svnh_seq_test")

print('Success!')
