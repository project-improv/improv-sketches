#  https://www.binarystudy.com/2021/09/how-to-load-preprocess-visualize-CIFAR-10-and-CIFAR-100.html
# tar -xvzf cifar-10-python.tar.gz
# https://github.com/nfmcclure/tensorflow_cookbook/blob/master/08_Convolutional_Neural_Networks/04_Retraining_Current_Architectures/04_download_cifar10.py
# https://stackoverflow.com/questions/60138697/typeerror-cannot-handle-this-data-type-1-1-3-f4
import os
import numpy as np
from skimage.io import imsave
import tarfile
import urllib.request

<<<<<<< HEAD

def unpickle(file):
    import pickle

    with open(file, "rb") as fo:
        dict = pickle.load(fo, encoding="latin1")
    return dict


cifar_link = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
=======
def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='latin1')
    return dict

cifar_link = 'https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'
>>>>>>> 2a4b6a6fc166dedce4b966b2b1523a41db721fcb

data_dir = "data/CIFAR10"
os.makedirs(data_dir, exist_ok=True)

<<<<<<< HEAD
cifar_file = os.path.join(data_dir, "cifar-10-python.tar.gz")
=======
cifar_file = os.path.join(data_dir, 'cifar-10-python.tar.gz')
>>>>>>> 2a4b6a6fc166dedce4b966b2b1523a41db721fcb
if not os.path.isfile(cifar_file):
    filename, headers = urllib.request.urlretrieve(cifar_link, cifar_file)

with tarfile.open(cifar_file) as tar:
    tar.extractall(path=data_dir)
    tar.close()

<<<<<<< HEAD
batch_path = os.path.join(data_dir, "cifar-10-batches-py")
batch_names = ["data_batch_" + str(x) for x in range(1, 6)]
=======
batch_path = os.path.join(data_dir, 'cifar-10-batches-py')
batch_names = ['data_batch_' + str(x) for x in range(1,6)]
>>>>>>> 2a4b6a6fc166dedce4b966b2b1523a41db721fcb

file = os.path.join(batch_path, batch_names[0])

data_batch_1 = unpickle(file)
<<<<<<< HEAD
data = data_batch_1["data"]

labels = data_batch_1["labels"]
=======
data = data_batch_1['data']

labels = data_batch_1['labels']
>>>>>>> 2a4b6a6fc166dedce4b966b2b1523a41db721fcb

meta_file = os.path.join(batch_path, "batches.meta")
meta_data = unpickle(meta_file)

<<<<<<< HEAD
label_names = meta_data["label_names"]
=======
label_names = meta_data['label_names']
>>>>>>> 2a4b6a6fc166dedce4b966b2b1523a41db721fcb

os.makedirs(os.path.join(data_dir, "images"), exist_ok=True)
os.makedirs(os.path.join(data_dir, "labels"), exist_ok=True)

<<<<<<< HEAD
for i in range(300):
=======
for i in range(len(data)):
>>>>>>> 2a4b6a6fc166dedce4b966b2b1523a41db721fcb
    img = data[i]
    R = img[0:1024].reshape(32, 32)
    G = img[1024:2048].reshape(32, 32)
    B = img[2048:3072].reshape(32, 32)
    img = np.dstack((R, G, B))

    imsave(os.path.join(data_dir, "images/{}.jpg".format(i)), img)
<<<<<<< HEAD

=======
    
>>>>>>> 2a4b6a6fc166dedce4b966b2b1523a41db721fcb
    with open(os.path.join(data_dir, "labels/{}.txt".format(i)), "w") as text_file:
        text_file.write("%s" % labels[i])
        text_file.close()

with open(os.path.join(data_dir, "label_names.txt"), "w") as text_file:
    text_file.write(", ".join(label_names))
<<<<<<< HEAD
    text_file.close()
=======
    text_file.close()
>>>>>>> 2a4b6a6fc166dedce4b966b2b1523a41db721fcb
