import numpy as np
import os
import struct
import torch
from PIL import Image
from skimage.morphology import disk, skeletonize, erosion
from skimage.transform import resize
from torch.utils.data import Dataset
from torchvision import transforms as T

__all__ = ["ThinMNIST"]


class ThinMNIST(Dataset):
    def __init__(self, which_set, path, transform_list=[T.ToTensor()], out_channels=3, colors=False,
                 erosion_ratio=4, working_im_size=(64, 64), add_skeleton=True, split=1.0, require_labels=False):
        if which_set == 'train':
            fimages = os.path.join(path, 'train-images-idx3-ubyte')
            flabels = os.path.join(path, 'train-labels-idx1-ubyte')
        elif which_set == 'val':
            fimages = os.path.join(path, 'train-images-idx3-ubyte')
            flabels = os.path.join(path, 'train-labels-idx1-ubyte')
        else:
            fimages = os.path.join(path, 't10k-images-idx3-ubyte')
            flabels = os.path.join(path, 't10k-labels-idx1-ubyte')

        # Load images
        with open(fimages, 'rb') as f:
            _, _, rows, cols = struct.unpack(">IIII", f.read(16))
            images = np.fromfile(f, dtype=np.uint8).reshape(-1, rows, cols)

        # Load labels
        with open(flabels, 'rb') as f:
            struct.unpack(">II", f.read(8))
            labels = np.fromfile(f, dtype=np.int8)
            labels = torch.from_numpy(labels.astype(np.int))

        # split
        if which_set == 'train':
            ntotal = images.shape[0]
            ntrain = int(ntotal * split)
            self.images = images[:ntrain]
            self.labels = labels[:ntrain]
        elif which_set == 'val':
            ntotal = images.shape[0]
            ntrain = int(ntotal * split)
            self.images = images[ntrain:]
            self.labels = labels[ntrain:]
        else:
            self.images = images
            self.labels = labels

        print('Number of images: ', len(self.images), which_set)
        print('Number of labels: ', len(self.labels), which_set)

        # class attributes
        self.transform = T.Compose(transform_list)
        self.working_im_size = working_im_size
        self.add_skeleton = add_skeleton
        self.n_classes = 2
        self.out_channels = out_channels
        self.colors = colors
        self.require_labels = require_labels

        # thinning
        self.structuring_element = disk(erosion_ratio)

        # adding channels
        if self.out_channels > 1:
            self.images = np.tile(self.images[:, :, :, np.newaxis], self.out_channels)
            self.structuring_element = np.tile(self.structuring_element[:, :, np.newaxis], self.out_channels)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):

        image = self.images[idx] / 255.  # Range [0,255]
        label = self.labels[idx]

        # resize image
        image = resize(image, self.working_im_size, mode='constant', cval=0, anti_aliasing=False)
        image = image > .2

        # make digit thin
        if self.add_skeleton:
            skeleton = skeletonize(image[:, :, 0] if self.out_channels > 1 else image)
        image = erosion(image, self.structuring_element)
        if self.add_skeleton:
            image = np.maximum(image, skeleton[:, :, np.newaxis] if self.out_channels > 1 else skeleton)

        image = resize(image.astype(np.float), (28, 28), mode='constant', cval=0, anti_aliasing=True) > .15
        if self.out_channels > 1:
            image = np.pad(image, ((2, 2), (2, 2), (0, 0)), 'constant', constant_values=((0, 0), (0, 0), (0, 0)))
        else:
            image = np.pad(image, ((2, 2), (2, 2)), 'constant', constant_values=((0, 0), (0, 0)))

        # get image range back
        if self.out_channels>1:
            mask = image[:, :, 0].astype(np.uint8)
        else:
            mask = image.astype(np.uint8)
        image = (image * 255).astype(np.uint8)

        # add colors
        if self.colors:
            background = np.random.randint(0, 255, 3)
            digit = np.random.randint(0, 255, 3)
            image[mask == 0] = background
            image[mask == 1] = digit

        # to tensors & transformations
        image = self.transform(Image.fromarray(image))
        # mask = self.transform(np.expand_dims(mask*255, 2)).squeeze()
        mask = torch.LongTensor(mask)

        if not self.require_labels:
            return image, mask
        else:
            return image, mask, label


if __name__ == '__main__':
    train_data = ThinMNIST('train', 'path_to_mnist', split=1.0)
    test_data = ThinMNIST('test', 'path_to_mnist', split=1.0)

    im_tr, mask_tr = train_data[0]
    im_te, mask_te = test_data[0]

    print(im_tr.size)
    print(im_te.size)

    print(mask_tr.size)
    print(mask_te.size)
