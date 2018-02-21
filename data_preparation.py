from __future__ import absolute_import
import cv2
import numpy as np
import os

def read_img(im_dir, im_w, im_h):
    im = cv2.resize(cv2.imread(im_dir), (im_w, im_h)).astype(np.float32)
    im[:,:,0] -= 103.939
    im[:,:,1] -= 116.779
    im[:,:,2] -= 123.68
    # im = im.transpose((2,0,1))
    im = np.expand_dims(im, axis=0)
    return im

def trans_label(str_label, label_dim):
    label = np.zeros((1, label_dim), dtype='float32')
    for i in xrange(label_dim):
        label[0, i] = float(str_label[i*2])
    return label

def load_data(im_dir, train_im_list, test_im_list):
    """Loads SUN-Attribute dataset.
    # Returns
        Tuple of Numpy arrays: `(x_train, y_train), (x_test, y_test)`.
    """
    train_num_samples = 7170
    test_num_samples = 7169
    img_width, img_height = 224, 224
    label_dim = 102

    train_info = [line.rstrip('\n') for line in open(train_im_list)]
    test_info = [line.rstrip('\n') for line in open(test_im_list)]
    
    # Channel last
    x_train = np.zeros((train_num_samples, img_width, img_height, 3), dtype='float32')
    y_train = np.zeros((train_num_samples, label_dim), dtype='float32')

    x_test = np.zeros((test_num_samples, img_width, img_height, 3), dtype='float32')
    y_test = np.zeros((test_num_samples, label_dim), dtype='float32')

    # Generate training samples.
    for i in xrange(train_num_samples):
        im_name = im_dir + train_info[i*2]
        str_label = train_info[i*2+1]
        x_train[i,:,:,:] = read_img(im_name, img_width, img_height)
        y_train[i,:] = trans_label(str_label, label_dim)

    # Generate testing samples.
    for i in xrange(test_num_samples):
        im_name = im_dir + test_info[i*2]
        str_label = test_info[i*2+1]
        x_test[i,:,:,:] = read_img(im_name, img_width, img_height)
        y_test[i,:] = trans_label(str_label, label_dim)

    return (x_train, y_train), (x_test, y_test)

if __name__ == "__main__":
    im_dir = "../dataset/images/"
    train_im_list = "../dataset/SUNAttributeDB/train.txt" 
    test_im_list = "../dataset/SUNAttributeDB/test.txt"
    load_data(im_dir, train_im_list, test_im_list)