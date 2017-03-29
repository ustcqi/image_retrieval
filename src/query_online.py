# -*- coding: utf-8 -*-
import os
import sys

from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input

from numpy import linalg as LA
import numpy as np
import h5py

import vgg16
import utils

class ImageRetrieval:
    def __init__(self):
        self.input_shape = (224, 224, 3)
        self.dumped_h5py_feats_file = 'vgg16_feats.h5'
        self.feats_dataset = 'feats_dataset'
        self.names_dataset = 'names_dataset'
        self.retrieval_images_path = os.getcwd()[:-3] + 'data/retrieval_images/'

        self.img_names, self.img_feats = self.load_h5py_names_and_feats()
        self.vgg16_model = self.get_vgg16_model()

    def load_h5py_names_and_feats(self, h5py_file=None):
        if h5py_file is None:
            h5f = h5py.File(self.dumped_h5py_feats_file, 'r')
        else:
            h5f = h5py.File(h5py_file, 'r')
        img_feats = h5f[self.feats_dataset][:]
        img_names = h5f[self.names_dataset][:]
        h5f.close()
        return img_names, img_feats

    def get_vgg16_model(self):
        vgg16_model = vgg16.VGG16(weights='imagenet',
                                  input_shape=(self.input_shape[0], self.input_shape[1], 3),
                                  pooling='max',
                                  include_top=False)
        return vgg16_model

    def retrieval_image(self, img_path):
        img = image.load_img(img_path, target_size=(self.input_shape[0], self.input_shape[1]))
        img = image.img_to_array(img)
        img = np.expand_dims(img, axis=0)
        img = preprocess_input(img)

        retrieval_feat = self.vgg16_model.predict(img)
        img_name = os.path.split(img_path)[1]
        norm_feat = retrieval_feat[0] / LA.norm(retrieval_feat[0])

        scores = np.dot(norm_feat, self.img_feats.T)
        rank_ID = np.argsort(scores)[::-1]
        rank_score = scores[rank_ID]
        # print rank_ID
        print('Retrieving the image %s' % img_name)
        print(rank_score)

        max_res = 4
        img_list = [self.img_names[index] for i, index in enumerate(rank_ID[0: max_res])]
        print('Search result: ', img_list)
        print('\n')


if __name__ == "__main__":
    image_searcher = ImageRetrieval()
    retrieval_images_path = os.getcwd()[:-3] + 'data/retrieval_images/'
    retrieval_img_list = utils.get_imlist(retrieval_images_path)
    for i, img_path in enumerate(retrieval_img_list):
        image_searcher.retrieval_image(img_path)

