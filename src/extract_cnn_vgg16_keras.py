# -*- coding: utf-8 -*-
import sys
import os

#from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input

import h5py
import numpy as np
from numpy import linalg as LA

import vgg16
import utils


class VGG16FeatureExtractor:

    def __init__(self):
        self.input_shape = (224, 224, 3)
        self.train_images_path = os.getcwd()[:-3] + 'data/training_images/'
        self.dumped_h5py_filename = 'vgg16_feats.h5'
        self.feats_dataset = 'feats_dataset'
        self.names_dataset = 'names_dataset'
        self.vgg16_model = self.get_vgg16_model()

    def get_train_images(self, img_path=None):
        if img_path is None:
            return utils.get_imlist(self.train_images_path)
        return utils.get_imlist(img_path)

    def get_vgg16_model(self):
        vgg16_model = vgg16.VGG16(weights='imagenet',
                            input_shape=(self.input_shape[0], self.input_shape[1], 3),
                            pooling='max',
                            include_top=False)
        return vgg16_model

    def extract_img_vgg16_feature(self, img_path):
        img = image.load_img(img_path, target_size=(self.input_shape[0], self.input_shape[1]))
        img = image.img_to_array(img)

        img = np.expand_dims(img, axis=0)
        img = preprocess_input(img)

        feat = self.vgg16_model.predict(img)
        img_name = os.path.split(img_path)[1]
        img_norm_feat = feat[0] / LA.norm(feat[0])
        return img_name, img_norm_feat

    def extract_all_img_vgg16_feature(self, train_imgs_path=None):
        if train_imgs_path is None:
            trian_img_list = self.get_train_images(self.train_images_path)
        else :
            trian_img_list = self.get_train_images(train_imgs_path)
        img_names = []
        img_feats = []
        for i, img_path in enumerate(trian_img_list):
            img_name, img_feat = self.extract_img_vgg16_feature(img_path)
            img_feats.append(img_feat)
            img_names.append(img_name)
            print "image %d feature extraction, total %d images" % ((i + 1), len(trian_img_list))
        return img_names, img_feats

    def dump_img_features(self, img_names, img_feats):
        img_feats = np.array(img_feats)
        h5f = h5py.File(self.dumped_h5py_filename, 'w')
        h5f.create_dataset(self.feats_dataset, data=img_feats)
        h5f.create_dataset(self.names_dataset, data=img_names)
        h5f.close()

if __name__ == "__main__":
    # weights: 'imagenet'
    # pooling: 'max' or 'avg'
    # input_shape: (width, height, 3), width and height should >= 48
    feature_extractor = VGG16FeatureExtractor()
    img_names, img_feats = feature_extractor.extract_all_img_vgg16_feature(feature_extractor.train_images_path)
    feature_extractor.dump_img_features(img_names, img_feats)