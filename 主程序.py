from ast import arg
import os
import random
from tkinter import HORIZONTAL
from tkinter.tix import IMAGE
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

#训练图片的大小
IMAGE_SIZE = 48

#训练次数
epochs = 20

#批量大小
batch_size = 5

#训练样本总数
train_samples = train.shape[0]

#测试样本总数
test_samples = test.shape[0]

#样本图片所在路径
train_data_dir = 'D:\\code\\learn\\data_sex\\train_data\\'
validation_data_dir = 'D:\\data_sex\\test_data\\'

#模型存放路径
File_PATH = 'model.h5'

class Dataset(object):

    def __init__(self):
        self.train = None
        self.test  = None

    def read(self, img_rows = IMAGE_SIZE,img_cols = IMAGE_SIZE):
        train_datagen = ImageDataGenerator(
            rescale =1,255,
            holizental_flip = True
        )

        test_datagen = ImageDataGenerator(rescale=1./255)

        validation_generator = train_datagen.flow_from_diretory(
            train_data_dir,
            target_size = (img_rows,img_cols),
            class_mode = 'binary'
        )

        self.train = train_generator
        self.valid = validation_generator

class Model(object):


    def __init__(self):
        self.model = tf.keras.Sequential()
        self.model.add(tf.keras.layers.Conv2D(32,(3,3),input_shape = (IMG_SIZE,IMG_SIZE,3),activation='relu'))#卷积
        self.model.add(tf.keras.layers.MaxPooling2D((2,2)))#池化
        self.model.add(tf.keras.layers.Conv2D(32,(3,3),activation='relu'))#卷积
        self.model.add(tf.keras.layers.MaxPooling2D((2,2)))#池化
        self.model.add(tf.keras.layers.Conv2D(64,(3,3),activation='relu'))#卷积
        self.model.add(tf.keras.layers.MaxPooling2D((2,2)))#池化
        self.model.add(tf.keras.layers.Conv2D(64,(3,3),activation='relu'))#卷积
        self.model.add(tf.keras.layers.MaxPooling2D((2,2)))#池化
        self.model.add(tf.keras.layers.Flatten())#展开
        self.model.add(tf.keras.layers.Dense(64,activation='relu'))#全连接层
        self.model.add(tf.keras.layers.Dense(7,activation='softmax'))
        self.model.summary()

    def train(self, dataset, batch_size = batch_size, nb_epoch = epochs):
        self.model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
        self.model.fit_generator(dataset.train,
                                 steps_per_epoch=nb_train_samples // batch_size,
                                 epochs=epochs,
                                 validation_data=dataset.valid,
                                 validation_steps=nb_validation_samples//batch_size)

    def save(self, file_path=FILE_PATH):
        print('Model Saved.')
        self.model.save(file_path)

    def load(self, file_path = FILE_PATH):
        print('Model Loaded.')
        self.model.load(file_path)

    def predict(self,image):
        #预测样本分类
        img = image.resize((1, IMAGE_SIZE, IMAGE_SIZE, 3))
        img = image.astype('float32')

        #归一化
        img /= 255

        #概率
        result = model.predict(img)
        
        return np.argmax(img[0])

    def evaluate(self, dataset):
        # 测试样本准确率
        score = self.model.evaluate(dataset.valid)