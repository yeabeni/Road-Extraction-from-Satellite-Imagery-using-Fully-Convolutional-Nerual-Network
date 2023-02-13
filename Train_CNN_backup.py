import glob
import os
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.style.use("ggplot")
#matplotlib inline

from tqdm import tqdm_notebook, tnrange
from itertools import chain
from skimage.io import imread, imshow, concatenate_images
from skimage.transform import resize
from skimage.morphology import label
from sklearn.model_selection import train_test_split

import tensorflow as tf
import keras.backend as K

from keras.models import Model, load_model, Sequential
from keras.layers import Input, BatchNormalization, Activation, Dense, Dropout
from keras.layers.core import RepeatVector
from keras.layers import Lambda, LeakyReLU, ZeroPadding2D, Reshape, Add, Cropping2D, Conv2DTranspose, Permute, Conv2D, MaxPooling2D

from keras.layers.pooling import GlobalMaxPool2D
from keras.layers.merge import concatenate, add
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
#%%

# Set some parameters
im_width = 224
im_height = 224
border = 5
path_train = './dataset/train/'
path_test = './dataset/valid/'

#%%
# Get and resize train images and masks
def get_data(path, train=True):
    ids = next(os.walk(path + "sat1"))[2]
    X = np.zeros((len(ids), im_height, im_width, 1), dtype=np.float32)
    if train:
        y = np.zeros((len(ids), im_height, im_width, 1), dtype=np.float32)
    print('Getting and resizing images ... ')
    for n, id_ in tqdm_notebook(enumerate(ids), total=len(ids)):
        # Load images
        img = load_img(path + '/sat1/' + id_, color_mode="grayscale")
        x_img = img_to_array(img)
        x_img = resize(x_img, (224, 224, 1), mode='constant', preserve_range=True)

        # Load masks
        if train:
            mask = img_to_array(load_img(path + '/map1/' + id_, color_mode="grayscale"))
            mask = resize(mask, (224, 224, 1), mode='constant', preserve_range=True)

        # Save images
        X[n, ..., 0] = x_img.squeeze() / 255
        if train:
            y[n] = mask / 255
    print('Done!')
    if train:
        return X, y
    else:
        return X

X, y = get_data(path_train, train=True)
#%%

# Split train and valid
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.15, random_state=2018)

#%%
# Check if training data looks all right
ix = random.randint(0, len(X_train))
has_mask = y_train[ix].max() > 0

fig, ax = plt.subplots(1, 2, figsize=(20, 10))

ax[0].imshow(X_train[ix, ..., 0], cmap='seismic', interpolation='bilinear')
if has_mask:
    ax[0].contour(y_train[ix].squeeze(), colors='k', levels=[0.5])
ax[0].set_title('Seismic')

ax[1].imshow(y_train[ix].squeeze(), interpolation='bilinear', cmap='gray')
ax[1].set_title('Salt');

#%%
#def conv2d_block(input_tensor, n_filters, kernel_size=3, batchnorm=True):
#    # first layer
#    x = Conv2D(filters=n_filters, kernel_size=(kernel_size, kernel_size), kernel_initializer="he_normal",
#               padding="same")(input_tensor)
#    if batchnorm:
#        x = BatchNormalization()(x)
#    x = Activation("relu")(x)
#    # second layer
#    x = Conv2D(filters=n_filters, kernel_size=(kernel_size, kernel_size), kernel_initializer="he_normal",
#               padding="same")(x)
#    if batchnorm:
#        x = BatchNormalization()(x)
#    x = Activation("relu")(x)
#    return x

#def get_unet(input_img, n_filters=16, dropout=0.5, batchnorm=True):
#    # contracting path
#    c1 = conv2d_block(input_img, n_filters=n_filters*1, kernel_size=3, batchnorm=batchnorm)
#    p1 = MaxPooling2D((2, 2)) (c1)
#    p1 = Dropout(dropout*0.5)(p1)
#
#    c2 = conv2d_block(p1, n_filters=n_filters*2, kernel_size=3, batchnorm=batchnorm)
#    p2 = MaxPooling2D((2, 2)) (c2)
#    p2 = Dropout(dropout)(p2)
#
#    c3 = conv2d_block(p2, n_filters=n_filters*4, kernel_size=3, batchnorm=batchnorm)
#    p3 = MaxPooling2D((2, 2)) (c3)
#    p3 = Dropout(dropout)(p3)
#
#    c4 = conv2d_block(p3, n_filters=n_filters*8, kernel_size=3, batchnorm=batchnorm)
#    p4 = MaxPooling2D(pool_size=(2, 2)) (c4)
#    p4 = Dropout(dropout)(p4)
#
#    c5 = conv2d_block(p4, n_filters=n_filters*16, kernel_size=3, batchnorm=batchnorm)
#
#    # expansive path
#    u6 = Conv2DTranspose(n_filters*8, (3, 3), strides=(2, 2), padding='same') (c5)
#    u6 = concatenate([u6, c4])
#    u6 = Dropout(dropout)(u6)
#    c6 = conv2d_block(u6, n_filters=n_filters*8, kernel_size=3, batchnorm=batchnorm)
#
#    u7 = Conv2DTranspose(n_filters*4, (3, 3), strides=(2, 2), padding='same') (c6)
#    u7 = concatenate([u7, c3])
#    u7 = Dropout(dropout)(u7)
#    c7 = conv2d_block(u7, n_filters=n_filters*4, kernel_size=3, batchnorm=batchnorm)
#
#    u8 = Conv2DTranspose(n_filters*2, (3, 3), strides=(2, 2), padding='same') (c7)
#    u8 = concatenate([u8, c2])
#    u8 = Dropout(dropout)(u8)
#    c8 = conv2d_block(u8, n_filters=n_filters*2, kernel_size=3, batchnorm=batchnorm)
#
#    u9 = Conv2DTranspose(n_filters*1, (3, 3), strides=(2, 2), padding='same') (c8)
#    u9 = concatenate([u9, c1], axis=3)
#    u9 = Dropout(dropout)(u9)
#    c9 = conv2d_block(u9, n_filters=n_filters*1, kernel_size=3, batchnorm=batchnorm)
#
#    outputs = Conv2D(1, (1, 1), activation='sigmoid') (c9)
#    model = Model(inputs=[input_img], outputs=[outputs])
#    return model
#%%
def model(train_vgg=False):
    """Defines a VGG-16 based FCN with one skip connection

    Args: train_vgg (bool, optional): False by default. Set to True if you would like to train the
                                      VGG-16 portion along with the upsampling portion.

    Returns: a Keras funtional model
    """
        #First half of model is the VGG-16 Network with the last layer removed

    #Input is 224x224 RGB images
    K.set_image_data_format( 'channels_last' )

    model = Sequential()     
    model.add(Permute((1,2,3), input_shape=(224, 224, 1)))
        
    model.add(Conv2D(64, kernel_size=(3,3), padding='same', activation='relu', name="conv1_1", trainable=train_vgg))
    model.add(Conv2D(64, kernel_size=(3,3), padding='same', activation='relu', name="conv1_2", trainable=train_vgg))
    model.add(MaxPooling2D((2,2), strides=(2,2)))
            
    model.add(Conv2D(128, kernel_size=(3,3), padding='same', activation='relu', name="conv2_1", trainable=train_vgg))
    model.add(Conv2D(128, kernel_size=(3,3), padding='same', activation='relu', name="conv2_2", trainable=train_vgg))
    model.add(MaxPooling2D((2,2), strides=(2,2)))
            
    model.add(Conv2D(256, kernel_size=(3,3), padding='same', activation='relu', name="conv3_1", trainable=train_vgg))
    model.add(Conv2D(256, kernel_size=(3,3), padding='same', activation='relu', name="conv3_2", trainable=train_vgg))
    model.add(Conv2D(256, kernel_size=(3,3), padding='same', activation='relu', name="conv3_3", trainable=train_vgg))
    model.add(MaxPooling2D((2,2), strides=(2,2)))
            
    model.add(Conv2D(512, kernel_size=(3,3), padding='same', activation='relu', name="conv4_1", trainable=train_vgg))
    model.add(Conv2D(512, kernel_size=(3,3), padding='same', activation='relu', name="conv4_2", trainable=train_vgg))
    model.add(Conv2D(512, kernel_size=(3,3), padding='same', activation='relu', name="conv4_3", trainable=train_vgg))
    model.add(MaxPooling2D((2,2), strides=(2,2)))
            
    model.add(Conv2D(512, kernel_size=(3,3), padding='same', activation='relu', name="conv5_1", trainable=train_vgg))
    model.add(Conv2D(512, kernel_size=(3,3), padding='same', activation='relu', name="conv5_2", trainable=train_vgg))
    model.add(Conv2D(512, kernel_size=(3,3), padding='same', activation='relu', name="conv5_3", trainable=train_vgg))
    model.add(MaxPooling2D((2,2), strides=(2,2)))
            
    model.add(Conv2D(4096, kernel_size=(7,7), padding='same', activation='relu', name='fc6', trainable=train_vgg))
    model.add(Conv2D(4096, kernel_size=(1,1), padding='same', activation='relu', name='fc7', trainable=train_vgg))

        #Second half of model upsamples output to a 224x224 grayscale image
    
    #reduce the number of channels to 1 (skip connection)
    model.add(Conv2D(1, kernel_size=(1, 1), padding='same', activation='relu', name='score_fr'))
#    model.add(ZeroPadding2D(padding=(1,1)))

    #First Convolution Transpose
    model.add(Conv2DTranspose(1, kernel_size=(4, 4), strides=(2, 2), padding='valid', name='upsample1'))
    model.add(LeakyReLU())

    #trim output for skip connection
    model.add(Cropping2D(cropping=((0, 2), (0, 2))))

    #Reduce number of channels of layer-14 to 1 so it can be added to the output of previous layer
    score14 = LeakyReLU()(Conv2D(1, kernel_size=(1,1), padding='same', name='score14')(model.layers[14].output))

    #Skip connection
    skip1 = Add()([score14, model.layers[-1].output])

    #Perform two more convolutional transposes
    up2 = LeakyReLU()(Conv2DTranspose(1,kernel_size=(8,8), strides=(4,4), padding='valid', name='upsample2')(skip1))
    crop_margin = Cropping2D(cropping=((2, 2), (2, 2)))(up2)

    up2 = LeakyReLU()(Conv2DTranspose(1,kernel_size=(8,8), strides=(4,4), padding='valid', name='upsample3')(crop_margin))
    crop_margin2 = Cropping2D(cropping=((2, 2), (2, 2)))(up2)

    #Flatten the output and remove the last dimensionm
    #model = Model(model.input, (Activation('sigmoid')(Reshape((224*224, 1))(crop_margin2))))

    outputs = Conv2D(1, (1, 1), activation='sigmoid')(crop_margin2)
    model = Model(model.input, outputs=[outputs])

    return model

#%%
#def image_generator(img_path, mask_path, img_dir_path, mask_dir_path):
#    """Function to create a image generator for fit_generator
#
#    https://keras.io/preprocessing/image/#imagedatagenerator-class
#
#    Args:
#        img_path (str): path to a single face image
#        mask_path (str): path to a single ground truth image
#        img_dir_path (str): path to a directory contaning training images
#        mask_dir_path (str): path to a directory containing ground truth training labels
#    
#    Returns:
#        a generator for fit_generator
#    """
##    data_gen_args = dict(rotation_range=20.,
##                     zoom_range=0.2,
##                    horizontal_flip=True)
#    image_datagen = ImageDataGenerator(
#            rotation_range=20,
#            rescale=1./255,
#            horizontal_flip=True,
#            zoom_range=0.2
#            )
#    
#    mask_datagen = ImageDataGenerator(
#            rotation_range=20,
#            rescale=1./255,
#            horizontal_flip=True,
#            zoom_range=0.2
#            )
##    image_datagen = ImageDataGenerator(**data_gen_args)
##    mask_datagen = ImageDataGenerator(**data_gen_args)
##    seed=1
##    image_datagen.fit(np.expand_dims(np.asarray(Image.open(img_path)), axis=0), augment=True, seed=seed)
##    mask_datagen.fit(np.expand_dims(np.expand_dims(np.asarray(Image.open(mask_path)), axis=0), axis = 4), augment=True, seed=seed)
#
#    image_generator = image_datagen.flow_from_directory(
#        img_dir_path,
#        class_mode="binary",
#        seed=1,
#        batch_size=1,
#        target_size=(224,224)
#        )
#
#    mask_generator = mask_datagen.flow_from_directory(
#        mask_dir_path,
#        class_mode="binary",
#        seed=1,color_mode='grayscale',
#        batch_size=1,
#        target_size=(224,224))
#
#    def gt_gen():
#        while True:
#            y = (mask_generator.next() / 255.).astype(np.float32)
#            y = y.reshape((y.shape[0], 224 * 224))
#            y[:,0] = 0.
#            y[:,1] = 1.
#            yield y
#
#    train_generator = zip(image_generator, gt_gen())
#
#    return train_generator 
#
#def valid_image_generator(img_path, mask_path, img_dir_path, mask_dir_path):
#    """Function to create a image generator for fit_generator
#
#    https://keras.io/preprocessing/image/#imagedatagenerator-class
#
#    Args:
#        img_path (str): path to a single face image
#        mask_path (str): path to a single ground truth image
#        img_dir_path (str): path to a directory contaning training images
#        mask_dir_path (str): path to a directory containing ground truth training labels
#    
#    Returns:
#        a generator for fit_generator
#    """
#    valid_datagen = ImageDataGenerator(
#        rotation_range=20,
#        rescale=1./255,
#        horizontal_flip=True,
#        zoom_range=0.2
#        )
#    
#    mask_valid_datagen = ImageDataGenerator(
#            rotation_range=20,
#            rescale=1./255,
#            horizontal_flip=True,
#            zoom_range=0.2
#            )
##    data_gen_args = dict(rotation_range=20.,
##                     zoom_range=0.2,
##                    horizontal_flip=True)
##    valid_datagen = ImageDataGenerator(**data_gen_args)
##    mask_valid_datagen = ImageDataGenerator(**data_gen_args)
##    seed=1
##    valid_datagen.fit(np.expand_dims(np.asarray(Image.open(img_path)), axis=0), augment=True, seed=seed)
##    mask_valid_datagen.fit(np.expand_dims(np.expand_dims(np.asarray(Image.open(mask_path)), axis=0), axis = 4), augment=True, seed=seed)
#
#    image_valid_generator = valid_datagen.flow_from_directory(
#        img_dir_path,
#        class_mode="binary",
#        seed=1,
#        batch_size=1,
#        target_size=(224,224)
#        )
#
#    mask_valid_generator = mask_valid_datagen.flow_from_directory(
#        mask_dir_path,
#        class_mode="binary",
#        seed=1,color_mode='grayscale',
#        batch_size=1,
#        target_size=(224,224))
#
#    def gt_gen():
#        while True:
#            y = (mask_valid_generator.next() / 255.).astype(np.float32)
#            y = y.reshape((y.shape[0], 224 * 224))
#            y[:,0] = 0.
#            y[:,1] = 1.
#            yield y
#
#    val_generator = zip(image_valid_generator, gt_gen())
#
#    return val_generator   
#  
##%%
#images = "./dataset/train/sat1"
#g_labels = "./dataset/train/map1"
#
#epochs = 150
#
#
#img_dir_path = images #args.imgs
#img_dir_name =  os.path.basename(os.path.normpath(img_dir_path))
#print(img_dir_path, img_dir_name)
#
#label_dir_path = g_labels #args.labels
#label_dir_name = os.path.basename(os.path.normpath(label_dir_path))
#print(label_dir_name)
#
##epochs = args.epochs
#
#imgs = glob.glob(img_dir_path + '/*')# + img_dir_name + '/*')
#labels = glob.glob(label_dir_path + '/*')# + label_dir_name + '/*')
#
#if len(imgs) != len(labels):
#    raise ValueError('Error: Different number of images and labels')
#
##if epochs is None:
##    epochs = 50
#
#img_path = imgs[0]
#label_path = labels[0]
#
#print(img_path)
#print(label_path)
##%%
#val_images = "./dataset/valid/sat1"
#val_g_labels = "./dataset/valid/map1"
#
#val_img_dir_name =  os.path.basename(os.path.normpath(val_images))
#print(val_img_dir_name)
#
#
#val_label_dir_name = os.path.basename(os.path.normpath(val_g_labels))
#print(label_dir_name)
#
##epochs = args.epochs
#
#val_imgs = glob.glob(val_images + '/*')# + img_dir_name + '/*')
#val_labels = glob.glob(val_g_labels + '/*')# + label_dir_name + '/*')
#
#if len(val_imgs) != len(val_labels):
#    raise ValueError('Error: Different number of images and labels')
#
##if epochs is None:
##    epochs = 50
#
#val_img_path = val_imgs[0]
#val_label_path = val_labels[0]
#
#print(val_img_path)
#print(val_label_path)

#%%
#input_img = Input((im_height, im_width, 1), name='img')
#model = get_unet(input_img, n_filters=16, dropout=0.05, batchnorm=True)
model = model()

model.compile(optimizer=Adam(), loss="binary_crossentropy", metrics=["accuracy"])
model.summary()

#train_generator = image_generator(img_path, label_path, img_dir_path, label_dir_path)
#valid_generator = valid_image_generator(val_img_path, val_label_path, val_images, val_g_labels)

#%%
callbacks = [
    EarlyStopping(patience=10, verbose=1),
    ReduceLROnPlateau(factor=0.1, patience=3, min_lr=0.00001, verbose=1),
    ModelCheckpoint('model-satelite.h5', verbose=1, save_best_only=True, save_weights_only=True)
]

history = model.fit(X_train, y_train, batch_size=10, epochs=50, callbacks=callbacks,
                    validation_data=(X_valid, y_valid))


#history = model.fit_generator(
#        train_generator,
#        steps_per_epoch=1108,
#        epochs = epochs,
#        verbose=1,
#        validation_data = valid_generator,
#        validation_steps=14,
#        callbacks=callbacks)

#%%
#%%
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title("Model Accuracy")
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train_acc', 'val_acc'], loc="lower right")
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title("Model Loss")
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train_loss', 'val_loss'], loc="upper right")
plt.show()

plt.style.use(['classic'])
#%%
# Load best model
model.load_weights('model-satelite.h5')
# Evaluate on validation set (this must be equals to the best log_loss)
model.evaluate(X_valid, y_valid, verbose=1)

#%%
# Predict on train, val and test
preds_train = model.predict(X_train, verbose=1)
preds_val = model.predict(X_valid, verbose=1)

# Threshold predictions
preds_train_t = (preds_train > 0.5).astype(np.uint8)
preds_val_t = (preds_val > 0.5).astype(np.uint8)
#%%
def plot_sample(X, y, preds, binary_preds, ix=None):
    if ix is None:
        ix = random.randint(0, len(X))

    has_mask = y[ix].max() > 0

    fig, ax = plt.subplots(1, 4, figsize=(20, 10))
    ax[0].imshow(X[ix, ..., 0], cmap='seismic')
    if has_mask:
        ax[0].contour(y[ix].squeeze(), colors='k', levels=[0.5])
    ax[0].set_title('Seismic')

    ax[1].imshow(y[ix].squeeze())
    ax[1].set_title('Salt')

    ax[2].imshow(preds[ix].squeeze(), vmin=0, vmax=1)
    if has_mask:
        ax[2].contour(y[ix].squeeze(), colors='k', levels=[0.5])
    ax[2].set_title('Salt Predicted')

    ax[3].imshow(binary_preds[ix].squeeze(), vmin=0, vmax=1)
    if has_mask:
        ax[3].contour(y[ix].squeeze(), colors='k', levels=[0.5])
    ax[3].set_title('Salt Predicted binary');

#%%
# Check if training data looks all right
plot_sample(X_train, y_train, preds_train, preds_train_t, ix=22)


