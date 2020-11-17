import numpy as np

import config as cfg

from keras.models import Model
from keras.layers import Input, concatenate, Conv2D, MaxPooling2D, UpSampling2D, Reshape, core, Dropout
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras import backend as K
from keras.utils.vis_utils import plot_model as plot
from keras.optimizers import SGD


from help_functions import *
#from help_functions import visualize
#from help_functions import group_images
#from help_functions import masks_Unet

# function to obtain data for training/testing (validation)
from extract_patches import get_data_training


# Define the neural network
def get_unet(n_channels, patch_height, patch_width):
    inputs = Input(shape=(n_channels, patch_height, patch_width))
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same',
                   data_format='channels_first')(inputs)
    conv1 = Dropout(0.2)(conv1)
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same',
                   data_format='channels_first')(conv1)
    pool1 = MaxPooling2D((2, 2))(conv1)

    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same',
                   data_format='channels_first')(pool1)
    conv2 = Dropout(0.2)(conv2)
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same',
                   data_format='channels_first')(conv2)
    pool2 = MaxPooling2D((2, 2))(conv2)

    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same',
                   data_format='channels_first')(pool2)
    conv3 = Dropout(0.2)(conv3)
    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same',
                   data_format='channels_first')(conv3)

    up1 = UpSampling2D(size=(2, 2))(conv3)
    up1 = concatenate([conv2, up1], axis=1)
    conv4 = Conv2D(64, (3, 3), activation='relu', padding='same',
                   data_format='channels_first')(up1)
    conv4 = Dropout(0.2)(conv4)
    conv4 = Conv2D(64, (3, 3), activation='relu', padding='same',
                   data_format='channels_first')(conv4)

    up2 = UpSampling2D(size=(2, 2))(conv4)
    up2 = concatenate([conv1, up2], axis=1)
    conv5 = Conv2D(32, (3, 3), activation='relu', padding='same',
                   data_format='channels_first')(up2)
    conv5 = Dropout(0.2)(conv5)
    conv5 = Conv2D(32, (3, 3), activation='relu', padding='same',
                   data_format='channels_first')(conv5)

    conv6 = Conv2D(2, (1, 1), activation='relu', padding='same',
                   data_format='channels_first')(conv5)
    conv6 = core.Reshape((2, patch_height*patch_width))(conv6)
    conv6 = core.Permute((2, 1))(conv6)

    conv7 = core.Activation('softmax')(conv6)

    model = Model(inputs=inputs, outputs=conv7)

    model.compile(optimizer='sgd', loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model


# ============ Load the data and divided in patches
patches_imgs_train, patches_masks_train = get_data_training(
    DRIVE_train_imgs_original=cfg.datasets + cfg.train_imgs_original,
    DRIVE_train_groudTruth=cfg.datasets + cfg.train_groundTruth,  # masks
    patch_height=cfg.patch_height,
    patch_width=cfg.patch_width,
    N_subimgs=cfg.N_subimgs,
    inside_FOV=cfg.inside_FOV
)


# ========= Save a sample of what you're feeding to the neural network ==========
n_sample = min(patches_imgs_train.shape[0], 40)
visualize(group_images(patches_imgs_train[0:n_sample, :, :, :], 5), cfg.results  + "sample_input_imgs")
visualize(group_images(patches_masks_train[0:n_sample, :, :, :], 5), cfg.results + "sample_input_masks")


# =========== Construct and save the model arcitecture =====
n_channels = patches_imgs_train.shape[1]
patch_height = patches_imgs_train.shape[2]
patch_width = patches_imgs_train.shape[3]
model = get_unet(n_channels, patch_height, patch_width)  # the U-net model
print("Check: final output of the network:")
print(model.output_shape)
json_string = model.to_json()
open(cfg.results + cfg.pass_name + '_architecture.json', 'w').write(json_string)

# ============  Training ==================================
checkpointer = ModelCheckpoint(filepath= cfg.results + cfg.pass_name + '_best_weights.h5',
                               verbose=1, monitor='val_loss', mode='auto', save_best_only=True)  # save at each epoch if the validation decreased

patches_masks_train = masks_Unet(patches_masks_train)
model.fit(patches_imgs_train, patches_masks_train, epochs=cfg.N_epochs, batch_size=cfg.batch_size,
          verbose=2, shuffle=True, validation_split=0.1, callbacks=[checkpointer])

# ========== Save and test the last model ===================
model.save_weights(cfg.results +
                   cfg.pass_name + '_last_weights.h5', overwrite=True)
