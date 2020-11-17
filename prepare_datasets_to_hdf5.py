import os
import h5py
import numpy as np
from PIL import Image

import config as cfg

def write_hdf5(arr, outfile):
    with h5py.File(outfile, "w") as f:
        f.create_dataset("image", data=arr, dtype=arr.dtype)


# train
original_imgs_train = cfg.datasets + "raw_images\\training\\images\\"
groundTruth_imgs_train = cfg.datasets + "raw_images\\training\\1st_manual\\"
borderMasks_imgs_train = cfg.datasets + "raw_images\\training\\mask\\"
# test
original_imgs_test = cfg.datasets + "raw_images\\test\\images\\"
groundTruth_imgs_test = cfg.datasets + "raw_images\\test\\1st_manual\\"
borderMasks_imgs_test = cfg.datasets + "raw_images\\test\\mask\\"


def get_datasets(imgs_dir, groundTruth_dir, borderMasks_dir, train_test="null", n_imgs=cfg.n_imgs_training):
    # for scaling img, mask and groundtruth
    scaled_width = int(cfg.width / 2)
    scaled_height = int(cfg.height / 2)
    
    imgs = np.empty((n_imgs, scaled_height, scaled_width, cfg.channels))
    groundTruth = np.empty((n_imgs, scaled_height, scaled_width))
    border_masks = np.empty((n_imgs, scaled_height, scaled_width))

    # list all files, directories in the path
    for path, subdirs, files in os.walk(imgs_dir):
        for i in range(len(files)):
            # original
            print("original image: " + files[i])
            print("full path: " + imgs_dir + files[i])
            img = Image.open(imgs_dir + files[i])

            # scale image
            img = img.resize((scaled_width, scaled_height))

            imgs[i] = np.asarray(img)
            # corresponding ground truth
            groundTruth_name = files[i][0:2] + "_manual1.gif"
            print("ground truth name: " + groundTruth_name)
            g_truth = Image.open(groundTruth_dir + groundTruth_name)

            # scale groundtruth
            g_truth = g_truth.resize((scaled_width, scaled_height))

            groundTruth[i] = np.asarray(g_truth)
            # corresponding border masks
            border_masks_name = ""
            if train_test == "train":
                border_masks_name = files[i][0:2] + "_training_mask.gif"
            elif train_test == "test":
                border_masks_name = files[i][0:2] + "_test_mask.gif"
            else:
                print("specify if train or test!!")
                exit()
            print("border masks name: " + border_masks_name)
            b_mask = Image.open(borderMasks_dir + border_masks_name)

            # scale border mask
            b_mask = b_mask.resize((scaled_width, scaled_height))

            border_masks[i] = np.asarray(b_mask)
            print("ground truth max this image: " +
                  str(np.max(groundTruth[i])))
    groundTruth[groundTruth > 0] = 255.0
    print("imgs max: " + str(np.max(imgs)))
    print("imgs min: " + str(np.min(imgs)))

    print("imgs max groundTruth: " + str(np.max(groundTruth)))
    print("imgs min groundTruth: " + str(np.min(groundTruth)))

    print("imgs max border_masks: " + str(np.max(border_masks)))
    print("imgs min border_masks: " + str(np.min(border_masks)))

    assert(np.min(groundTruth) == 0 and np.min(border_masks) == 0)
    print("ground truth and border masks are correctly withih pixel value range 0-255 (black-white)")

    imgs = np.transpose(imgs, (0, 3, 1, 2))
    assert(imgs.shape == (n_imgs, cfg.channels, scaled_height, scaled_width))
    groundTruth = np.reshape(groundTruth, (n_imgs, 1, scaled_height, scaled_width))
    border_masks = np.reshape(border_masks, (n_imgs, 1, scaled_height, scaled_width))
    assert(groundTruth.shape == (n_imgs, 1, scaled_height, scaled_width))
    assert(border_masks.shape == (n_imgs, 1, scaled_height, scaled_width))
    return imgs, groundTruth, border_masks


# getting the training datasets
imgs_train, groundTruth_train, border_masks_train = get_datasets(original_imgs_train, groundTruth_imgs_train, borderMasks_imgs_train, "train", n_imgs=cfg.n_imgs_training)
print("saving train datasets")
write_hdf5(imgs_train, cfg.datasets + cfg.train_imgs_original)
write_hdf5(groundTruth_train, cfg.datasets + cfg.train_groundTruth)
write_hdf5(border_masks_train, cfg.datasets + cfg.train_border_masks)

# getting the testing datasets
imgs_test, groundTruth_test, border_masks_test = get_datasets(original_imgs_test, groundTruth_imgs_test, borderMasks_imgs_test, "test", n_imgs=cfg.n_imgs_testing)
print("training images shape: " + str(imgs_train.shape))
print("test images shape: " + str(imgs_test.shape))
print("saving test datasets")
write_hdf5(imgs_test, cfg.datasets + cfg.test_imgs_original)
write_hdf5(groundTruth_test, cfg.datasets + cfg.test_groundTruth)
write_hdf5(border_masks_test, cfg.datasets + cfg.test_border_masks)
