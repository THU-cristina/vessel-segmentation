base_path = "C:\\Users\\crist.DESKTOP-ATO4N8C\\Desktop\\BA Final Code Ragusa\\"

datasets = base_path + "datasets\\"
results = base_path + "results\\"

train_imgs_original = "dataset_imgs_train.hdf5"
train_groundTruth = "dataset_groundTruth_train.hdf5"
train_border_masks = "dataset_borderMasks_train.hdf5"
test_imgs_original = "dataset_imgs_test.hdf5"
test_groundTruth = "dataset_groundTruth_test.hdf5"
test_border_masks = "dataset_borderMasks_test.hdf5"

# training values
pass_name = "BA_RAGUSA"

patch_height = 48
patch_width = 48

N_subimgs = 190000 
inside_FOV = False
N_epochs = 100
batch_size = 32

# test values
best_last = "best"
full_images_to_test = 10
n_group_visual = 1
average_mode = True
stride_height = 5
stride_width = 5
test_batch_size = 32

# prepare values
n_imgs_training = 20
n_imgs_testing = 10
channels = 3
height = 1200
width = 1600