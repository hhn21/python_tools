# FOCAL LOSS (train_on_multiple_gpus) ###########################
cd /home/vsocr/hanh/train_pl_focal_loss
git pull origin train_on_multiple_gpus
source activate pixel_link
export PYTHONPATH=$(pwd)/pylib/src/:$PYTHONPATH

bash scripts/continue_train_ha.sh 0 2 20200529_long_space_table 132 /home/vsocr/workspace_hieu_infordio3_ssd3/model_checkpoints/delivers/pixellink/conv2_2_20200108_step1_bg_smbc_dark_green_200108132257_selected/model.ckpt-180423
# TEST_PL (test_crrn)############################################
sshfs infordio3:/home/vsocr/hanh /home/vsocr/hanh

## normal ###################
sshfs infordio3:/home/vsocr/hanh /home/vsocr/hanh

cd /home/vsocr/hanh/test_crnn_normal/image-detection
cd /home/vsocr/hanh/printed_hw_test
cd /home/vsocr/hanh/test_crnn_central

## infordio4 ################
conda activate infor_pl

## 4GPU #####################
/home/aimenext/hanh/test_crnn
source activate infor_pl_py3.6

## set up ###################
source activate infor_pl_py3.6
export PYTHONPATH=$(pwd)/pixel_link/pylib/src/:$PYTHONPATH

## test pixel_link ##########
/mnt/disk1/common/OCR/pixellink_data
/home/vsocr/workspace_hieu/pixellink_data/vua
/home/vsocr/workspace_hieu_infordio3_ssd3/pixellink_train_data/hw_test

/home/vsocr/workspace_hieu_infordio3_ssd3/model_checkpoints/delivers
/home/vsocr/hanh/test_data/1img_test/no_gt/ 
/home/vsocr/workspace_hieu/pixellink_data/20200424_improve_bill

CUDA_VISIBLE_DEVICES=2 python -m infordio_ocr.dev_crnn_only /home/vsocr/hanh/test_data/line_bug /home/vsocr/hanh/test_results/line_bug

CUDA_VISIBLE_DEVICES=2 python -m infordio_ocr.dev_pl_pyramid_crnn_new /home/vsocr/hanh/test_data/20200819 /home/vsocr/hanh/test_results/20200819



<!-- CUDA_VISIBLE_DEVICES=2 python -m infordio_ocr.dev_pl_pyramid_binh /home/vsocr/workspace_hieu_infordio3_ssd3/pixellink_train_data/hw_test/imgs/ /home/vsocr/hanh/test_results -->
<!-- CUDA_VISIBLE_DEVICES= 0 or 1 for GPU 0 or 1 -->
<!-- CUDA_VISIBLE_DEVICES= any other than 0 and 1 for CPU. Ex: CUDA_VISIBLE_DEVICES=2 -->
<!-- dev_pl_pyramid_binh is for testing pixellink -->
<!-- file dev_pl_pyramid_crnn is for testing crnn và pixellink -->
<!-- /home/vsocr/workspace_hieu_infordio3_ssd3/pixellink_train_data/hw_test/imgs/ is test folder -->
<!-- /home/vsocr/hanh/test_results is for saving test result -->

## crnn_pl_config.cfg #######
pretrained_h<!-- to change pretrained horizontal model -->
pretrained_v<!-- to change pretrained vertical model -->
checkpoint<!-- to test from checkpoint -->
.
.
.
.
.
.
# TRAIN_CRNN (train_crnn_for_new_char)############################################
Train data:
/home/vsocr/thuntm/image-detection/train-crnn/image-detection/crnn/datasets/250619/
/mnt/disk1/common/OCR/crnn_data/250619
.
/home/vsocr/hanh/model_checkpoints/delivers/crnn/horizontal/
/home/vsocr/workspace_hieu_infordio3_ssd3/model_checkpoints/delivers/crnn/horizontal/
/home/vsocr/thuntm/image-detection/train-crnn/image-detection/crnn/datasets/250619/
/mnt/disk1/common/OCR/crnn_data/250619
/home/vsocr/thuntm/image-detection/train-crnn/image-detection/crnn/datasets/hand_writing/real_sequences
/home/vsocr/workspace_hieu_infordio3_ssd3/crnn_eval_data
/mnt/disk1/common/OCR/crnn_data/250619/
/mnt/disk1/common/OCR/crnn_data/crnn_eval_data


cd /home/vsocr/hanh/train_crnn/image-detection/crnn
cd /home/vsocr/hanh/train_crnn_hw/crnn
source activate crnn_train
export PYTHONPATH=../hyper_document_generator/:../

ln -s /home/vsocr/workplace/crnn/datasets ./
ln -s /home/vsocr/workplace/crnn_mask/resources ./

bash scripts/train/horizon/train_mask_crnn_raw_2_max_len_32_val_real_img_h_32_rand_invert_table_extra_real.sh 0 8 /home/vsocr/workspace_hieu_infordio3_ssd3/model_checkpoints/delivers/crnn/horizontal/_real_table_img_h_32_random_invert_crnn_raw_2_adam_relu_20200628233629_phong/weights.3650-0.93793.hdf5 3650


## Test_crnn_hw (train_hw) Test tổng hợp #################
cd /home/vsocr/hanh/train_crnn_hw/crnn
source activate crnn_train
export PYTHONPATH=../hyper_document_generator/:../

ln -s /home/vsocr/workplace/crnn/datasets ./
ln -s /home/vsocr/workplace/crnn_mask/resources ./

/home/vsocr/workspace_hieu_infordio3_ssd3/model_checkpoints/delivers/crnn/hw/_real_table_img_h_32_random_invert_crnn_raw_2_adam_relu_20200801120034_phong/weights.9184-10.95914.hdf5

/home/vsocr/workspace_hieu_infordio3_ssd3/model_checkpoints/delivers/crnn/horizontal/_real_table_img_h_32_random_invert_crnn_raw_2_adam_relu_20200814171039_phong_transaction_history/weights.4571-0.86753_transaction_history.hdf5



CUDA_VISIBLE_DEVICES=1 python image_ocr_ja_evaluate_model.py --pretrained /home/vsocr/thuntm/model_checkpoints/crnn/_real_table_img_h_32_random_invert_crnn_raw_2_adam_relu_20200816133158_phong/weights.4632-0.85326.hdf5 --pretrained_hw /home/vsocr/thuntm/model_checkpoints/crnn/_real_table_img_h_32_random_invert_crnn_raw_2_adam_relu_20200816133158_phong/weights.4632-0.85326.hdf5

CUDA_VISIBLE_DEVICES=0 python image_ocr_ja_evaluate_model.py --pretrained /home/vsocr/workspace_hieu_infordio3_ssd3/model_checkpoints/crnn/_real_table_img_h_32_random_invert_crnn_raw_2_adam_relu_20200814200204_phong/weights.10827-11.44860.hdf5 --pretrained_hw /home/vsocr/workspace_hieu_infordio3_ssd3/model_checkpoints/crnn/_real_table_img_h_32_random_invert_crnn_raw_2_adam_relu_20200814200204_phong/weights.10827-11.44860.hdf5

CUDA_VISIBLE_DEVICES=2 python image_ocr_ja_evaluate_model.py --pretrained /home/vsocr/workspace_hieu_infordio3_ssd3/model_checkpoints/delivers/crnn/horizontal/_real_table_img_h_32_random_invert_crnn_raw_2_adam_relu_20200817164240_phong/weights.4632-0.85326_transaction_history.hdf5 --pretrained_hw /home/vsocr/workspace_hieu_infordio3_ssd3/model_checkpoints/delivers/crnn/horizontal/_real_table_img_h_32_random_invert_crnn_raw_2_adam_relu_20200817164240_phong/weights.4632-0.85326_transaction_history.hdf5
<!-- /home/vsocr/workspace_hieu_infordio3_ssd3/model_checkpoints/delivers/crnn/hw/_real_table_img_h_32_random_invert_crnn_raw_2_adam_relu_20200628202531_phong/weights.5909-11.33030.hdf5 -->
.
.
.
.
.

# OTHERS#############################################
## ssh 4gpu  ################
ssh aimenext@118.70.72.252 -p 2468

scp -r -P 2468 20200721_improve_arrow  aimenext@118.70.72.252:/mnt/disk1/common/OCR/crnn_data/250619


scp -P 2468 aimenext@118.70.72.252:

## open on file
sftp://infordio3/home/vsocr/hanh/test_data/time_test

sftp://aimenext@10.10.10.90/

## mounting #################
sshfs infordio3:/home/vsocr/hanh /home/vsocr/hanh

## unmount ##################
fusermount -u /home/vsocr/hanh

## check mount ##############
df -h

## git ######################
192.168.121.99
phongnd@vnext.com.vn
tqd84Img
git remote set-url origin https://phongnd3482:tqd84Img@github.com/infordio/image-detection.git
.
.
.
.
.
.
# Infordio ###########################################
## Save models#################
/home/vsocr/workspace_hieu_infordio3_ssd3/model_checkpoints/pixellink/
## Train crnn #####################
/home/vsocr/thuntm/image-detection/train-crnn/image-detection/crnn/datasets/250619/

## train pl ##############
/home/vsocr/workspace_hieu_infordio3_ssd3/pixellink_train_data

## test pl ##############
/home/vsocr/workspace_hieu/pixellink_data/



# 4GPU ###############################################
## Train crnn #####################
/mnt/disk1/common/OCR/crnn_data/250619

/mnt/disk1/common/OCR/models/pixellink/conv2_2_20200117_all_real_kimono_laundry_0312_200117183302/model.ckpt-255044
sftp://aimenext@10.10.10.90/mnt/disk1/common/OCR/pixellink_train_data

# ####################################################

/home/vsocr/workspace_hieu/pixellink_data/vua


## hw #######################
/home/vsocr/workspace_hieu/train_hw/crnn/datasets/hand_writing/real_sequences/

## chữ in ###################
/home/vsocr/workspace_hieu_infordio3_ssd3/crnn_eval_data/20200626_printed_crossout_all/result_decoded_weights.10827-11.44860.csv

## questionaire #############
train:
/home/vsocr/workspace_hieu/train_hw/crnn/datasets/hand_writing/real_sequences/20200317_hw_questionaire
test:
/home/vsocr/workspace_hieu_infordio3_ssd3/crnn_eval_data/20200317_hw_questionaire_test

## test ###################
/home/vsocr/workspace_hieu/train_hw/crnn/datasets/hand_writing/real_sequences/20200320_receptionist/

/home/vsocr/workspace_hieu_infordio3_ssd3/crnn_eval_data/20200317_hw_questionaire_test/

/home/vsocr/workspace_hieu/train_hw/crnn/datasets/hand_writing/real_sequences/20200317_hw_questionaire/

# CHECK POINTS ######################################
## all deliver checkpoint ###
/home/vsocr/workspace_hieu_infordio3_ssd3/model_checkpoints/delivers/

## all checkpoint ###########
/home/vsocr/thuntm/pixellink_checkpoints_s3/*

# train_pl ###########################################
## set up
cd /home/vsocr/hanh/train_pl/image-detection
source activate pixel_link
export PYTHONPATH=$(pwd)/pylib/src/:$PYTHONPATH

## branches #################
git checkout modify_loss_4_incremental_learning

python datasets/genimg_to_tfrecords_multiple_folders.py "/home/vsocr/workspace_hieu_infordio3_ssd3/pixellink_train_data/long_space_table/ /home/vsocr/workspace_hieu_infordio3_ssd3/pixellink_train_data/hw_test/" /home/vsocr/hanh/train_data/20200531_long_space_table_hw_test

## continue_train_ha.sh
TRAIN_DIR=/home/vsocr/workspace_hieu_infordio3_ssd3/model_checkpoints/pixellink/conv2_2_${DATASET}

DATASET_DIR=/home/vsocr/hanh/train_data/${DATASET}

## Train
bash scripts/continue_train_ha.sh 0 2 20200611_long_space_table 132 /home/vsocr/workspace_hieu_infordio3_ssd3/model_checkpoints/delivers/pixellink/conv2_2_20200108_step1_bg_smbc_dark_green_200108132257_selected/model.ckpt-180423


/home/vsocr/workspace_hieu/pixellink_data/20200602_passbook
/home/vsocr/workspace_hieu_infordio3_ssd3/pixellink_train_data/20200424_improve_bill
/home/vsocr/workspace_hieu_infordio3_ssd3/pixellink_train_data/20191212_SMBC
/home/vsocr/workspace_hieu_infordio3_ssd3/pixellink_train_data/20191227_background
/home/vsocr/workspace_hieu_infordio3_ssd3/pixellink_train_data/real_hilmar_globalrunners
/home/vsocr/workspace_hieu_infordio3_ssd3/pixellink_train_data/real_green_bg
/home/vsocr/workspace_hieu_infordio3_ssd3/pixellink_train_data/10_dark_bg_synthesis
/home/vsocr/workspace_hieu_infordio3_ssd3/pixellink_train_data/20200531_long_space_table
## tf record ################
python datasets/genimg_to_tfrecords_multiple_folders.py "/home/vsocr/workspace_hieu/pixellink_data/20200602_passbook /home/vsocr/workspace_hieu_infordio3_ssd3/pixellink_train_data/20200424_improve_bill /home/vsocr/workspace_hieu_infordio3_ssd3/pixellink_train_data/20191212_SMBC /home/vsocr/workspace_hieu_infordio3_ssd3/pixellink_train_data/20191227_background /home/vsocr/workspace_hieu_infordio3_ssd3/pixellink_train_data/real_hilmar_globalrunners /home/vsocr/workspace_hieu_infordio3_ssd3/pixellink_train_data/real_green_bg /home/vsocr/workspace_hieu_infordio3_ssd3/pixellink_train_data/10_dark_bg_synthesis /home/vsocr/workspace_hieu_infordio3_ssd3/pixellink_train_data/20200531_long_space_table" /home/vsocr/hanh/train_data/20200611_long_space_table

# train_pl_weighted ##################################
## set up
cd /home/vsocr/hanh/train_pl_weighted/image-detection
source activate pixel_link
export PYTHONPATH=$(pwd)/pylib/src/:$PYTHONPATH

## config_190419.py line 139
weight_white_per_normal_bbox=5

## genimg_to_tfrecords_multiple_folders.py line 80
white_bboxes.append(config.white_bbox) if lbl=="w" else white_bboxes.append(config.normal_bbox)


# gen dữ liệu train_hw###############################
## set up ###################
cd /home/vsocr/hanh/train_hw
source activate crnn_train
export PYTHONPATH=../hyper_document_generator/:../
cd crnn
## config.py ################
BALANCING_VERSION = BALANCING_VERSION_0
USE_REAL_OR_PRE_GENERATED_IMGS=False

## paint.py #################
draw_text_v3_hw

bash scripts/train/horizon/train_mask_crnn_raw_2_max_len_32_val_real_img_h_32_rand_invert_table_extra_real.sh 0 16 /home/vsocr/workspace_hieu_infordio3_ssd3/model_checkpoints/crnn/_real_table_img_h_32_random_invert_crnn_raw_2_adam_relu_20200316112514_phong/weights.2649-1.21339.hdf5 2649


# gen image (crnn_train_resnet_vgg)##################
cd /home/vsocr/hanh/crnn_train_resnet_vgg
conda activate crnn_train
export PYTHONPATH=../hyper_document_generator/:../

paint.py


bash scripts/train/horizon/train_mask_crnn_raw_2_max_len_32_val_real_img_h_32_rand_invert_table_extra_real.sh 1 8 /home/vsocr/workspace_hieu_infordio3_ssd3/model_checkpoints/delivers/crnn/horizontal/_real_table_img_h_32_random_invert_phong2_crnn_raw_2_adam_relu_20191108200924/weights.1927-1.10477.hdf5 1927


# soft link: from crnn folder ########################
ln -s /home/vsocr/workplace/crnn/datasets ./
ln -s /home/vsocr/workplace/crnn_mask/resources/20_fonts resources/
ln -s /home/vsocr/workplace/crnn_mask/resources/dict resources/
ln -s /home/vsocr/workplace/crnn_mask/resources/fonts_number resources/
ln -s /home/vsocr/workplace/crnn_mask/resources/new_fonts resources/
ln -s /home/vsocr/workplace/crnn_mask/resources/train_japanese_sentences.txt resources/
ln -s /home/vsocr/workplace/crnn_mask/resources/calligraphy_font resources/
ln -s /home/vsocr/workplace/crnn_mask/resources/ominus_font resources/
ln -s /home/vsocr/workplace/crnn_mask/resources/real_sequences/ resources/
ln -s /home/vsocr/workplace/crnn_mask/resources/chars resources/


## from image-detection folder:
ln -s  /home/vsocr/workplace/hyper_document_generator_binhnq ./
ln -s  /home/vsocr/workplace/noto-cjk ./
ln -s /home/vsocr/workplace/datasets ./

# docker#############################################
##run

##show images
docker ps -a

# eval_pixellink (Tự đánh giá)#####################################
## setup
cd /home/vsocr/hanh/eval_pixel_link_v3.3
source activate infor_pl
export PYTHONPATH=$(pwd)/pylib/src/:$PYTHONPATH

## copy this to ckpt file
all_model_checkpoint_paths: "/home/vsocr/workspace_hieu_infordio3_ssd3/model_checkpoints/delivers/pixellink/conv2_2_20200117_all_real_kimono_laundry_0312_200117183302/model.ckpt-255044"

/home/vsocr/workspace_hieu_infordio3_ssd3/model_checkpoints/delivers/pixellink/conv2_2_20200108_step1_bg_smbc_dark_green_200108132257_selected/model.ckpt-180423
## copy 
scp -r /home/hhn21/Desktop/eval/gt.zip /home/hhn21/Desktop/eval/det.zip infordio3:/home/vsocr/hanh/eval_pixel_link_v3.3/infordio_ocr/eval_pixellink


bash infordio_ocr/eval_pixellink/run_eval_pl.sh 0 /home/vsocr/thuntm/pixellink_checkpoints_s3/conv2_2_20200428_all_real_bill_200430213150/checkpoint_ha /home/vsocr/workspace_hieu_infordio3_ssd3/pixellink_train_data/hw_test/gt_hw_test.zip /home/vsocr/workspace_hieu_infordio3_ssd3/pixellink_train_data/hw_test/imgs /home/vsocr/hanh/testo/20200505

/home/vsocr/workspace_hieu_infordio_ssd2/pixellink_train_data/20191204_hw_test
/home/vsocr/workspace_hieu_infordio_ssd2/pixellink_train_data/20191205_rakuten_sample



# auto_incremental_crnn##############################
## set up ###################
cd /home/vsocr/hanh/auto_incremental_crnn
source activate crnn_train
export PYTHONPATH=../hyper_document_generator/:../

## docker ###################
/home/vsocr/full_source
dock
## ssh forward port #########
ssh infordio3 -L 5000:0.0.0.0:5000

## soft link: from crnn folder:
ln -s /home/vsocr/workplace/crnn/datasets ./
ln -s /home/vsocr/workplace/crnn_mask/resources/20_fonts resources/
ln -s /home/vsocr/workplace/crnn_mask/resources/dict resources/
ln -s /home/vsocr/workplace/crnn_mask/resources/fonts_number resources/
ln -s /home/vsocr/workplace/crnn_mask/resources/new_fonts resources/
ln -s /home/vsocr/workplace/crnn_mask/resources/train_japanese_sentences.txt resources/
ln -s /home/vsocr/workplace/crnn_mask/resources/calligraphy_font resources/
ln -s /home/vsocr/workplace/crnn_mask/resources/ominus_font resources/
ln -s /home/vsocr/workplace/crnn_mask/resources/real_sequences/ resources/
ln -s /home/vsocr/workplace/crnn_mask/resources/chars resources/


## from image-detection folder:
ln -s  /home/vsocr/workplace/hyper_document_generator_binhnq ./
ln -s  /home/vsocr/workplace/noto-cjk ./
ln -s /home/vsocr/workplace/datasets ./