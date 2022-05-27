nvidia-smi
MODEL=NAO_deeplabv3plus_cityscapes
OUTPUT_DIR=exp/$MODEL
DATA_DIR=data

mkdir -p $OUTPUT_DIR

# wo res50 12
fixed_arc="0 6 1 0 0 2 0 4 0 0 3 4 4 4 2 6 0 3 1 5"

CUDA_VISIBLE_DEVICES=0 python train_NAO_deeplabv3plus.py \
  --data_root "/home/student/workspace_Yufei/CityScapes/NAO_Cityscapes" \
  --arch="$fixed_arc" \
  --model "deeplabv3plus_resnet50"\
  --output_stride 8 \
  --batch_size 12 \
  --crop_size 768 \
  --gpu_id 0,1 \
  --lr 0.1 \
  --search_space with_mor_ops\
  --seed 0\
  --ckpt "./checkpoints_resnet50_with_mo_arc_300_100_best_r2_bad/best_deeplabv3plus_resnet50_cityscapes_os8.pth" \
  --test_only \
  --save_val_results \
  --output_dir=$OUTPUT_DIR | tee -a $OUTPUT_DIR/train.log \
  > test.out
