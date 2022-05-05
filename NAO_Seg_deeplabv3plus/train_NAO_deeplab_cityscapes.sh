nvidia-smi
MODEL=NAO_deeplabv3plus_cityscapes
OUTPUT_DIR=exp/$MODEL
DATA_DIR=data

mkdir -p $OUTPUT_DIR

# with mor seed0 100-30
fixed_arc="0 5 0 5 2 3 1 1 3 1 2 6 4 2 2 3 0 5 4 2"
# with mor seed1 150-50
fixed_arc="1 2 1 3 1 5 1 5 3 3 2 4 3 2 3 4 2 4 1 4"

CUDA_VISIBLE_DEVICES=0,1 python train_NAO_deeplabv3plus.py \
  --data=$DATA_DIR \
  --data_root "/home/student/workspace_Yufei/CityScapes/leftImg8bit/" \
  --arch="$fixed_arc" \
  --model "deeplabv3plus_resnet50"\
  --output_stride 8 \
  --batch_size 16 \
  --crop_size 768 \
  --gpu_id 0,1 \
  --lr 0.1 \
  --search_space with_mor_ops\
  --seed 0\
  --output_dir=$OUTPUT_DIR | tee -a $OUTPUT_DIR/train.log \
  > train_deeplabv3plus_cityscapes.out
