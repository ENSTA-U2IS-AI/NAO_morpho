nvidia-smi
MODEL=NAONet_BSD_500
OUTPUT_DIR=exp/$MODEL
DATA_DIR=data

mkdir -p $OUTPUT_DIR

# with mor seed2
fixed_arc="0 0 0 2 0 3 1 5 1 5 1 6 3 0 1 5 1 5 1 1"
# retry mor seed0
#fixed_arc="0 1 1 0 0 1 0 1 1 5 0 2 1 5 0 6 1 2 0 6"
CUDA_VISIBLE_DEVICES=1 python train_BSD500_aux.py \
  --data=$DATA_DIR \
  --arch="$fixed_arc" \
  --search_space with_mor_ops\
  --seed 0\
  --output_dir=$OUTPUT_DIR | tee -a $OUTPUT_DIR/train.log
