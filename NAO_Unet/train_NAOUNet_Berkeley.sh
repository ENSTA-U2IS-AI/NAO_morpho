nvidia-smi
MODEL=NAONet_BSD_500
OUTPUT_DIR=exp/$MODEL
DATA_DIR=data

mkdir -p $OUTPUT_DIR
#discovered by our search space 28.31%
fixed_arc="0 1 1 1 2 6 2 8 2 5 3 8 3 6 1 4 3 8 3 6 1 9 0 6 1 10 0 7 0 8 3 5 2 8 2 8 1 10 4 6 "
python train_BSD500.py \
  --data=$DATA_DIR \
  --arch="$fixed_arc" \
  --output_dir=$OUTPUT_DIR | tee -a $OUTPUT_DIR/train.log
