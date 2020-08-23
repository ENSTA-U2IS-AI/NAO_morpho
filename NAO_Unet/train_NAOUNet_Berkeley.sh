nvidia-smi
MODEL=NAONet_BSD_500
OUTPUT_DIR=exp/$MODEL
DATA_DIR=data

mkdir -p $OUTPUT_DIR
#discovered by our search space 43.63%
fixed_arc="1 3 0 3 2 5 1 1 3 8 0 1 4 6 2 6 5 8 2 8 1 9 0 7 2 7 0 8 2 7 3 7 0 6 4 8 2 5 1 9 "
python train_BSD500.py \
  --data=$DATA_DIR \
  --arch="$fixed_arc" \
  --output_dir=$OUTPUT_DIR | tee -a $OUTPUT_DIR/train.log
