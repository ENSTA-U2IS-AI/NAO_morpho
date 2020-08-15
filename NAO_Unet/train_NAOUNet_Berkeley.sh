nvidia-smi
MODEL=NAONet_BSD_500
OUTPUT_DIR=exp/$MODEL
DATA_DIR=data

mkdir -p $OUTPUT_DIR
#discovered by our search space 43.63%
fixed_arc="0 4 1 4 2 5 0 3 0 4 0 0 4 8 2 8 5 5 5 6 1 10 0 8 0 8 1 9 2 5 0 6 2 5 3 7 0 8 2 6 "
python train_BSD500.py \
  --data=$DATA_DIR \
  --arch="$fixed_arc" \
  --output_dir=$OUTPUT_DIR | tee -a $OUTPUT_DIR/train.log
