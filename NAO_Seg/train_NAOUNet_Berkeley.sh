nvidia-smi
MODEL=NAONet_BSD_500
OUTPUT_DIR=exp/$MODEL
DATA_DIR=data

mkdir -p $OUTPUT_DIR
#with mor 

# fixed_arc="1 1 0 2 1 1 1 2 2 8 3 7 0 4 0 4 0 4 5 5 0 8 1 10 1 10 0 6 2 7 3 5 0 6 2 7 0 7 4 5 "
# fixed_arc="1 3 0 3 2 5 1 1 3 8 0 1 4 6 2 6 5 8 2 8 1 9 0 7 2 7 0 8 2 7 3 7 0 6 4 8 2 5 1 9 "
#without mor ops
fixed_arc="0 0 1 0 2 5 0 2 3 4 3 6 1 1 1 1 0 1 1 1 1 7 0 6 0 4 0 5 3 6 1 8 2 5 2 6 2 6 1 7 "
python train_BSD500.py \
  --data=$DATA_DIR \
  --arch="$fixed_arc" \
  --search_space without_mor_ops\
  --seed 0\
  --output_dir=$OUTPUT_DIR | tee -a $OUTPUT_DIR/train.log
