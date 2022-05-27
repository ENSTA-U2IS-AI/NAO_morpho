nvidia-smi
MODEL=NAONet_BSD_500
OUTPUT_DIR=exp/$MODEL
DATA_DIR=data

mkdir -p $OUTPUT_DIR

# arch for edge detection bsd500
fixed_arc="0 0 0 2 0 3 1 5 1 5 1 6 3 0 1 5 1 5 1 1"

python test_BSD500.py \
  --data=$DATA_DIR \
  --arch="$fixed_arc" \
  --search_space with_mor_ops\
  --seed 0\
  --output_dir=$OUTPUT_DIR | tee -a $OUTPUT_DIR/train.log