nvidia-smi
MODEL=NAONet_BSD_500
OUTPUT_DIR=exp/$MODEL
DATA_DIR=data

mkdir -p $OUTPUT_DIR
# -- mor gradient seed 2
# fixed_arc="1 3 0 0 2 6 0 2 1 0 1 0 1 0 1 2 2 4 4 5 0 7 1 8 2 5 2 4 2 6 0 5 1 8 2 6 0 5 2 6 "

# -- mor gradient seed 1
# fixed_arc="0 3 1 3 1 1 0 3 2 6 3 6 3 7 2 5 2 5 4 5 1 8 0 7 1 8 2 5 1 8 0 6 0 6 1 8 4 4 3 4 "

# -- mor gradient seed 0
# fixed_arc="0 2 1 2 0 0 0 1 1 1 1 0 1 0 3 5 4 7 5 4 1 8 0 7 2 6 1 8 2 6 3 5 0 5 4 7 1 8 0 5 "

# -- wihout mor seed 0
# fixed_arc="1 1 0 1 2 4 0 1 0 1 2 4 0 1 4 4 2 4 4 4 1 8 0 4 2 4 2 4 3 4 3 4 4 4 4 4 2 4 4 4 "

# # -- wihout mor seed 1
fixed_arc="0 3 1 0 1 0 1 2 2 4 0 0 1 0 0 0 2 4 4 4 1 8 0 6 2 4 2 4 2 4 2 4 2 4 2 4 2 4 5 4 "

# # -- wihout mor seed 2
# fixed_arc="1 3 0 1 0 1 2 4 2 4 2 4 2 4 0 1 4 4 2 4 1 8 0 6 2 4 2 4 2 4 2 4 2 4 2 4 2 6 4 6 "

python train_BSD500.py \
  --data=$DATA_DIR \
  --arch="$fixed_arc" \
  --search_space without_mor_ops\
  --seed 0\
  --output_dir=$OUTPUT_DIR | tee -a $OUTPUT_DIR/train.log
