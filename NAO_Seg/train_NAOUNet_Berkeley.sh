nvidia-smi
MODEL=NAONet_BSD_500
OUTPUT_DIR=exp/$MODEL
DATA_DIR=data

mkdir -p $OUTPUT_DIR
#with mor --mor dilation
# fixed_arc="0 3 1 3 1 2 0 2 1 2 0 0 2 7 0 3 1 2 1 2 0 7 1 8 2 4 2 7 2 4 0 7 4 5 0 7 3 6 4 6 "
# -- mor gradient
# fixed_arc="1 0 0 0 2 6 0 2 3 5 3 5 0 0 1 2 1 3 0 3 0 7 1 8 1 8 1 8 1 8 2 5 4 6 1 8 2 6 1 8 "--S1
#without mor ops
# fixed_arc="0 0 1 0 2 5 0 2 3 4 3 6 1 1 1 1 0 1 1 1 1 7 0 6 0 4 0 5 3 6 1 8 2 5 2 6 2 6 1 7 "
python train_BSD500.py \
  --data=$DATA_DIR \
  --arch="$fixed_arc" \
  --search_space with_mor_ops\
  --seed 0\
  --output_dir=$OUTPUT_DIR | tee -a $OUTPUT_DIR/train.log
