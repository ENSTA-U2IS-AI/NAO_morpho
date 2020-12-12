nvidia-smi
MODEL=NAONet_BSD_500
OUTPUT_DIR=exp/$MODEL
DATA_DIR=data

mkdir -p $OUTPUT_DIR
#with mor --mor dilation
# fixed_arc="0 3 1 3 1 2 0 2 1 2 0 0 2 7 0 3 1 2 1 2 0 7 1 8 2 4 2 7 2 4 0 7 4 5 0 7 3 6 4 6 "
# -- mor gradient
# fixed_arc="1 0 0 0 2 6 0 2 3 5 3 5 0 0 1 2 1 3 0 3 0 7 1 8 1 8 1 8 1 8 2 5 4 6 1 8 2 6 1 8 "--S1
# fixed_arc="1 2 0 0 1 1 1 0 1 0 1 0 1 1 1 0 1 0 1 0 0 7 1 8 1 8 1 8 1 8 1 8 1 8 1 8 1 8 1 8 "--S2
# fixed_arc="0 1 1 2 1 1 0 1 0 2 3 6 2 5 1 1 1 0 1 3 1 8 0 6 1 8 0 5 0 5 2 6 0 6 1 8 5 5 1 8 "--S3
# fixed_arc="1 0 0 1 1 3 2 4 1 1 3 7 4 5 4 5 2 5 5 7 0 7 1 8 2 7 2 6 2 4 3 6 2 7 0 5 4 7 0 7 "--S4
# fixed_arc="1 1 0 2 1 1 0 1 2 4 1 2 0 2 2 4 3 5 1 0 1 8 0 7 1 8 1 8 1 8 1 8 1 8 4 4 1 8 0 5 "--S5
fixed_arc="1 0 0 3 1 3 1 1 3 7 3 6 0 1 0 1 3 6 0 3 1 8 0 7 1 8 1 8 2 6 2 4 4 4 4 6 1 8 2 6 "
#without mor ops
# fixed_arc="0 0 1 0 2 5 0 2 3 4 3 6 1 1 1 1 0 1 1 1 1 7 0 6 0 4 0 5 3 6 1 8 2 5 2 6 2 6 1 7 "
# fixed_arc="1 2 0 2 1 2 2 6 0 1 1 1 1 2 4 6 3 5 0 3 1 7 0 4 2 4 2 4 3 6 2 5 4 4 0 5 1 8 5 5 "--S6
python train_BSD500_v2.py \
  --data=$DATA_DIR \
  --arch="$fixed_arc" \
  --search_space with_mor_ops\
  --seed 0\
  --output_dir=$OUTPUT_DIR | tee -a $OUTPUT_DIR/train.log
