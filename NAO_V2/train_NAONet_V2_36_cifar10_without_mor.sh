# Train the architecture discovered by NAO-V2, with channel size of 36, noted as NAONet-V2-36
nvidia-smi
MODEL=NAONet_V2_36_cifar10
OUTPUT_DIR=exp/$MODEL
DATA_DIR=data

mkdir -p $OUTPUT_DIR
#discovered by our search space, the error=2.65%
fixed_arc="0 4 0 0 0 1 0 4 0 5 3 1 2 1 0 5 3 4 0 5 0 2 0 2 2 4 2 2 3 5 0 5 2 5 4 5 0 2 2 2" #discovered by ourself and added new operations
# fixed_arc="1 1 0 1 2 4 0 0 0 0 2 1 0 0 0 2 0 3 0 0 0 2 0 2 2 2 1 0 0 0 0 1 0 0 0 3 0 0 0 1" #original paper


python train_cifar.py \
  --data=$DATA_DIR \
  --output_dir=$OUTPUT_DIR \
  --arch="$fixed_arc" \
  --use_aux_head \
  --search_space small_without_mor\
  --cutout_size=16 | tee -a $OUTPUT_DIR/train.log
