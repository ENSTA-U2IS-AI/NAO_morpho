MODEL=search_NAO_deeplab
OUTPUT_DIR=exp/$MODEL

mkdir -p $OUTPUT_DIR

CUDA_VISIBLE_DEVICES=0 python train_NAO_deeplabv3plus_search.py --search_space with_mor_ops\
    --seed 0\
    --output_dir=$OUTPUT_DIR | tee -a $OUTPUT_DIR/train.log \
    > train_deeplab_search.out
