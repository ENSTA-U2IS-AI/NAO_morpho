MODEL=search_BSD500
OUTPUT_DIR=exp/$MODEL

mkdir -p $OUTPUT_DIR

python train_search.py --search_space small_with_mor \
        --output_dir=$OUTPUT_DIR | tee -a $OUTPUT_DIR/train.log
