python build_hard_negative_dataset.py \
--pred-dir validation_results/e20 \
--train-img-dir data/validation/images \
--train-label-dir data/validation/labels \
--output-dir data/training_hn \
--threshold 0.1 \
--enable-hard-negative \
--hard-negative-ratio 0.3 \
--pos-ratio 0.4 \
--edge-ratio 0.1