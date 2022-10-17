cd ../src

../../parallel -j10 --resume-failed --results ../Output/1016ERO_proximal --joblog ../joblog/1016ERO_proximal CUDA_VISIBLE_DEVICES=1 python ./train.py --dataset ERO --N 350 --trainable_alpha   --hidden 8 --pretrain_with {1} --upset_ratio_coeff {2} --upset_margin_coeff {3} --eta {4} --ERO_style {5} --train_with {6} --p {7} --trainable_alpha --size_ratio 1 --all_methods all_GNNs -All ::: serial_similarity innerproduct dist ::: 1 0 ::: 1 0 ::: 0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8  ::: uniform gamma  ::: proximal_dist proximal_innerproduct ::: 0.05 1
