cd ../src

../../parallel -j4 --resume-failed --results ../Output/HeadToHead_non_proximal --joblog ../joblog/HeadToHead_non_proximal CUDA_VISIBLE_DEVICES=5 python ./train.py  --num_trials 10  --hidden 16 --train_with {1} --upset_ratio_coeff {2} --upset_margin_coeff {3}  --dataset HeadToHead  --all_methods all_GNNs -All ::: dist innerproduct ::: 1 0 ::: 1 0  

../../parallel -j4 --resume-failed --results ../Output/HeadToHead_proximal --joblog ../joblog/HeadToHead_proximal CUDA_VISIBLE_DEVICES=5 python ./train.py --trainable_alpha --num_trials 10  --hidden 16 --pretrain_with {1} --upset_ratio_coeff {2} --upset_margin_coeff {3}  --dataset HeadToHead --train_with {4} --all_methods all_GNNs -All ::: serial_similarity innerproduct dist ::: 1 0 ::: 1 0   ::: proximal_dist proximal_innerproduct

../../parallel -j4 --resume-failed --results ../Output/HeadToHead_proximal_baseline --joblog ../joblog/HeadToHead_proximal_baseline CUDA_VISIBLE_DEVICES=5 python ./train.py --trainable_alpha --num_trials 10  --hidden 16 --pretrain_with {1} --upset_ratio_coeff {2} --upset_margin_coeff {3} --train_with proximal_baseline --dataset HeadToHead --cluster_rank_baseline {4} --all_methods all_GNNs -All ::: serial_similarity innerproduct dist ::: 1 0 ::: 1 0   ::: syncRank SpringRank btl serialRank eigenvectorCentrality PageRank SVD_NRS
