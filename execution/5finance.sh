cd ../src

../../parallel -j10 --resume-failed --results ../Output/finance_non_proximal --joblog ../joblog/finance_non_proximal CUDA_VISIBLE_DEVICES=5 python ./train.py --trainable_alpha --num_trials 10  --hidden 32 --train_with {1} --upset_ratio_coeff {2} --upset_margin_coeff {3}  --imbalance_coeff 0 --dataset finance  --all_methods all_GNNs -All ::: anchor_dist anchor_innerproduct ::: 1 0 ::: 1 0  

../../parallel -j10 --resume-failed --results ../Output/finance_proximal --joblog ../joblog/finance_proximal CUDA_VISIBLE_DEVICES=5 python ./train.py --trainable_alpha --num_trials 10  --hidden 32 --pretrain_with {1} --upset_ratio_coeff {2} --upset_margin_coeff {3}  --imbalance_coeff 0 --dataset finance --train_with {4} --all_methods all_GNNs -All ::: serial_similarity innerproduct dist ::: 1 0 ::: 1 0   ::: emb_dist emb_innerproduct

../../parallel -j10 --resume-failed --results ../Output/finance_proximal_baseline --joblog ../joblog/finance_proximal_baseline CUDA_VISIBLE_DEVICES=5 python ./train.py --trainable_alpha --num_trials 10  --hidden 32 --pretrain_with {1} --upset_ratio_coeff {2} --upset_margin_coeff {3} --imbalance_coeff 0 --train_with emb_baseline --dataset finance --cluster_rank_baseline {4} --all_methods all_GNNs -All ::: serial_similarity innerproduct dist ::: 1 0 ::: 1 0   ::: syncRank SpringRank btl serialRank eigenvectorCentrality PageRank SVD_NRS
