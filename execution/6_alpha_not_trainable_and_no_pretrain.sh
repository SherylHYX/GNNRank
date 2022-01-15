cd ../src 

# alpha not trainable
# faculty
../../parallel -j10 --resume-failed --results ../Output/faculty_alpha_not_trainable --joblog ../joblog/faculty_alpha_not_trainable CUDA_VISIBLE_DEVICES=6 python ./train.py   --num_trials 10  --hidden 8 --pretrain_with {1} --upset_ratio_coeff {2} --upset_margin_coeff {3}  --imbalance_coeff 0 --dataset {4} --train_with {5} --all_methods all_GNNs -All ::: serial_similarity innerproduct dist ::: 1 0 ::: 1 0  ::: faculty_business faculty_cs faculty_history  ::: emb_dist emb_innerproduct

../../parallel -j10 --resume-failed --results ../Output/faculty_alpha_not_trainable_baseline --joblog ../joblog/faculty_alpha_not_trainable_baseline CUDA_VISIBLE_DEVICES=6 python ./train.py   --num_trials 10  --hidden 8 --pretrain_with {1} --upset_ratio_coeff {2} --upset_margin_coeff {3} --imbalance_coeff 0 --train_with emb_baseline --dataset {4} --cluster_rank_baseline {5} --all_methods all_GNNs -All ::: serial_similarity innerproduct dist ::: 1 0 ::: 1 0  ::: faculty_business faculty_cs faculty_history  ::: syncRank SpringRank btl serialRank eigenvectorCentrality PageRank SVD_NRS

# animal
../../parallel -j10 --resume-failed --results ../Output/animal_alpha_not_trainable --joblog ../joblog/animal_alpha_not_trainable CUDA_VISIBLE_DEVICES=6 python ./train.py   --num_trials 10  --hidden 4 --pretrain_with {1} --upset_ratio_coeff {2} --upset_margin_coeff {3}  --imbalance_coeff 0 --dataset animal --train_with {4} --all_methods all_GNNs -All ::: serial_similarity innerproduct dist ::: 1 0 ::: 1 0   ::: emb_dist emb_innerproduct

../../parallel -j10 --resume-failed --results ../Output/animal_alpha_not_trainable_baseline --joblog ../joblog/animal_alpha_not_trainable_baseline CUDA_VISIBLE_DEVICES=6 python ./train.py   --num_trials 10  --hidden 4 --pretrain_with {1} --upset_ratio_coeff {2} --upset_margin_coeff {3} --imbalance_coeff 0 --train_with emb_baseline --dataset animal --cluster_rank_baseline {4} --all_methods all_GNNs -All ::: serial_similarity innerproduct dist ::: 1 0 ::: 1 0   ::: syncRank SpringRank btl serialRank eigenvectorCentrality PageRank SVD_NRS

# football
../../parallel -j10 --resume-failed --results ../Output/football_alpha_not_trainable --joblog ../joblog/football_alpha_not_trainable CUDA_VISIBLE_DEVICES=6 python ./train.py   --num_trials 10  --hidden 4 --pretrain_with {1} --upset_ratio_coeff {2} --upset_margin_coeff {3} --season {4} --imbalance_coeff 0 --dataset {5} --train_with {6} --all_methods all_GNNs -All ::: serial_similarity innerproduct dist ::: 1 0 ::: 1 0 ::: {2009..2014}  ::: football finer_football  ::: emb_dist emb_innerproduct

../../parallel -j10 --resume-failed --results ../Output/football_alpha_not_trainable_baseline --joblog ../joblog/football_alpha_not_trainable_baseline CUDA_VISIBLE_DEVICES=6 python ./train.py   --num_trials 10  --hidden 4 --pretrain_with {1} --upset_ratio_coeff {2} --upset_margin_coeff {3} --season {4} --imbalance_coeff 0 --train_with emb_baseline --dataset {5} --cluster_rank_baseline {6} --all_methods all_GNNs -All ::: serial_similarity innerproduct dist ::: 1 0 ::: 1 0 ::: {2009..2014}  ::: football finer_football  ::: syncRank SpringRank btl serialRank eigenvectorCentrality PageRank SVD_NRS

# no pretrain
# faculty
../../parallel -j10 --resume-failed --results ../Output/faculty_no_pretrain --joblog ../joblog/faculty_no_pretrain CUDA_VISIBLE_DEVICES=6 python ./train.py   --num_trials 10  --hidden 8 -- pretrain_epochs 0 --upset_ratio_coeff {1} --upset_margin_coeff {2}  --imbalance_coeff 0 --dataset {3} --train_with {4} --all_methods all_GNNs -All  ::: 1 0 ::: 1 0  ::: faculty_business faculty_cs faculty_history  ::: emb_dist emb_innerproduct

../../parallel -j10 --resume-failed --results ../Output/faculty_no_pretrain_baseline --joblog ../joblog/faculty_no_pretrain_baseline CUDA_VISIBLE_DEVICES=6 python ./train.py   --num_trials 10  --hidden 8 -- pretrain_epochs 0 --upset_ratio_coeff {1} --upset_margin_coeff {2} --imbalance_coeff 0 --train_with emb_baseline --dataset {3} --cluster_rank_baseline {4} --all_methods all_GNNs -All  ::: 1 0 ::: 1 0  ::: faculty_business faculty_cs faculty_history  ::: syncRank SpringRank btl serialRank eigenvectorCentrality PageRank SVD_NRS

# animal
../../parallel -j10 --resume-failed --results ../Output/animal_no_pretrain --joblog ../joblog/animal_no_pretrain CUDA_VISIBLE_DEVICES=6 python ./train.py   --num_trials 10  --hidden 4 -- pretrain_epochs 0 --upset_ratio_coeff {1} --upset_margin_coeff {2}  --imbalance_coeff 0 --dataset animal --train_with {3} --all_methods all_GNNs -All  ::: 1 0 ::: 1 0   ::: emb_dist emb_innerproduct

../../parallel -j10 --resume-failed --results ../Output/animal_no_pretrain_baseline --joblog ../joblog/animal_no_pretrain_baseline CUDA_VISIBLE_DEVICES=6 python ./train.py   --num_trials 10  --hidden 4 -- pretrain_epochs 0 --upset_ratio_coeff {1} --upset_margin_coeff {2} --imbalance_coeff 0 --train_with emb_baseline --dataset animal --cluster_rank_baseline {3} --all_methods all_GNNs -All  ::: 1 0 ::: 1 0   ::: syncRank SpringRank btl serialRank eigenvectorCentrality PageRank SVD_NRS

# football
../../parallel -j10 --resume-failed --results ../Output/football_no_pretrain --joblog ../joblog/football_no_pretrain CUDA_VISIBLE_DEVICES=6 python ./train.py   --num_trials 10  --hidden 4 -- pretrain_epochs 0 --upset_ratio_coeff {1} --upset_margin_coeff {2} --season {3} --imbalance_coeff 0 --dataset {4} --train_with {5} --all_methods all_GNNs -All  ::: 1 0 ::: 1 0 ::: {2009..2014}  ::: football finer_football  ::: emb_dist emb_innerproduct

../../parallel -j10 --resume-failed --results ../Output/football_no_pretrain_baseline --joblog ../joblog/football_no_pretrain_baseline CUDA_VISIBLE_DEVICES=6 python ./train.py   --num_trials 10  --hidden 4 -- pretrain_epochs 0 --upset_ratio_coeff {1} --upset_margin_coeff {2} --season {3} --imbalance_coeff 0 --train_with emb_baseline --dataset {4} --cluster_rank_baseline {5} --all_methods all_GNNs -All  ::: 1 0 ::: 1 0 ::: {2009..2014}  ::: football finer_football  ::: syncRank SpringRank btl serialRank eigenvectorCentrality PageRank SVD_NRS