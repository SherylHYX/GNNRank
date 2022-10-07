cd ../src 

# faculty
../../parallel -j10 --resume-failed --results ../Output/faculty_non_proximal --joblog ../joblog/faculty_non_proximal CUDA_VISIBLE_DEVICES=4 python ./train.py  --num_trials 10  --hidden 8 --train_with {1} --upset_ratio_coeff {2} --upset_margin_coeff {3}  --dataset {4}  --all_methods all_GNNs -All ::: dist innerproduct ::: 1 0 ::: 1 0   ::: faculty_business faculty_cs faculty_history

../../parallel -j10 --resume-failed --results ../Output/faculty_proximal --joblog ../joblog/faculty_proximal CUDA_VISIBLE_DEVICES=4 python ./train.py --trainable_alpha --num_trials 10  --hidden 8 --pretrain_with {1} --upset_ratio_coeff {2} --upset_margin_coeff {3}  --dataset {4} --train_with {5} --all_methods all_GNNs -All ::: serial_similarity innerproduct dist ::: 1 0 ::: 1 0  ::: faculty_business faculty_cs faculty_history  ::: proximal_dist proximal_innerproduct

../../parallel -j10 --resume-failed --results ../Output/faculty_proximal_baseline --joblog ../joblog/faculty_proximal_baseline CUDA_VISIBLE_DEVICES=4 python ./train.py --trainable_alpha --num_trials 10  --hidden 8 --pretrain_with {1} --upset_ratio_coeff {2} --upset_margin_coeff {3} --train_with proximal_baseline --dataset {4} --baseline {5} --all_methods all_GNNs -All ::: serial_similarity innerproduct dist ::: 1 0 ::: 1 0  ::: faculty_business faculty_cs faculty_history  ::: syncRank SpringRank btl serialRank eigenvectorCentrality PageRank SVD_NRS

# animal
../../parallel -j10 --resume-failed --results ../Output/animal_non_proximal --joblog ../joblog/animal_non_proximal CUDA_VISIBLE_DEVICES=4 python ./train.py  --num_trials 10  --hidden 4 --train_with {1} --upset_ratio_coeff {2} --upset_margin_coeff {3}  --dataset animal  --all_methods all_GNNs -All ::: dist innerproduct ::: 1 0 ::: 1 0  

../../parallel -j10 --resume-failed --results ../Output/animal_proximal --joblog ../joblog/animal_proximal CUDA_VISIBLE_DEVICES=4 python ./train.py --trainable_alpha --num_trials 10  --hidden 4 --pretrain_with {1} --upset_ratio_coeff {2} --upset_margin_coeff {3}  --dataset animal --train_with {4} --all_methods all_GNNs -All ::: serial_similarity innerproduct dist ::: 1 0 ::: 1 0   ::: proximal_dist proximal_innerproduct

../../parallel -j10 --resume-failed --results ../Output/animal_proximal_baseline --joblog ../joblog/animal_proximal_baseline CUDA_VISIBLE_DEVICES=4 python ./train.py --trainable_alpha --num_trials 10  --hidden 4 --pretrain_with {1} --upset_ratio_coeff {2} --upset_margin_coeff {3} --train_with proximal_baseline --dataset animal --baseline {4} --all_methods all_GNNs -All ::: serial_similarity innerproduct dist ::: 1 0 ::: 1 0   ::: syncRank SpringRank btl serialRank eigenvectorCentrality PageRank SVD_NRS

# football
../../parallel -j10 --resume-failed --results ../Output/football_non_proximal --joblog ../joblog/football_non_proximal CUDA_VISIBLE_DEVICES=4 python ./train.py  --num_trials 10  --hidden 4 --train_with {1} --upset_ratio_coeff {2} --upset_margin_coeff {3} --season {4} --dataset {5}  --all_methods all_GNNs -All ::: dist innerproduct ::: 1 0 ::: 1 0 ::: {2009..2014}  ::: football finer_football

../../parallel -j10 --resume-failed --results ../Output/football_proximal --joblog ../joblog/football_proximal CUDA_VISIBLE_DEVICES=4 python ./train.py --trainable_alpha --num_trials 10  --hidden 4 --pretrain_with {1} --upset_ratio_coeff {2} --upset_margin_coeff {3} --season {4} --dataset {5} --train_with {6} --all_methods all_GNNs -All ::: serial_similarity innerproduct dist ::: 1 0 ::: 1 0 ::: {2009..2014}  ::: football finer_football  ::: proximal_dist proximal_innerproduct

../../parallel -j10 --resume-failed --results ../Output/football_proximal_baseline --joblog ../joblog/football_proximal_baseline CUDA_VISIBLE_DEVICES=4 python ./train.py --trainable_alpha --num_trials 10  --hidden 4 --pretrain_with {1} --upset_ratio_coeff {2} --upset_margin_coeff {3} --season {4} --train_with proximal_baseline --dataset {5} --baseline {6} --all_methods all_GNNs -All ::: serial_similarity innerproduct dist ::: 1 0 ::: 1 0 ::: {2009..2014}  ::: football finer_football  ::: syncRank SpringRank btl serialRank eigenvectorCentrality PageRank SVD_NRS