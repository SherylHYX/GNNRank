cd ../src

python ./train.py -D --p 1 --ERO_style gamma --N 400 --all_methods SpringRank -All

python ./train.py -D --dataset football --season 2011 --all_methods serialRank

python ./train.py -D --dataset basketball_finer --season 2000 --all_methods syncRank

python ./train.py -D --dataset animal --all_methods davidScore ib --trainable_alpha --train_with proximal_baseline --baseline syncRank

python ./train.py -D --dataset faculty_business --all_methods DIGRAC --train_with proximal_innerproduct --pretrain_with innerproduct

python ./train.py -D --dataset faculty_cs --all_methods eigenvectorCentrality

python ./train.py -D --dataset faculty_history --all_methods PageRank

python ./train.py -D --dataset HeadToHead --all_methods rankCentrality

python ./train.py -D --dataset finance --all_methods btl

python ./train.py -D --dataset finer_football --all_methods mvr