cd ../src

../../parallel -j15 --resume-failed --results ../Output/footballbaselines --joblog ../joblog/footballbaselines python ./train.py --num_trials 10  --season {1} --dataset {2} --no-cuda  --all_methods baselines_shorter -All ::: 2009 2010 2011 2012 2013 2014 ::: football finer_football

../../parallel -j15 --resume-failed --results ../Output/realAbaselines --joblog ../joblog/realAbaselines python ./train.py --num_trials 10 --dataset {1} --all_methods baselines_shorter --no-cuda -All ::: faculty_business faculty_cs faculty_history animal

../../parallel -j15 --resume-failed --results ../Output/realBbaselines --joblog ../joblog/realBbaselines python ./train.py --num_trials 10 --dataset {1} --all_methods baselines_shorter --no-cuda -All ::: HeadToHead finance

../../parallel -j15 --resume-failed --results ../Output/basketballbaselines --joblog ../joblog/basketballbaselines python ./train.py --num_trials 10  --no-cuda --season {1} --dataset {2} --all_methods baselines_shorter -All ::: {1985..2014} ::: basketball finer_basketball

../../parallel -j15 --resume-failed --results ../Output/ERObaselines --joblog ../joblog/ERObaselines python ./train.py --p {1} --eta {2} --no-cuda --ERO_style {3} --dataset ERO --all_methods baselines_shorter -All --ambient 0 ::: 0.05 1 ::: 0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9  ::: uniform gamma

../../parallel -j15 --resume-failed --results ../Output/realAmvr --joblog ../joblog/realAmvr python ./train.py --num_trials 10 --dataset {1} --all_methods mvr --no-cuda -All ::: faculty_business faculty_cs faculty_history animal

../../parallel -j15 --resume-failed --results ../Output/footballmvr --joblog ../joblog/footballmvr python ./train.py --num_trials 10  --season {1} --dataset {2} --no-cuda  --all_methods mvr -All ::: 2009 2010 2011 2012 2013 2014 ::: football

../../parallel -j15 --resume-failed --results ../Output/finerfootballmvr --joblog ../joblog/finerfootballmvr python ./train.py --num_trials 10  --season {1} --dataset {2} --no-cuda  --all_methods mvr -All ::: 2009 2010 2011 2012 2013 2014 ::: finer_football

../../parallel -j15 --resume-failed --results ../Output/realBmvr --joblog ../joblog/realBmvr python ./train.py --num_trials 10 --dataset {1} --all_methods mvr --no-cuda -All ::: HeadToHead finance

../../parallel -j15 --resume-failed --results ../Output/basketballmvr --joblog ../joblog/basketballmvr python ./train.py --num_trials 10  --no-cuda --season {1} --dataset {2} --all_methods mvr -All ::: {1985..2014} ::: basketball finer_basketball

../../parallel -j15 --resume-failed --results ../Output/EROmvr --joblog ../joblog/EROmvr python ./train.py --p {1} --eta {2} --no-cuda --ERO_style {3} --dataset ERO --all_methods mvr -All --ambient 0 ::: 0.05 1 ::: 0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9  ::: uniform gamma

