#!/usr/bin/env bash
1 17 18 ##
20 21 34 35 36 37
2 6 7 8  15 19 22 25 30 31
3 12 13 14 23 26 28 32
4
5
9 10 11 16  24 27 29 33
python server_lstm.py --port 7001 --stop_model True #two ONE LAST GG ONE NEW GOOD
python server_lstm.py --port 7002
python server_lstm.py --port 7003 --filemode 2 #good
python server_lstm.py --port 7004 --filemode 3
python server_lstm.py --port 7005 --filemode 4
python server_lstm.py --port 7006 --rewardchange 1.02
python server_lstm.py --port 7007 --rewardchange 10.
python server_lstm.py --port 7008 --rewardchange 0.1 #reacll up
python server_lstm.py --port 7009 --filemode 5
python server_lstm.py --port 7010 --filemode 5 --rewardchange 0.1
python server_lstm.py --port 7011 --filemode 5 --rewardchange 10.
python server_lstm.py --port 7012 --filemode 2 --rewardchange 0.1
python server_lstm.py --port 7013 --filemode 2 --rewardchange 10.
python server_lstm.py --port 7014 --filemode 2 #100W #good
python server_lstm.py --port 7015  #100W
python server_lstm.py --port 7016 --filemode 5 #100W
python server_lstm.py --port 7017 --stop_model True #100W #good
python server_lstm.py --port 7018 --stop_model True # test4 #good
python server_lstm.py --port 7019 --confidence_mode True  #add confidence and etc
#50W  stop_model filemode test4
#change all the reward as I use after "#" #before is something wrong with stop_model True
# all 50w
python server_lstm.py --port 7020 --stop_model True --confidence_mode True
python server_lstm.py --port 7021 --stop_model True --confidence_mode True #test4
python server_lstm.py --port 7022 --confidence_mode True
python server_lstm.py --port 7023 --filemode 2  --confidence_mode True
python server_lstm.py --port 7024 --filemode 5  --confidence_mode True
python server_lstm.py --port 7025
python server_lstm.py --port 7026 --filemode 2
python server_lstm.py --port 7027 --filemode 5
#50w with confidence reward back 50w
python server_lstm.py --port 7028 --confidence_mode True --filemode 2
python server_lstm.py --port 7029 --confidence_mode True --filemode 5 #mean down
python server_lstm.py --port 7030 --confidence_mode True  #good
python server_lstm.py --port 7031 --confidence_mode True  --rewardchange 0.1 #recall up
python server_lstm.py --port 7032 --confidence_mode True  --rewardchange 0.1  --filemode 2
python server_lstm.py --port 7033 --confidence_mode True  --rewardchange 0.1  --filemode 5 #mean down
python server_lstm.py --port 7034 --stop_model True --confidence_mode True  #good
python server_lstm.py --port 7035 --stop_model True --confidence_mode True #test4
python server_lstm.py --port 7036 --stop_model True --confidence_mode  True #action
python server_lstm.py --port 7037 --stop_model True --confidence_mode True #test4 #action  #good
#tune more
python server_lstm.py --port 7038 --confidence_mode True # --rewardchange  #### 0.01 0.1
#here
FOR TEST4 ACTION IS BETTER? MAYBE TUNE MORE FROM 21 22 USE REWARD 0.1
python server_lstm.py --port 7039 --stop_model True  #action
python server_lstm.py --port 7040 --stop_model True  #test4 #action
python server_lstm.py --port 7041 --stop_model True
python server_lstm.py --port 7042 --stop_model True  #test4
python server_lstm.py --stop_model True --confidence_mode True   --rewardchange 0.1 --port 7057
python server_lstm.py --stop_model True --confidence_mode True  --rewardchange 0.1 --port 7058  #test4 #action

FOR filemode 1 confidence is useless   most important!
python server_lstm.py --port 7043 --confidence_mode True  --rewardchange 0.01
python server_lstm.py --port 7044 --rewardchange 0.01
python server_lstm.py --port 7045 or 7046   only one 0.1
python server_lstm.py --port 7049 or 50 only one 10
python server_lstm.py --port 7051 -0.1 -10 order
python server_lstm.py --port 7052 -0.01 -100 order
python server_lstm.py --port 7053 -0.1 -100 order
python server_lstm.py --port 7054 -0.1 -30 order
python server_lstm.py --port 7055  overfit
python server_lstm.py --port 7056 -0.1 -1000  overfit

FORT filemode 2 confidence is worse same as filemode 1

FOR filemode 5
python server_lstm.py --port 7047  --filemode 5
python server_lstm.py --port 7048  --filemode 5 --rewardchange 0.1

#important
python server_lstm.py --port 7025

23 24 buchong


first 43 44 45 46 47 48 49 50 25  39 40 42 51 41 52 53 54 55 56 57 58 23  24 running

#8 30 31 43 44 45 46 49 50 25