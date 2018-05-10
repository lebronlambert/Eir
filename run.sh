#!/usr/bin/env bash
python server_lstm.py --port 7001 --stop_model True #two ONE LAST GG ONE NEW GOOD
python server_lstm.py --port 7002
python server_lstm.py --port 7003 --filemode 2
python server_lstm.py --port 7004 --filemode 3
python server_lstm.py --port 7005 --filemode 4
python server_lstm.py --port 7006 --rewardchange 1.02
python server_lstm.py --port 7007 --rewardchange 10.
python server_lstm.py --port 7008 --rewardchange 0.1
python server_lstm.py --port 7009 --filemode 5
python server_lstm.py --port 7010 --filemode 5 --rewardchange 0.1
python server_lstm.py --port 7011 --filemode 5 --rewardchange 10.
python server_lstm.py --port 7012 --filemode 2 --rewardchange 0.1
python server_lstm.py --port 7013 --filemode 2 --rewardchange 10.
python server_lstm.py --port 7014 --filemode 2 #100W
python server_lstm.py --port 7015  #100W
python server_lstm.py --port 7016 --filemode 5 #100W
python server_lstm.py --port 7017 --stop_model True #100W
python server_lstm.py --port 7018 --stop_model True # test4
python server_lstm.py --port 7019 --confidence_mode True  #add confidence and etc
#50W  stop_model filemode test4
#change all the reward as I use after "#" #before is something wrong with stop_model True
python server_lstm.py --port 7020 --stop_model True --confidence_mode True
python server_lstm.py --port 7021 --stop_model True --confidence_mode True #test4
python server_lstm.py --port 7022 --confidence_mode True #--here
python server_lstm.py --port 7023 --filemode 2  --confidence_mode True
python server_lstm.py --port 7024 --filemode 5  --confidence_mode True