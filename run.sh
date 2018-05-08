#!/usr/bin/env bash
python server_lstm.py --port 7002
python server_lstm.py --port 7003 --filemode 2
python server_lstm.py --port 7004 --filemode 3
python server_lstm.py --port 7005 --filemode 4
python server_lstm.py --port 7006 --rewardchange 1.02
python server_lstm.py --port 7007 --rewardchange 10.