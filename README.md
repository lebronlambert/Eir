# Eir

## Introduction
Example codes for the paper:

H. Wang, X. Liu, Y. Tao, W. Ye, Q. Jin, W. W. Cohen and E. P. Xing, [Automatic Human-like Mining and Constructing Reliable Genetic Association Database with Deep Reinforcement Learning](https://psb.stanford.edu/psb-online/proceedings/psb19/wang2.pdf), Proceedings of 24th Pacific Symposium on Biocomputing (PSB 2019).

## Requirements
you need to install pytorch and torch.
```
sh install_torch.sh or sh ./code/dqn/install_dependencies.sh
pip install -r requirements.txt
pip install torch
```
## An Example Command
```
python server_lstm.py
```
this command will run the program to use the lstm part.
```
cd code/dqn
./run_cpu.sh 7001 tmp/
```
this command will run the program to use the dqn part.
