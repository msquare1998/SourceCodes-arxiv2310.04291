rm out.log
export CUDA_VISIBLE_DEVICES=1
#python3 main.py
nohup python3 main.py > out.log 2>&1 &
