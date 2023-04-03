python main.py  --mode peer --log --version 90
# python main.py  --mode peer --log --version 18  --dataset  office_home  --save_path ../checkpoint/office_home
# python experiment.py  --mode peer --log --version 18 --iters 2
# python experiment.py  --mode peer --log --version 18  --dataset  office_home  --save_path ../checkpoint/office_home --iters 2

# salloc -N 1 -n 4 -p debug --gres=gpu:1 --exclude=ai_gpu[34],sist_gpu[45,65,61],sist-a40-07
