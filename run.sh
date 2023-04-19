# python main.py  --mode peer --log --version 91
# python main.py  --mode peer --log --version 18  --dataset  office_home  --save_path ../checkpoint/office_home
# python main.py  --mode COPA --log --version 1  --dataset  digits  --save_path ../checkpoint/digits
# python main.py  --mode moon --log --version 1  

python main.py  --mode COPA --log --version 2
# python main.py  --mode COPA --log --version 2  --dataset  office_home  --save_path ../checkpoint/office_home
# python main.py  --mode COPA --log --version 2  --dataset  digits  --save_path ../checkpoint/digits

# python experiment.py  --mode peer --log --version 18 --iters 2
# python experiment.py  --mode peer --log --version 18  --dataset  office_home  --save_path ../checkpoint/office_home --iters 2

# salloc -N 1 -n 4 -p debug --gres=gpu:1 --exclude=sist_gpu[62],ai_gpu[34],sist_gpu[45,65,61],sist-a40-07


# python converge.py  --mode peer --log --version 18 --iters 1
# python converge.py  --mode fedavg --log --version 18 --iters 1