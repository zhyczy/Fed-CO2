python main.py  --mode AlignFed --log --version 1


# salloc -N 1 -n 4 -p debug --gres=gpu:1 --exclude=sist_gpu[36],ai_gpu[14,18,22],sist_gpu[45,65,61],sist-a40-07