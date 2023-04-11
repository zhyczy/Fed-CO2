cd /public/home/caizhy/work/Peer/NIID-Bench-main/
# salloc -N 1 -n 4 -p debug --gres=gpu:1 --exclude=ai_gpu[34],sist_gpu[45,65,61],sist-a40-07
python experiments.py --model=cnn-b \
  --dataset=cifar100 \
  --alg=pfedKL-abl \
  --lr=0.01 \
  --batch-size=64 \
  --epochs=5 \
  --n_parties=100 \
  --test_round=2\
  --comm_round=3\
  --eval_step=1 \
  --rho=0.9 \
  --version=1 \
  --kl_epochs=5 \
  --partition=noniid-labeldir100\
  --beta=0.3\
  --device='cuda:0'\
  --datadir='./data/' \
  --logdir='./logs_emb/' \
  --noise=0\
  --init_seed=0\
  --sample=0.05\
  --show_all_accuracy

  # --save_model

# python experiments.py --model=vit \
#   --dataset=cifar10 \
#   --alg=hyperVit \
#   --lr=0.01 \
#   --hyper_hid=150\
#   --client_embed_size=32\
#   --batch-size=64 \
#   --epochs=5 \
#   --n_parties=50 \
#   --rho=0.9 \
#   --eval_step=5 \
#   --version=7 \
#   --partition=noniid-labeluni\
#   --beta=0.3\
#   --device='cuda:0'\
#   --datadir='./data/' \
#   --logdir='./logs_emb/' \
#   --noise=0\
#   --init_seed=0\
#   --sample=0.1\
#   --save_model

# python experiments.py --model=cnn-b \
#   --dataset=cifar10 \
#   --alg=fedAP \
#   --lr=0.01 \
#   --batch-size=64 \
#   --epochs=1 \
#   --n_parties=50 \
#   --rho=0.9 \
#   --comm_round=200 \
#   --test_round=200 \
#   --eval_step=1 \
#   --partition=noniid-labeldir \
#   --device='cuda:0'\
#   --datadir='./data/' \
#   --logdir='./logs_emb/' \
#   --init_seed=0\
#   --sample=1
  
  # --dataset=cifar10 \
  # --alg=hyperVit_cluster \
  # --lr=0.01 \
  # --batch-size=64 \
  # --epochs=1 \
  # --n_parties=50 \
  # --rho=0.9 \
  # --comm_round=40 \
  # --test_round=40 \
  # --eval_step=1 \
  # --partition=noniid-labeluni \
  # --device='cuda:0'\
  # --datadir='./data/' \
  # --logdir='./logs_emb/' \
  # --init_seed=0\
  # --sample=1


# python experiments.py --model=cnn \
#   --dataset=cifar10 \
#   --alg=proto_cluster \
#   --lr=0.01 \
#   --batch-size=64 \
#   --epochs=1 \
#   --n_parties=10 \
#   --rho=0.9 \
#   --comm_round=20 \
#   --test_round=20 \
#   --eval_step=1 \
#   --partition=2-cluster \
#   --device='cuda:0'\
#   --datadir='./data/' \
#   --logdir='./logs_emb/' \
#   --init_seed=0\
#   --sample=1


  # --partition=homo \
  # --device='cuda:0'\
  # --datadir='./data/' \
  # --logdir='./logs_emb/' \
  # --init_seed=0\
  # --sample=0.5


  # --partition=2-cluster \
  # --device='cuda:0'\
  # --datadir='./data/' \
  # --logdir='./logs_emb/' \
  # --init_seed=0\
  # --sample=0.5
  # --save_model \