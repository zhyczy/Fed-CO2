cd /public/home/caizhy/work/Peer/NIID-Bench-main/
python main.py --model=cnn \
	--dataset=cifar100 \
	--n_parties=100 \
	--beta=0.3\
	--alg=peer \
	--batch-size=64 \
	--partition=iid-label100\
	--device='cuda:0'\
	--noise=0\
	--init_seed=0