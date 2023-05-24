cifar10:
-lbs 16 
-num_clients 20 -sample_rate 1 -m cnn -global_rounds 2000  -k=5
-lr 0.01 -p_learning_rate 0.01 -beta 1 -lambda 15 -local_step=1

-lbs 16 -num_clients 20 -sample_rate 1 -m resnet -global_rounds 2000 -k=5
-lr 0.01 -p_learning_rate 0.01 -beta 1 -lambda 15 -local_step=1

cifar100:
-lbs 4 -num_clients 20 -sample_rate 1 -m cnn -global_rounds 2000 -k=5 
-lr 0.01 -p_learning_rate 0.01 -beta 1 -lambda 15 -local_step=1

-lbs 4 -num_clients 20 -sample_rate 1 -m resnet -global_rounds 2000 -k=5
-lr 0.01 -p_learning_rate 0.01 -beta 1 -lambda 15 -local_step=1