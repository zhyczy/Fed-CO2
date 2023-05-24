# python main.py  --mode fed-co2 --log  --dataset  digits  --save_path checkpoint/digits --imbalance_train --beta 0.3 --min_img_num 2 --divide 3
# python main.py  --mode fed-co2 --log  --dataset  digits  --save_path checkpoint/digits 
# python main.py  --mode fed-co2 --log  --dataset  office --save_path checkpoint/office --imbalance_train --beta 0.3 --min_img_num 2 --divide 2
# python main.py  --mode copa --log   --dataset  digits  --save_path checkpoint/digits
# python main.py  --mode moon --log   

# python main.py  --mode copa --log
# python main.py  --mode copa --log  --dataset  office  --save_path checkpoint/office
# python main.py  --mode copa --log  --dataset  digits  --save_path checkpoint/digits

# python main.py  --mode fed-co2 --log --save_model --iters 2
# python main.py  --mode fed-co2 --log  --dataset  office  --save_path checkpoint/office --iters 2
# python main.py  --mode fed-co2 --log  --dataset  office  --save_path checkpoint/office
python main.py  --mode fed-co2 --log --save_model --iters 2 --test