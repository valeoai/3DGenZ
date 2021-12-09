cd ../../../..
pip install -e 3DGenZ/
cd 3DGenZ/genz3d/seg/
python3 train_point_sk.py --iter=1000 --epochs=20  --dataset="sk" --savedir="../kpconv/results/Log_2020-09-18_19-00-03/checkpoints"  --unseen_weight=50 --checkname="sk_gmmn_weighted50_zsl_retrained2" --load_embedding="glove_w2v" --w2c_size=600 --embed_dim=600
