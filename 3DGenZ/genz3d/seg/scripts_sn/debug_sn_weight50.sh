cd ../../../..
sudo pip3 install -e 3DGenZ/
cd 3DGenZ/genz3d/seg/
python3 train_point_sn.py  --iter=10 --epochs=5 --eval-interval=5 --config_savedir="../fkaconv/examples/scannet/FKAConv_scannet_ZSL4" --rootdir="../../../../data/scannet/"  --unseen_weight=50 --checkname="sn_gmmn_weighted50_zsl_debug" --load_embedding="glove_w2v" --w2c_size=600 --embed_dim=600
