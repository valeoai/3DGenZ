cd ../
python3 train.py --test=True --config_name="FKAConv_scannet_ZSL4/config.yaml" --ckpt_path="../../../seg/run/scannet/sn_gmmn_weighted50_zsl_retrained/experiment_0/20_model.pth.tar"
python3 eval.py --predfolder="eval_results"
