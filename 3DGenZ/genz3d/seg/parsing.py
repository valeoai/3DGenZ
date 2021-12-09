import os
import argparse
import yaml
import numpy as np

def get_parser():
    parser = argparse.ArgumentParser(description="Generator Training")
    parser.add_argument("--workers", type=int, default=6, metavar="N", help="dataloader threads")
    parser.add_argument("--freeze-bn", type=bool, default=False, help="whether to freeze bn parameters (default: False)",)
    parser.add_argument(
        "--loss-type",
        type=str,
        default="ce",
        choices=["ce", "focal"],
        help="loss func type (default: ce)",
    )
    parser.add_argument("--exp_path", type=str, default="run", help="set the checkpoint name")
    parser.add_argument(
        "--seed", type=int, default=1, metavar="S", help="random seed (default: 1)"
    )
    parser.add_argument(
        "--no-cuda", action="store_true", default=False, help="disables CUDA training"
    )
    parser.add_argument(
        "--gpu-ids",
        type=str,
        default="0",
        help="use which gpu to train, must be a \
            comma-separated list of integers only (default=0)",
    )
    parser.add_argument(
        "--start_epoch",
        type=int,
        default=0,
        metavar="N",
        help="start epochs (default:0)",
    )

    # finetuning pre-trained models
    parser.add_argument(
        "--ft",
        action="store_true",
        default=False,
        help="finetuning on a different dataset",
    )

    parser.add_argument(
        "--use-balanced-weights",
        action="store_true",
        default=False,
        help="whether to use balanced weights (default: False)",
    )
    # optimizer params
    parser.add_argument(
        "--lr",
        type=float,
        default=0.007,
        metavar="LR",
        help="learning rate (default: auto)",
    )

    parser.add_argument(
        "--lr-scheduler",
        type=str,
        default="poly",
        choices=["poly", "step", "cos"],
        help="lr scheduler mode: (default: poly)",
    )
    parser.add_argument(
        "--momentum",
        type=float,
        default=0.9,
        metavar="M",
        help="momentum (default: 0.9)",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=5e-4,
        metavar="M",
        help="w-decay (default: 5e-4)",
    )
    parser.add_argument(
        "--nesterov",
        action="store_true",
        default=False,
        help="whether use nesterov (default: False)",
    )

    parser.add_argument(
        "--load_embedding",
        type=str,
        default="glove_w2v",
        choices=["glove_w2v", None],
    )
    parser.add_argument("--w2c_size", type=int, default=300)
    parser.add_argument("--bias", type=float, default=None)
    parser.add_argument("--unseen_weight", type=int, default=50, help="Weight for the unseen classes")

    ### GENERATOR ARGS
    parser.add_argument("--noise_dim", type=int, default=300)
    parser.add_argument("--embed_dim", type=int, default=300)
    parser.add_argument("--hidden_size", type=int, default=256)
    parser.add_argument("--lr_generator", type=float, default=0.0002)
    parser.add_argument("--batch_size_generator", type=int, default=128)
    parser.add_argument("--saved_validation_images", type=int, default=10)
    parser.add_argument("--generator_model", type=str, default="gmmn", choices=["gmmn", "dae"])

    return parser

def complete_parser_s3dis():

    #Parser for s3dis

    parser = get_parser()
    parser.add_argument("--dataset", type=str, default="s3dis", help="dataset name (default: s3dis)")

    # training hyper params
    parser.add_argument("--epochs", type=int, default=20, metavar="N", help="number of epochs to train (default: auto)")
    # checking point
    parser.add_argument("--resume", type=str, default=None, help="put the path to resuming file if needed")
    parser.add_argument("--checkname", type=str, default="gmmn_context_w2c300_linear_weighted100_hs256_2_unseen",)

    # false if embedding resume
    parser.add_argument("--global_avg_pool_bn", type=bool, default=True)

    # evaluation option
    parser.add_argument("--eval_interval", type=int, default=20, help="evaluation interval (default: 1)")
    # keep empty
    parser.add_argument("--unseen_classes_idx", type=int, default=[])

    unseen_classes_idx_metric = []
    print("unseen classes idx metric {}".format(unseen_classes_idx_metric))
    unseen_classes_idx_metric = [4, 5, 7, 10]


    seen_classes_idx_metric = np.arange(13)

    seen_classes_idx_metric = np.delete(seen_classes_idx_metric, unseen_classes_idx_metric).tolist()
    parser.add_argument("--seen_classes_idx_metric", type=int, default=seen_classes_idx_metric)
    parser.add_argument("--unseen_classes_idx_metric", type=int, default=unseen_classes_idx_metric)
    parser.add_argument("--real_seen_features", type=bool, default=True, help="real features for seen classes",)
    #Specific size of scannet
    parser.add_argument("--feature_dim", type=int, default=128)


    #Arguments for ConvPoint
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--ply", action="store_true", help="save ply files (test mode)")
    parser.add_argument("--savedir", default="results/", type=str)
    parser.add_argument("--rootdir", type=str, required=True)
    parser.add_argument("--batchsize", "-b", default=4, type=int) #Originally 16
    parser.add_argument("--npoints", default=8192, type=int)
    parser.add_argument("--area", default=1, type=int)
    parser.add_argument("--blocksize", default=2, type=int)
    parser.add_argument("--iter", default=1000, type=int)
    parser.add_argument("--threads", default=4, type=int)
    parser.add_argument("--npick", default=16, type=int)
    parser.add_argument("--savepts", action="store_true")
    parser.add_argument("--nocolor", action="store_true")
    parser.add_argument("--test_step", default=0.2, type=float)
    parser.add_argument("--jitter", default=0.4, type=float)
    parser.add_argument("--model", default="SegBig", type=str)
    parser.add_argument("--drop", default=0, type=float)
    parser.add_argument("--cluster", default=False, type=bool)
    parser.add_argument("--batch-size", type=int, default=8, help="input batch size for training (default: auto)")

    args = parser.parse_args()
    if args.checkname is None:
        args.checkname = "s3dis_zsl_default"
    return args

def complete_parser_sn():
    parser = get_parser()
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--config_savedir", type=str, default="../fkaconv/examples/scannet/FKAConv_scannet_test_ZSL")
    parser.add_argument("--resume", type=str, default=None, help="put the path to resuming file if needed")
    parser.add_argument("--checkname", type=str, default="sn_default")
    # evaluation option
    parser.add_argument("--eval-interval", type=int, default=20, help="evaluation interval (default: 20)")
    parser.add_argument("--real_seen_features", type=bool, default=True, help="real features for seen classes")
    #Specific size of scannet
    parser.add_argument("--feature_dim", type=int, default=64)

    #Arguments for ConvPoint
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--ply", action="store_true", help="save ply files (test mode)")
    parser.add_argument("--savedir", default="results/", type=str)
    parser.add_argument("--rootdir", type=str, required=True)
    parser.add_argument("--iter", default=None, type=int)
    parser.add_argument("--batch_size", default=4, type=int)

    args = parser.parse_args()

    config = yaml.load(open("{}".format(os.path.join(args.config_savedir, "config.yaml")), "r"), Loader=yaml.FullLoader)

    unseen_classes_idx_metric = config["ignore_value"]
    seen_classes_idx_metric = np.arange(0, 21)
    seen_classes_idx_metric = np.delete(seen_classes_idx_metric, unseen_classes_idx_metric).tolist()
    args.unseen_classes_idx_metric = unseen_classes_idx_metric
    args.seen_classes_idx_metric = seen_classes_idx_metric
    args.config = config
    #Read the configs, and take default values from the config. Parameters passed directly are presevered and overwrite config
    if args.epochs is None:
        args.epochs = args.config["epoch_nbr_generator"]

    if args.bias is None:
        args.bias = args.config["bias"]
    else:
        args.config["bias"] = args.bias
    if not args.iter is None:
        args.config["train_iter_per_epoch"] = args.iter
    args.dataset = args.config["dataset_name"]
    if args.checkname is None:
        args.checkname = "sn_zsl_default"
    return args

def complete_parser_sk():
    parser = get_parser()
    parser.add_argument("--dataset", type=str, default="sk", help="dataset name (default: s3dis)")

    # checking point
    parser.add_argument("--resume", type=str, default=None, help="put the path to resuming file if needed")
    parser.add_argument("--checkname", type=str, default="gmmn_context_w2c300_linear_weighted100_hs256_2_unseen")
    # evaluation option
    parser.add_argument("--eval-interval", type=int, default=20, help="evaluation interval (default: 1)")

    # keep empty
    parser.add_argument("--unseen_classes_idx", type=int, default=[])
    unseen_classes_idx_metric = []
    #for name in unseen_names:
    print("unseen classes idx metric {}".format(unseen_classes_idx_metric))
    unseen_classes_idx_metric = [0, 3, 4, 7, 19] #As class 0 is remove (unlabeled), everything is decreased by 1

    ### FOR METRIC COMPUTATION IN ORDER TO GET PERFORMANCES FOR TWO SETS
    seen_classes_idx_metric = np.arange(0, 20)
    seen_classes_idx_metric = np.delete(seen_classes_idx_metric, unseen_classes_idx_metric).tolist()
    print("Seen class idx metric {}".format(seen_classes_idx_metric))
    parser.add_argument("--seen_classes_idx_metric", type=int, default=seen_classes_idx_metric)
    parser.add_argument("--unseen_classes_idx_metric", type=int, default=unseen_classes_idx_metric)
    parser.add_argument("--real_seen_features", type=bool, default=True, help="real features for seen classes")


    #Specific size of features
    parser.add_argument("--feature_dim", type=int, default=128)
    #Arguments for result saving
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--ply", action="store_true", help="save ply files (test mode)")
    parser.add_argument("--savedir", default="results/", type=str)
    parser.add_argument("--iter", default=None, type=int)
    parser.add_argument("--epochs", type=int, default=None, help="number of epochs to train")
    parser.add_argument("--batch-size", type=int, default=8, help="input batch size for training (default: auto)")
    
    args = parser.parse_args()

    if args.bias is None:
        args.bias = 0.0 #Default Bias
    if args.checkname is None:
        args.checkname = "sk_zsl_default"
    return args
