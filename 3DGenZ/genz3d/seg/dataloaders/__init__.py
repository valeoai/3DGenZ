import os
import sys
import torch
from torch.utils.data import DataLoader
import numpy as np
from pathlib import Path


def make_data_loader(
        args,
        flist_train=None,
        flist_val=None,
        data_dir=None,
        network_function=None,
        training_transformations_data=None,
        validation_transformations_data=None,
        training_transformations_points=None,
        config=None,
    ):
    home = Path.home()
    print("Args dataset {}".format(args.dataset))


    if args.dataset == "s3dis":
        from genz3d.convpoint.examples.s3dis.s3dis_seg import PartDatasetTrainVal
        #args.iter = 3 #Number of iterations
        train_set = PartDatasetTrainVal(flist_train, args.rootdir, training=True,
                                        block_size=args.blocksize, npoints=args.npoints,
                                        iteration_number=args.batchsize*args.iter,
                                        nocolor=args.nocolor, jitter=args.jitter,
                                        use_unseen_seen=True, attribute=args.load_embedding)
        train_loader = DataLoader(train_set, batch_size=args.batchsize, shuffle=True,\
                                  num_workers=args.threads)
        val_set = PartDatasetTrainVal(flist_val, args.rootdir,\
                                      training=False, block_size=args.blocksize,
                                      npoints=args.npoints, nocolor=args.nocolor,
                                      attribute=args.load_embedding)
        val_loader = DataLoader(val_set, batch_size=args.batchsize, shuffle=False,\
                                num_workers=args.threads)

        num_class = 13

        return train_loader, val_loader, None, num_class

    elif args.dataset == "sk":
        # Initialize datasets
        from genz3d.kpconv.datasets.SemanticKitti import SemanticKittiDataset, SemanticKittiSampler, SemanticKittiCollate
        training_dataset = SemanticKittiDataset(config, args, set='training',\
                                                balance_classes=True, ignored_labels=np.sort([0]), visual_set=not config.cluster_use)
        test_dataset = SemanticKittiDataset(config, args, set='validation',
                                            balance_classes=False, ignored_labels=np.sort([0]), visual_set=not config.cluster_use)

        # Initialize samplers
        #Change number of iterations if passed as an parameter
        if not(args.iter is None): 
            config.epoch_steps = args.iter
        
        training_sampler = SemanticKittiSampler(training_dataset)
        test_sampler = SemanticKittiSampler(test_dataset)

        # Initialize the dataloader
        training_loader = DataLoader(training_dataset,
                                     batch_size=1,
                                     sampler=training_sampler,
                                     collate_fn=SemanticKittiCollate,
                                     num_workers=config.input_threads,
                                     pin_memory=True)
        test_loader = DataLoader(test_dataset,
                                 batch_size=1,
                                 sampler=test_sampler,
                                 collate_fn=SemanticKittiCollate,
                                 num_workers=config.input_threads,
                                 pin_memory=True)

        # Calibrate max_in_point value
        training_sampler.calib_max_in(config, training_loader, verbose=True)
        test_sampler.calib_max_in(config, test_loader, verbose=True)

        # Calibrate samplers
        training_sampler.calibration(training_loader, verbose=True)
        test_sampler.calibration(test_loader, verbose=True)
        return training_loader, test_loader, training_dataset, test_dataset
    elif args.dataset == "scannet":
        from genz3d.fkaconv.examples.scannet.train import Dataset #import the dataloader for Scannet
        #Number of iterations
        #args.config["train_iter_per_epoch"] = 3
        train_dataset = Dataset(
            data_dir, split='train',
            iter_nbr=args.config["train_iter_per_epoch"]*args.config["batch_size"],
            network_function=network_function,
            transformations_data=training_transformations_data,
            transformations_points=training_transformations_points,
            pillar_size=args.config["dataset_pillar_size"],
            ignore_value=args.config["ignore_test_value"], no_attribute=False, attribute=args.load_embedding
        )

        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=args.config["batch_size"], shuffle=True,
            num_workers=args.config['threads']
        )

        validation_dataset = Dataset(
            data_dir, split='test',
            iter_nbr=args.config["val_iter_per_epoch"]*args.config["batch_size"],
            network_function=network_function,
            transformations_data=validation_transformations_data,
            pillar_size=args.config["dataset_pillar_size"],
            ignore_value=np.array(args.config["ignore_test_value"]),
            no_attribute=False, attribute=args.load_embedding
        )

        val_loader = torch.utils.data.DataLoader(
            validation_dataset,
            batch_size=args.config["batch_size"], shuffle=False,
            num_workers=args.config['threads']
        )
        num_class = 21
        return train_loader, val_loader, None, num_class
    else:
        print("Not founded dataset {}".format(args.dataset))
        raise NotImplementedError
