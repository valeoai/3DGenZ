import os
import numpy as np
import torch
from torch.nn.functional import softmax
from tqdm import tqdm

#Imports for Backbone
import genz3d.fkaconv.lightconvpoint.utils.transformations as lcp_transfo
from genz3d.fkaconv.lightconvpoint.networks.fkaconv import FKAConv as Network
from genz3d.fkaconv.examples.scannet.train import get_data

#Imports from 3DGenZ modules
from genz3d.seg.dataloaders import make_data_loader
from genz3d.seg.utils.loss import SegmentationLosses
from genz3d.seg.utils.lr_scheduler import LR_Scheduler
from genz3d.seg.utils.metrics import Evaluator
from genz3d.seg.class_names import CLASS_NAMES_SN
from genz3d.seg.trainer_class import Trainer_default, main


class Trainer(Trainer_default):

    def __init__(self, args):
        super().__init__(args)
        # Define Dataloader
        self.N_CLASSES = 21
        training_transformations_data = [lcp_transfo.RandomSubSample(args.config["dataset_num_points"])]
        validation_transformations_data = [lcp_transfo.RandomSubSample(args.config["dataset_num_points"])]
        training_transformations_points = [lcp_transfo.RandomRotation(rotation_axis=2),]
        self.class_names = CLASS_NAMES_SN

        def network_function():
            return Network(3, self.N_CLASSES, segmentation=True)

        (self.train_loader, self.val_loader, _, self.nclass,) = make_data_loader(
            args, data_dir=args.rootdir, network_function=network_function,
            training_transformations_points=training_transformations_points,
            training_transformations_data=training_transformations_data,
            validation_transformations_data=validation_transformations_data)
        device = torch.device("cuda")
        net = network_function()
        net.to(device)
        model = net
        #Load the ConvBasic version
        if args.test:
            chkp_path = os.path.join(args.savedir)
            print("Load the complete trained model from {}".format(chkp_path))
            checkpoint = torch.load(chkp_path)
            model.load_state_dict(checkpoint['state_dict'], strict=False)
        else:
            print("Load the pre-trained network from {}".\
                format(os.path.join(args.config_savedir, "checkpoint.pth")))
            model.load_state_dict(torch.load(os.path.join(args.config_savedir,\
                 "checkpoint.pth"))['state_dict'], strict=False)
        #Freeze all layers besides the linear layer that gets finetuned
        model.freeze_backbone_fcout()
        model.freeze_bn()

        train_params = [{"params": model.get_1x_lr_params(), "lr": args.lr},\
            {"params": model.get_10x_lr_params(), "lr": args.lr * 10}]

        # Define Optimizer adn generator optimizer
        optimizer = torch.optim.Adam(train_params, lr=args.config["lr_start"])

        # Define Generator
        embed_feature_size = 0
        self.set_generator(embed_feature_size=embed_feature_size)

        class_weight = torch.ones(self.nclass)
        class_weight[args.unseen_classes_idx_metric] = args.unseen_weight

        if args.cuda:
            class_weight = class_weight.cuda()
        self.criterion = SegmentationLosses(weight=class_weight, cuda=args.cuda).build_loss(mode=args.loss_type)
        self.model, self.optimizer = model, optimizer

        # # Define Evaluator
        self.evaluator = Evaluator(self.nclass, args.seen_classes_idx_metric, args.unseen_classes_idx_metric)

        # # Define lr scheduler
        self.scheduler = LR_Scheduler(args.lr_scheduler, args.lr, args.epochs, len(self.train_loader))

        # Using cuda
        if args.cuda:
            self.model = torch.nn.DataParallel(self.model, device_ids=self.args.gpu_ids)
            self.model = self.model.cuda()
            self.generator = self.generator.cuda()

        # # Resuming checkpoint
        self.best_pred = 0.0
        if args.resume is not None:
            if not os.path.isfile(args.resume):
                raise RuntimeError("=> no checkpoint found at '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']

            if args.cuda:
                self.model.load_state_dict(checkpoint["state_dict"])

        # # Clear start epoch if fine-tuning
        if args.ft:
            args.start_epoch = 0



    def training(self, epoch, args):
        train_loss = 0.0
        self.model.module.train()
        self.model.module.freeze_backbone_fcout()
        self.model.module.freeze_bn()
        tbar = tqdm(self.train_loader)
        num_img_tr = len(self.train_loader)

        i = None
        for i, sample in enumerate(tbar):

            features, pts, target, net_ids, net_pts, idx_batch, embedding =\
                get_data(sample, attributes=True, device="cuda")

            target = target[:, :, None]

            if self.args.cuda:
                pts, features, target, embedding = (
                    pts.cuda(),
                    features.cuda(),
                    target.cuda(),
                    embedding.cuda()
                )
            self.scheduler(self.optimizer, i, epoch, self.best_pred)

            with torch.no_grad():
                real_features = self.model.module.backbone(features, pts, support_points=net_pts, indices=net_ids)
            real_features = real_features.permute(0, 2, 1)

            # ===================Generator training=====================
            fake_features = torch.zeros(real_features.shape)
            if args.cuda:
                fake_features = fake_features.cuda()
            generator_loss_batch = 0.0

            for (count_sample_i, (real_features_i, target_i, embedding_i)) in enumerate(zip(real_features, target, embedding)):
                generator_loss_sample = 0.0

                target_i = target_i.view(-1)
                fake_features_i = torch.zeros(real_features_i.shape)
                if args.cuda:
                    fake_features_i = fake_features_i.cuda()

                unique_class = torch.unique(target_i)

                # If PC has  a point belonging to unseen class pixel, no training for generator and generated features for the whole PC
                has_unseen_class = False
                for u_class in unique_class:
                    if u_class in args.unseen_classes_idx_metric:
                        has_unseen_class = True

                for idx_in in unique_class:
                    if idx_in != 255:
                        self.optimizer_generator.zero_grad()
                        idx_class = target_i == idx_in
                        real_features_class = real_features_i[idx_class]
                        embedding_class = embedding_i[idx_class]

                        #Noise generation
                        z = torch.rand((embedding_class.shape[0], args.noise_dim))
                        if args.cuda:
                            z = z.cuda()

                        # Generation of the features
                        if args.generator_model == "gmmn":
                            fake_features_class = self.generator(embedding_class, z.float())
                            fake_feature_train = fake_features_class
                        else:
                            print("Incorrect generator model selection")
                            raise NotImplementedError

                        if (idx_in in args.seen_classes_idx_metric and not has_unseen_class):
                            # Avoid CUDA out of memory
                            random_idx = torch.randint(low=0, high=fake_feature_train.shape[0], size=(args.batch_size_generator,))

                            #Generator loss
                            g_loss = self.criterion_generator(
                                fake_feature_train[random_idx],
                                real_features_class[random_idx],
                            )
                            generator_loss_sample += g_loss.item()
                            g_loss.backward()
                            self.optimizer_generator.step()
                        fake_features_i[idx_class] = fake_feature_train.clone()
                generator_loss_batch += generator_loss_sample / len(unique_class)

                if args.real_seen_features and not has_unseen_class:
                    fake_features[count_sample_i] = real_features_i
                else:
                    fake_features[count_sample_i] = fake_features_i

            target = torch.squeeze(target)
            # ===================classification=====================
            self.optimizer.zero_grad()
            fake_features = fake_features.permute(0, 2, 1).contiguous()
            output = self.model.module.training_generative(fake_features.detach())
            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()
            train_loss += loss.item()
            # ===================log=====================
            tbar.set_description(
                " G loss: {:.3f}".format(generator_loss_batch)
                + " C loss: {:.3f}".format(train_loss / (i + 1))
            )
            self.writer.add_scalar(
                "train/total_loss_iter", loss.item(), i + num_img_tr * epoch
            )

        self.writer.add_scalar("train/total_loss_epoch", train_loss, epoch)
        print("[Epoch: %d, numPCs: %5d]"% (epoch, i * self.args.batch_size + pts.data.shape[0]))
        print("Loss: {:.3f}".format(train_loss))


    def validation(self, epoch, args):
        print("IMPORTANT: This validation function is only for MONITORING (see Scannet folder for validation function)")
        print("Fast validation (only for monitoring during training):")
        self.model.eval()
        self.evaluator.reset()
        tbar = tqdm(self.val_loader, desc="\r")
        test_loss = 0.0

        i = None
        for i, sample in enumerate(tbar):
            features, pts, target, net_ids, net_pts, _, _ = get_data(sample, attributes=True, device="cuda")

            with torch.no_grad():
                output = self.model.module.forward(x=features, pos=pts, support_points=net_pts,\
                    indices=net_ids, gen_forward=True, backbone=False)
                output = output.permute(0, 2, 1).contiguous()
                loss = self.criterion(output.view(-1, self.N_CLASSES), target.view(-1))
                pred = softmax(output, dim=2).data.cpu().numpy()
                bias_matrix = np.zeros(pred.shape)
                bias_matrix[:, :, args.seen_classes_idx_metric] = 1 * args.bias
                pred = np.argmax(pred - bias_matrix, axis=2)

            test_loss += loss.item()
            tbar.set_description("Test loss: %.3f" % (test_loss / (i + 1)))
            target = target.cpu().numpy()
            # Add batch sample into evaluator
            self.evaluator.add_batch(target, pred)

        # Fast test during the training
        self.validation_metric_logs(test_loss=test_loss, epoch=epoch, num=i * self.args.batch_size + pts.data.shape[0])



if __name__ == "__main__":
    main(("sn"))
