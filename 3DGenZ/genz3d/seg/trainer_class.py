import torch
#Imports from 3DGenZ modules
from genz3d.seg.utils.saver import Saver
from genz3d.seg.utils.summaries import TensorboardSummary
from genz3d.seg.modeling.gmmn import GMMNnetwork
from genz3d.seg.utils.loss import GMMNLoss
from parsing import complete_parser_sk, complete_parser_sn, complete_parser_s3dis

class Trainer_default():

    def __init__(self, args):
        self.args = args
        self.saver = Saver(args)
        self.saver.save_experiment_config()
        # Define Tensorboard Summary
        self.summary = TensorboardSummary(self.saver.experiment_dir)
        self.writer = self.summary.create_summary()
        self.class_names = None
        #Define attributes (default: set to None)
        self.generator = None
        self.criterion_generator = None
        self.optimizer_generator = None
        self.best_pred = None


    def attribute_size_get(self, load_embedding):
        #Returns based on the name of the embedding the embedding size
        if load_embedding == "glove_w2v":
            attribute_size = 600
        else:
            raise NotImplementedError
        return attribute_size


    def set_generator(self, embed_feature_size):
        if self.args.generator_model == "gmmn":
            self.generator = GMMNnetwork(self.args.noise_dim, self.args.embed_dim,\
                self.args.hidden_size, self.args.feature_dim, embed_feature_size=embed_feature_size)
            self.criterion_generator = GMMNLoss(sigma=[2, 5, 10, 20, 40, 80], cuda=self.args.cuda).build_loss()
        else:
            raise NotImplementedError

        self.optimizer_generator = torch.optim.Adam(self.generator.parameters(), lr=self.args.lr_generator)


    def validation_metric_logs(self, test_loss, epoch, num):
        self.evaluator.print_confusion_matrix()
        Acc, Acc_seen, Acc_unseen = self.evaluator.Pixel_Accuracy()
        (Acc_class, Acc_class_by_class, Acc_class_seen, Acc_class_unseen) = self.evaluator.Pixel_Accuracy_Class()
        (mIoU, mIoU_by_class, mIoU_seen, mIoU_unseen) = self.evaluator.Mean_Intersection_over_Union()
        (FWIoU, FWIoU_seen, FWIoU_unseen) = self.evaluator.Frequency_Weighted_Intersection_over_Union()
        self.writer.add_scalar("val_overall/total_loss_epoch", test_loss, epoch)
        self.writer.add_scalar("val_overall/mIoU", mIoU, epoch)
        self.writer.add_scalar("val_overall/Acc", Acc, epoch)
        self.writer.add_scalar("val_overall/Acc_class", Acc_class, epoch)
        self.writer.add_scalar("val_overall/fwIoU", FWIoU, epoch)
        self.writer.add_scalar("val_seen/mIoU", mIoU_seen, epoch)
        self.writer.add_scalar("val_seen/Acc", Acc_seen, epoch)
        self.writer.add_scalar("val_seen/Acc_class", Acc_class_seen, epoch)
        self.writer.add_scalar("val_seen/fwIoU", FWIoU_seen, epoch)
        self.writer.add_scalar("val_unseen/mIoU", mIoU_unseen, epoch)
        self.writer.add_scalar("val_unseen/Acc", Acc_unseen, epoch)
        self.writer.add_scalar("val_unseen/Acc_class", Acc_class_unseen, epoch)
        self.writer.add_scalar("val_unseen/fwIoU", FWIoU_unseen, epoch)

        print("Validation:")
        print("[Epoch: %d, numPcs: %5d]"% (epoch, num))
        print("Loss: {:.3f}".format(test_loss))
        print("Overall: Acc:{}, Acc_class:{}, mIoU:{}, fwIoU:{}".format(Acc, Acc_class, mIoU, FWIoU))
        print("Seen: Acc: {}, Acc_class: {}, mIoU: {}, fwIoU: {}".format(
            Acc_seen, Acc_class_seen, mIoU_seen, FWIoU_seen
            ))
        print("Unseen: Acc:{}, Acc_class:{}, mIoU:{}, fwIoU:{}".format(
            Acc_unseen, Acc_class_unseen, mIoU_unseen, FWIoU_unseen))

        for class_name, acc_value, mIoU_value in zip(self.class_names, Acc_class_by_class, mIoU_by_class):
            self.writer.add_scalar("Acc_by_class/" + class_name, acc_value, epoch)
            self.writer.add_scalar("mIoU_by_class/" + class_name, mIoU_value, epoch)
            print(class_name, "- acc:", acc_value, " mIoU:", mIoU_value)
        is_best = True

        if not self.args.test:
            self.best_pred = mIoU_unseen
            self.saver.save_checkpoint(
                {
                    "epoch": epoch + 1,
                    "state_dict": self.model.module.state_dict(),
                    "optimizer": self.optimizer.state_dict(),
                    "best_pred": self.best_pred,
                },
                is_best,
                generator_state=
                {
                    "epoch": epoch + 1,
                    "state_dict": self.generator.state_dict(),
                    "optimizer": self.optimizer.state_dict(),
                    "best_pred": self.best_pred,
                },
            )
        self.evaluator.reset()

        return mIoU_unseen



def main(dataset_short):
    if dataset_short == "sk":
        from train_point_sk import Trainer
        args = complete_parser_sk()
    elif dataset_short == "sn":
        from train_point_sn import Trainer
        args = complete_parser_sn()
    elif dataset_short == "s3dis":
        from train_point_s3dis import Trainer
        args = complete_parser_s3dis()
        
        
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    if args.cuda:
        try:
            args.gpu_ids = [int(s) for s in args.gpu_ids.split(",")]
        except ValueError:
            raise ValueError(
                "Argument --gpu_ids must be a comma-separated list of integers only"
            )
    
    print(args)
    args.sync_bn = args.cuda and len(args.gpu_ids) > 1
    torch.manual_seed(args.seed)
    trainer = Trainer(args)
    print("Starting Epoch:", trainer.args.start_epoch)
    print("Total Epochs:", trainer.args.epochs)
    if not args.test:
        for epoch in range(trainer.args.start_epoch, trainer.args.epochs):
            trainer.training(epoch, args)
            if epoch % args.eval_interval == (args.eval_interval - 1) or epoch == trainer.args.epochs - 1:
                trainer.validation(epoch, args)
    else:
        print("Only validation")
        trainer.validation(trainer.args.start_epoch, args)

    trainer.writer.close()
