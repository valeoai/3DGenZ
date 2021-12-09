# imports

import os
import time
import pickle
import argparse
import logging
import yaml
import numpy as np
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
from pathlib import Path
import torch
import torch.nn.functional as F
import torch.utils.data

import genz3d.fkaconv.lightconvpoint.utils.transformations as lcp_transfo
import genz3d.fkaconv.lightconvpoint.utils.metrics as metrics
from genz3d.fkaconv.lightconvpoint.networks.fkaconv import FKAConv as Network
from genz3d.fkaconv.lightconvpoint.utils.misc import wblue, wgreen
from genz3d.fkaconv.lightconvpoint.nn import with_indices_computation_rotation
from genz3d.word_representations.word_representations_utils import get_word_vector
import genz3d.convpoint.convpoint.knn.lib.python.nearest_neighbors as nearest_neighbors

#import valeodata


def nearest_correspondance(pts_src, pts_dest, data_src, K=1):
    indices = nearest_neighbors.knn(pts_src.astype(np.float32), pts_dest.astype(np.float32), K, omp=True)
    if K == 1:
        indices = indices.ravel()
        data_dest = data_src[indices]
    else:
        data_dest = data_src[indices].mean(1)
    return data_dest


def get_data(data, attributes=False, device="cuda", test=False):
    pts = data['pts'].to(device)
    features = data['features'].to(device)
    seg = data['target'].to(device)
    net_ids = data["net_indices"]
    net_pts = data["net_support"]
    for i in range(len(net_ids)):
        net_ids[i] = net_ids[i].to(device)
    for i in range(len(net_pts)):
        net_pts[i] = net_pts[i].to(device)
    if test:
        return features, pts, seg, net_ids, net_pts
    else:
        idx = data["idx"]
        if attributes:
            attributes = data["attributes"].to(device)
            return features, pts, seg, net_ids, net_pts, idx, attributes
        else:
            return features, pts, seg, net_ids, net_pts, idx


class Dataset():

    def __init__(
            self, directory,
            split="train",
            iter_nbr=None,
            pillar_size=2,
            network_function=None,
            transformations_data=None,
            transformations_points=None,
            transformations_features=None,
            no_attribute=True,
            ignore_value=np.array([0, 5, 7, 8, 11]),
            attribute="glove_w2v"
    ):

        self.directory = directory
        self.split = split
        self.t_data = transformations_data
        self.t_points = transformations_points
        self.t_features = transformations_features
        self.no_attribute = no_attribute
        self.attribute = attribute
        if not self.no_attribute:
            self.Glove_num_dict, self.w2v_num_dict, self.img_embd_array = get_word_vector(dataset="scannet")
        print("directory {}".format(self.directory))
        self.data_filename = os.path.join(self.directory, 'scannet_%s.pickle' % (split))
        with open(self.data_filename, 'rb') as fp:
            self.scene_points_list = pickle.load(fp, encoding="latin1")
            self.semantic_labels_list = pickle.load(fp, encoding="latin1")

        if self.split == 'train':
            labelweights = np.zeros(21)
            for seg in self.semantic_labels_list:
                tmp, _ = np.histogram(seg, range(22))
                labelweights += tmp
            labelweights = labelweights.astype(np.float32)
            labelweights = labelweights / np.sum(labelweights)
            self.labelweights = 1 / np.log(1.2 + labelweights)
        elif self.split == 'test':
            self.labelweights = np.ones(21)

        if network_function is not None:
            self.net = network_function()
        else:
            self.net = None

        self.pillar_size = pillar_size
        self.pillar_infinite_dim = 2

        self.ignore_value = ignore_value  # labels to ignore
        print("self ignore value {}".format(self.ignore_value))
        self.number_of_tentative_pillar = 10
        self.counter_idx = {}

        for v in self.ignore_value:
            self.labelweights[v] = 0

        self.labelweights[0] = 0  # ignore unlabels

        self.classes = [
            'unannotated',  # 0
            'wall',  # 1
            'floor',  # 2
            'chair',  # 3
            'table',  # 4
            'desk',  # 5
            'bed',  # 6
            'bookshelf',  # 7
            'sofa',  # 8
            'sink',  # 9
            'bathtub',  # 10
            'toilet',  # 11
            'curtain',  # 12
            'counter',  # 13
            'door',  # 14
            'window',  # 15
            'shower curtain',  # 16
            'refrigerator',  # 17
            'picture',  # 18
            'cabinet',  # 19
            'otherfurniture',  # 20
        ]

        if iter_nbr is None:
            self.iter_nbr = len(self.scene_points_list)
        else:
            self.iter_nbr = iter_nbr

    def __len__(self):
        return self.iter_nbr

    @with_indices_computation_rotation
    def __getitem__(self, idx):

        if self.iter_nbr != len(self.scene_points_list):
            idx = torch.randint(0, len(self.scene_points_list), size=(1,)).item()

        count_change_idx = 0
        go_on = True

        data = self.scene_points_list[idx]
        lbs = self.semantic_labels_list[idx]

        while go_on:

            data = self.scene_points_list[idx]
            lbs = self.semantic_labels_list[idx]

            # pillar selection
            count_id = 0
            while go_on and (count_id < self.number_of_tentative_pillar):
                pillar_center = data[torch.randint(0, data.shape[0], size=(1,)).item(), :3]
                mask = None
                for i in range(pillar_center.shape[0]):
                    if self.pillar_infinite_dim != i:
                        mask_i = np.logical_and(data[:, i] <= pillar_center[i] + self.pillar_size / 2,
                                                data[:, i] >= pillar_center[i] - self.pillar_size / 2)
                        if mask is None:
                            mask = mask_i
                        else:
                            mask = np.logical_and(mask, mask_i)

                unique = np.unique(lbs[mask])
                if np.intersect1d(unique, self.ignore_value, assume_unique=True).shape[0] == 0:
                    go_on = False

                    if idx in self.counter_idx:
                        self.counter_idx[idx] = self.counter_idx[idx] + 1
                    else:
                        self.counter_idx[idx] = 1
                count_id += 1

            if go_on:
                idx = torch.randint(0, len(self.scene_points_list), size=(1,)).item()
                count_change_idx += 1
                logging.debug(f"Changing scene idx {count_change_idx}")
            lbs = lbs[mask]
            data = data[mask]

        lbs = np.expand_dims(lbs, axis=1)
        data = np.concatenate([data, lbs], axis=1)

        if self.t_data is not None:
            for t in self.t_data:
                data = t(data)

        # get the features, labels and points
        lbs = data[:, 3].astype(int)
        pts = data[:, :3]
        fts = np.ones_like(pts)

        if not self.no_attribute:
            if self.attribute == "w2v":
                attributes = np.array(list(map(lambda x: self.w2v_num_dict[x], np.squeeze(lbs.astype(np.float32)))), dtype=np.float32)

            elif self.attribute == "glove":
                attributes = np.array(list(map(lambda x: self.Glove_num_dict[x], np.squeeze(lbs.astype(np.float32)))), dtype=np.float32)
            elif self.attribute == "glove_w2v":
                w2v_gt_attributes = np.array(list(map(lambda x: self.w2v_num_dict[x], np.squeeze(lbs.astype(np.float32)))), dtype=np.float32)
                glove_gt_attributes = np.array(list(map(lambda x: self.Glove_num_dict[x], np.squeeze(lbs.astype(np.float32)))), dtype=np.float32)
                attributes = np.hstack((glove_gt_attributes, w2v_gt_attributes.astype(np.float32)))
            else:
                attributes = np.array(list(map(lambda x: self.img_embd_array[x], np.squeeze(lbs.astype(np.float32)))), dtype=np.float32)
        choice = 0  # not used at training or validation

        # apply transformations on points
        if self.t_points is not None:
            for t in self.t_points:
                pts = t(pts)

        # apply transformations on features
        if self.t_features is not None:
            for t in self.t_features:
                fts = t(fts)

        pts = torch.from_numpy(pts).float()
        fts = torch.from_numpy(fts).float()
        lbs = torch.from_numpy(lbs).long()

        pts = pts.transpose(0, 1)
        fts = fts.transpose(0, 1)
        if not self.no_attribute:
            attributes = torch.from_numpy(attributes).float()
            return_dict = {
                "pts": pts,
                "features": fts,
                "target": lbs,
                "pts_ids": choice,
                "idx": idx,
                "attributes": attributes
            }
            return return_dict

        else:
            return_dict = {
                "pts": pts,
                "features": fts,
                "target": lbs,
                "pts_ids": choice,
                "idx": idx
            }
            return return_dict


# Part dataset only for testing
class PartDatasetTest():
    def compute_mask(self, pt, bs):
        # build the mask
        mask_x = np.logical_and(self.xyzrgb[:, 0] <= pt[0] + bs / 2, self.xyzrgb[:, 0] >= pt[0] - bs / 2)
        mask_y = np.logical_and(self.xyzrgb[:, 1] <= pt[1] + bs / 2, self.xyzrgb[:, 1] >= pt[1] - bs / 2)
        mask = np.logical_and(mask_x, mask_y)
        return mask

    def __init__(self, pts, labels,
                 block_size=4, npoints=4096,
                 min_pick_per_point=1, test_step=0.5, transformations_data=None, transformations_points=None, transformations_features=None, network_function=None,):

        self.t_data = transformations_data
        self.t_points = transformations_points
        self.t_features = transformations_features
        self.bs = block_size
        self.npoints = npoints
        self.verbose = False
        self.min_pick_per_point = min_pick_per_point
        if network_function is not None:
            self.net = network_function()
        else:
            self.net = None

        # load data
        self.xyzrgb = pts
        self.labels = labels

        step = test_step
        mini = self.xyzrgb[:, :2].min(0)
        discretized = ((self.xyzrgb[:, :2] - mini).astype(float) / step).astype(int)
        self.pts = np.unique(discretized, axis=0)
        self.pts = self.pts.astype(np.float) * step + mini + step / 2

    @with_indices_computation_rotation
    def __getitem__(self, index):

        # get the data
        mask = self.compute_mask(self.pts[index], self.bs)
        data = self.xyzrgb[mask]

        # choose right number of points
        choice = np.random.choice(data.shape[0], self.npoints, replace=True)
        data = data[choice]

        # labels will contain indices in the original point cloud
        lbs = np.where(mask)[0][choice]
        lbs = np.expand_dims(lbs, axis=1).astype(int)
        data = np.concatenate([data, lbs], axis=1)

        if self.t_data is not None:
            for t in self.t_data:
                data = t(data)
        # get the features, labels and points
        lbs = data[:, 3]
        pts = data[:, :3]
        fts = np.ones_like(pts)

       # apply transformations on points
        if self.t_points is not None:
            for t in self.t_points:
                pts = t(pts)

        # apply transformations on features
        if self.t_features is not None:
            for t in self.t_features:
                fts = t(fts)

        pts = torch.from_numpy(pts).float()
        fts = torch.from_numpy(fts).float()
        lbs = torch.from_numpy(lbs).long()

        pts = pts.transpose(0, 1)
        fts = fts.transpose(0, 1)

        return_dict = {
            "pts": pts,
            "features": fts,
            "target": lbs,

        }
        return return_dict

    def __len__(self):
        return self.pts.shape[0]


def test(args, config):

    logging.info("getting the dataset")
    data_dir = config["dataset_dir"]
    device = torch.device(config["device"])

    logging.info("save the config file")
    save_dir = config["save_dir"]
    os.makedirs(save_dir, exist_ok=True)
    yaml.dump(eval(str(config)), open(os.path.join(save_dir, "config.yaml"), "w"))
    split = "test"
    N_CLASSES = 21

    seen_classes_idx_metric = np.arange(21)
    seen_classes_idx_metric = np.delete(
        seen_classes_idx_metric, np.array(config["ignore_value"])
    )

    validation_transformations_data = [
        # lcp_transfo.PillarSelection(config["dataset_pillar_size"]),
        lcp_transfo.RandomSubSample(config["dataset_num_points"])
    ]

    def network_function():
        return Network(3, 21, segmentation=True)

    data_filename = os.path.join(data_dir, 'scannet_%s.pickle' % (split))
    with open(data_filename, 'rb') as fp:
        scene_points_list = pickle.load(fp, encoding="latin1")
        semantic_labels_list = pickle.load(fp, encoding="latin1")

    print("scene points list {}".format(len(scene_points_list)))
    object_list_len = len(scene_points_list)
    device = torch.device("cuda")

    net = network_function()
    net.to(device)
    model = net
    # Load the ConvBasic version
    chkp_path = os.path.join(args.ckpt_path)
    print("Load the complete trained model from {}".format(chkp_path))
    checkpoint = torch.load(chkp_path)
    model.load_state_dict(checkpoint['state_dict'], strict=False)

    # create the network
    net.cuda()
    net.eval()

    for i in range(object_list_len):
        print(i)
        filename = str(i)
        pts = scene_points_list[i]
        gt_label = semantic_labels_list[i]
        print("pts shape {}".format(pts.shape))
        print("gt label {}".format(gt_label.shape))

        ds = PartDatasetTest(pts=pts, labels=gt_label,
                             block_size=4, npoints=config["dataset_num_points"],
                             min_pick_per_point=1, test_step=0.5, transformations_data=validation_transformations_data, network_function=network_function)
        loader = torch.utils.data.DataLoader(ds, batch_size=config["batch_size"], shuffle=False,
                                             num_workers=config['threads']
                                             )

        xyzrgb = ds.xyzrgb[:, :3]
        print("XYZRGB {}".format(xyzrgb.shape))
        scores = np.zeros((xyzrgb.shape[0], N_CLASSES))
        number_predicted = np.zeros((xyzrgb.shape[0]))
        total_time = 0
        iter_nb = 0
        with torch.no_grad():
            t = tqdm(loader, ncols=80)
            for data in t:

                t1 = time.time()
                x, pos, indices, net_ids, net_pts = get_data(data, test=True)
                outputs = model.forward(x=x, pos=pos, support_points=net_pts, indices=net_ids, gen_forward=True, backbone=False)
                outputs_np = np.swapaxes(outputs.cpu().numpy(), 1, 2).reshape((-1, N_CLASSES))
                t2 = time.time()
                scores[indices.cpu().numpy().ravel()] += outputs_np
                # Add the number of points
                number_predicted[indices.cpu().numpy().ravel()] = number_predicted[indices.cpu().numpy().ravel()] + 1

                iter_nb += 1
                total_time += (t2 - t1)
                t.set_postfix(time="{:05e}".format(total_time / (iter_nb * config["batch_size"])))

        mask = np.logical_not(scores.sum(1) == 0)
        scores = scores[mask]
        pts_src = xyzrgb[mask]

        # create the scores for all points
        scores = nearest_correspondance(pts_src, xyzrgb, scores, K=1)

        # compute softmax
        scores = scores - scores.max(axis=1)[:, None]
        scores = np.exp(scores) / np.exp(scores).sum(1)[:, None]
        scores = np.nan_to_num(scores)
        bias_matrix = np.zeros(scores.shape)
        bias_matrix[:, seen_classes_idx_metric] = 1 * float(config["bias"])
        print("fkiat {}".format(float(config["bias"])))
        os.makedirs(os.path.join(config["save_dir_pred"], filename), exist_ok=True)
        # saving labels
        save_fname = os.path.join(config["save_dir_pred"], filename, "pred.txt")

        # Bias values
        scores = scores - bias_matrix
        scores = scores.argmax(1)
        np.savetxt(save_fname, scores, fmt='%d')

        print("Gt label shape {}".format(gt_label.shape))

        # Save all the files
        tmp_save_path = os.path.join(config["save_dir_pred"], filename)
        Path(tmp_save_path).mkdir(parents=True, exist_ok=True)
        np.save(os.path.join(tmp_save_path, "label"), gt_label)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Run experiments with repetition character")

    # Setting the config file
    parser.add_argument('--config_name', type=str, default="config.yaml", help="Defines the used attribute. w2v, glove and glove_w2v")
    parser.add_argument('--ckpt_path', type=str, default="config.yaml", help="Defines the used attribute. w2v, glove and glove_w2v")
    parser.add_argument('--test', type=bool, default=False)

    pars_args = parser.parse_args()

    logging.info("Loading configuration file")
    config = yaml.load(open("{}".format(pars_args.config_name), "r"), Loader=yaml.FullLoader)

    logging.getLogger().setLevel(config["mode"])

    if logging.getLogger().getEffectiveLevel() == logging.DEBUG:
        logging.debug(" in DEBUG mode...")
        config["threads"] = 0

    if pars_args.test:
        test(pars_args, config)
    else:
        main(config)
