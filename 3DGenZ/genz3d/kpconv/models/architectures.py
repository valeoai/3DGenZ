#
#
#      0=================================0
#      |    Kernel Point Convolutions    |
#      0=================================0
#
#
# ----------------------------------------------------------------------------------------------------------------------
#
#      Define network architectures
#
# ----------------------------------------------------------------------------------------------------------------------
#
#      Hugues THOMAS - 06/03/2020
#


import numpy as np
import torch.nn.functional as F
from genz3d.kpconv.models.blocks import *

def p2p_fitting_regularizer(net):

    fitting_loss = 0
    repulsive_loss = 0

    for m in net.modules():

        if isinstance(m, KPConv) and m.deformable:

            ##############
            # Fitting loss
            ##############

            # Get the distance to closest input point and normalize to be independant from layers
            KP_min_d2 = m.min_d2 / (m.KP_extent ** 2)

            # Loss will be the square distance to closest input point. We use L1 because dist is already squared
            fitting_loss += net.l1(KP_min_d2, torch.zeros_like(KP_min_d2))

            ################
            # Repulsive loss
            ################

            # Normalized KP locations
            KP_locs = m.deformed_KP / m.KP_extent

            # Point should not be close to each other
            for i in range(net.K):
                other_KP = torch.cat([KP_locs[:, :i, :], KP_locs[:, i + 1:, :]], dim=1).detach()
                distances = torch.sqrt(torch.sum((other_KP - KP_locs[:, i:i + 1, :]) ** 2, dim=2))
                rep_loss = torch.sum(torch.clamp_max(distances - net.repulse_extent, max=0.0) ** 2, dim=1)
                repulsive_loss += net.l1(rep_loss, torch.zeros_like(rep_loss)) / net.K

    return net.deform_fitting_power * (2 * fitting_loss + repulsive_loss)





class KPFCNN(nn.Module):
    """
    Class defining KPFCNN
    """

    def __init__(self, config, lbl_values, ign_lbls):
        super(KPFCNN, self).__init__()

        ############
        # Parameters
        ############
        
        # Current radius of convolution and feature dimension
        layer = 0
        r = config.first_subsampling_dl * config.conv_radius
        in_dim = config.in_features_dim
        out_dim = config.first_features_dim
        self.K = config.num_kernel_points
        self.C = len(lbl_values) - len(ign_lbls)
        self.valid_labels = lbl_values
        self.config = config

        #####################
        # List Encoder blocks
        #####################

        # Save all block operations in a list of modules
        self.encoder_blocks = nn.ModuleList()
        self.encoder_skip_dims = []
        self.encoder_skips = []

        # Loop over consecutive blocks
        for block_i, block in enumerate(config.architecture):

            # Check equivariance
            if ('equivariant' in block) and (not out_dim % 3 == 0):
                raise ValueError('Equivariant block but features dimension is not a factor of 3')

            # Detect change to next layer for skip connection
            if np.any([tmp in block for tmp in ['pool', 'strided', 'upsample', 'global']]):
                self.encoder_skips.append(block_i)
                self.encoder_skip_dims.append(in_dim)

            # Detect upsampling block to stop
            if 'upsample' in block:
                break

            # Apply the good block function defining tf ops
            self.encoder_blocks.append(block_decider(block,
                                                    r,
                                                    in_dim,
                                                    out_dim,
                                                    layer,
                                                    config))

            # Update dimension of input from output
            if 'simple' in block:
                in_dim = out_dim // 2
            else:
                in_dim = out_dim

            # Detect change to a subsampled layer
            if 'pool' in block or 'strided' in block:
                # Update radius and feature dimension for next layer
                layer += 1
                r *= 2
                out_dim *= 2

        #####################
        # List Decoder blocks
        #####################

        # Save all block operations in a list of modules
        self.decoder_blocks = nn.ModuleList()
        self.decoder_concats = []

        # Find first upsampling block
        start_i = 0
        for block_i, block in enumerate(config.architecture):
            if 'upsample' in block:
                start_i = block_i
                break

        # Loop over consecutive blocks
        for block_i, block in enumerate(config.architecture[start_i:]):

            # Add dimension of skip connection concat
            if block_i > 0 and 'upsample' in config.architecture[start_i + block_i - 1]:
                in_dim += self.encoder_skip_dims[layer]
                self.decoder_concats.append(block_i)

            # Apply the good block function defining tf ops
            self.decoder_blocks.append(block_decider(block,
                                                    r,
                                                    in_dim,
                                                    out_dim,
                                                    layer,
                                                    config))

            # Update dimension of input from output
            in_dim = out_dim

            # Detect change to a subsampled layer
            if 'upsample' in block:
                # Update radius and feature dimension for next layer
                layer -= 1
                r *= 0.5
                out_dim = out_dim // 2

        self.head_mlp = UnaryBlock(out_dim, config.first_features_dim, False, 0)
        self.head_softmax = UnaryBlock(config.first_features_dim, self.C, False, 0)

        self.head_mlp_generative = UnaryBlock(out_dim, config.first_features_dim, False, 0)
        self.head_softmax_generative = UnaryBlock(config.first_features_dim, self.C, False, 0)

        #self.head_baseline = UnaryBlock(out_dim, config.attribute_size, False, 0, no_relu=True)

        #self.head_baseline_convex_1 = UnaryBlock(out_dim, 256, False, 0)
        #self.head_baseline_convex_2 = UnaryBlock(256, 512, False, 0)
        #self.head_baseline_convex_3 = UnaryBlock(512, config.attribute_size, False, 0, no_relu = True) #No Relu in the last layer to more easily get negative values
        ################
        # Network Losses
        ################

        # List of valid labels (those not ignored in loss)
        self.valid_labels = np.sort([c for c in lbl_values if c not in ign_lbls])

        # Choose segmentation loss
        if len(config.class_w) > 0:
            class_w = torch.from_numpy(np.array(config.class_w, dtype=np.float32))
            self.criterion = torch.nn.CrossEntropyLoss(weight=class_w, ignore_index=-1)
        else:
            self.criterion = torch.nn.CrossEntropyLoss(ignore_index=-1)
        self.deform_fitting_mode = config.deform_fitting_mode
        self.deform_fitting_power = config.deform_fitting_power
        self.deform_lr_factor = config.deform_lr_factor
        self.repulse_extent = config.repulse_extent
        self.output_loss = 0
        self.reg_loss = 0
        self.l1 = nn.L1Loss()

        return
    def backbone(self, batch, config, return_mid=False): 
        # Get input features
        x = batch.features.clone().detach()

        # Loop over consecutive blocks
        skip_x = []
        for block_i, block_op in enumerate(self.encoder_blocks):
            if block_i in self.encoder_skips:
                skip_x.append(x)
            x = block_op(x, batch)

        for block_i, block_op in enumerate(self.decoder_blocks):
            if block_i in self.decoder_concats:
                x = torch.cat([x, skip_x.pop()], dim=1)
            x = block_op(x, batch)
        return x, None


    def forward(self, batch, config):
        x, _ = self.backbone(batch, config)
        # Head of network
        x = self.head_mlp(x, batch)
        x = self.head_softmax(x, batch)
        return x
    
    def forward_generative(self, batch, config): 
        x,_ = self.backbone(batch, config)
        # Head of network, trained with generatives 
        x = self.head_mlp_generative(x, batch)
        x = self.head_softmax_generative(x, batch)
        return x
    
    def trained_ss_ll(self,x, batch, config):
        #In order to use the trained semantic segmentation 
        x = self.head_mlp(x,batch)
        x = self.head_softmax(x, batch)
        return x
    
    def training_generative(self, x): 
        x = self.head_mlp_generative(x)
        x = self.head_softmax_generative(x)
        return x
    

    def loss(self, outputs, labels):
        """
        Runs the loss on outputs of the model
        :param outputs: logits
        :param labels: labels
        :return: loss
        """

        # Set all ignored labels to -1 and correct the other label to be in [0, C-1] range
        target = - torch.ones_like(labels)
        for i, c in enumerate(self.valid_labels):
            target[labels == c] = i

        # Reshape to have a minibatch size of 1
        outputs = torch.transpose(outputs, 0, 1)
        outputs = outputs.unsqueeze(0)
        target = target.unsqueeze(0)

        # Cross entropy loss
        self.output_loss = self.criterion(outputs, target)

        # Regularization of deformable offsets
        if self.deform_fitting_mode == 'point2point':
            self.reg_loss = p2p_fitting_regularizer(self)
        elif self.deform_fitting_mode == 'point2plane':
            raise ValueError('point2plane fitting mode not implemented yet.')
        else:
            raise ValueError('Unknown fitting mode: ' + self.deform_fitting_mode)

        # Combined loss
        return self.output_loss + self.reg_loss
    

    def baseline_loss(self, outputs, labels, embedding, config): 
        if config.baseline:
            #Only the seen classes are used during training
            self.output_loss = 1 - F.cosine_similarity(outputs, embedding).mean()
            

        elif config.convex_baseline: 
            #Cross entropy, but only on the seen classes
            
                # Set all ignored labels to -1 and correct the other label to be in [0, C-1] range
            target = - torch.ones_like(labels)
            for i, c in enumerate(self.valid_labels):
                target[labels == c] = i

            # Reshape to have a minibatch size of 1
            #outputs = torch.transpose(outputs, 0, 1)
            #outputs = outputs.unsqueeze(0)
            #target = target.unsqueeze(0)

            # Cross entropy loss (only on the not ignored classes)
            #print("Outputs size {}".format(outputs.size()))
            #print("target size {}".format(target.size()))
            self.output_loss = self.criterion_convex(outputs.view(-1,self.C), target.view(-1))
        else:
            raise ValueError("Unknown basline mode")

        # Regularization of deformable offsets
        if self.deform_fitting_mode == 'point2point':
            self.reg_loss = p2p_fitting_regularizer(self)
        elif self.deform_fitting_mode == 'point2plane':
            raise ValueError('point2plane fitting mode not implemented yet.')
        else:
            raise ValueError('Unknown fitting mode: ' + self.deform_fitting_mode)

        # Combined loss
        return self.output_loss + self.reg_loss


    def labels_adaption(self, labels):
        """
        Runs the loss on outputs of the model
        :param outputs: logits
        :param labels: labels
        :return: loss
        """

        # Set all ignored labels to -1 and correct the other label to be in [0, C-1] range
        target = - torch.ones_like(labels)
        for i, c in enumerate(self.valid_labels):
            target[labels == c] = i

        
        target = target.unsqueeze(0)

       

        return target 

    def accuracy(self, outputs, labels):
        """
        Computes accuracy of the current batch
        :param outputs: logits predicted by the network
        :param labels: labels
        :return: accuracy value
        """

        # Set all ignored labels to -1 and correct the other label to be in [0, C-1] range
        target = - torch.ones_like(labels)
        for i, c in enumerate(self.valid_labels):
            target[labels == c] = i

        predicted = torch.argmax(outputs.data, dim=1)
        total = target.size(0)
        correct = (predicted == target).sum().item()

        return correct / total
    
    def accuracy_baseline(self, outputs, labels):
        """
        Computes accuracy of the current batch
        :param outputs: number label predicted by the network
        :param labels: labels
        :return: accuracy value
        """

        # Set all ignored labels to -1 and correct the other label to be in [0, C-1] range
        #target = - torch.ones_like(labels)
        #for i, c in enumerate(self.valid_labels):
        #    target[labels == c] = i

        #predicted = torch.argmax(outputs.data, dim=1)
        predicted = outputs.data
        total = target.size(0)
        correct = (predicted == target).sum().item()

        return correct / total


    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm1d):
                m.eval()

    def freeze_backbone_fcout(self): 
        for name, param in self.named_modules():
            if name == "head_mlp_generative":
                param.requires_grad = True
            else: 
                param.requires_grad = False

    def get_1x_lr_params(self): 
            for name, param in self.named_parameters():
                if name != "head_mlp_generative": 
                    if param.requires_grad:
                        yield param

    def get_10x_lr_params(self): 
            for name, param in self.named_parameters():
                if name == "head_mlp_generative":
                    if param.requires_grad:
                        yield param




















