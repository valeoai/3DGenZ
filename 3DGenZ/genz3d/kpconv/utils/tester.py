#
#
#      0=================================0
#      |    Kernel Point Convolutions    |
#      0=================================0
#
#
# ----------------------------------------------------------------------------------------------------------------------
#
#      Class handling the test of any model
#
# ----------------------------------------------------------------------------------------------------------------------
#
#      Hugues THOMAS - 11/06/2018
#


# ----------------------------------------------------------------------------------------------------------------------
#
#           Imports and global variables
#       \**********************************/
#


# Basic libs
import torch
import torch.nn as nn
import numpy as np
from os import makedirs, listdir
from os.path import exists, join
import time
import json
from sklearn.neighbors import KDTree

# Metrics
from genz3d.kpconv.utils.metrics import IoU_from_confusions, fast_confusion
from sklearn.metrics import confusion_matrix

#from utils.visualizer import show_ModelNet_models

# ----------------------------------------------------------------------------------------------------------------------
#
#           Tester Class
#       \******************/
#


class ModelTester:

    # Initialization methods
    # ------------------------------------------------------------------------------------------------------------------

    def __init__(self, net, chkp_path=None, on_gpu=True, baseline=False, generative=False):

        ############
        # Parameters
        ############

        # Choose to train on CPU or GPU
        if on_gpu and torch.cuda.is_available():
            self.device = torch.device("cuda:0")
        else:
            self.device = torch.device("cpu")
        net.to(self.device)

        ##########################
        # Load previous checkpoint
        ##########################
        print("Load chkpt {}".format(chkp_path))
        checkpoint = torch.load(chkp_path)
        self.baseline = baseline
        self.generative = generative
        if self.generative: 
            try: 
                net.load_state_dict(checkpoint['state_dict'])
            except Exception as e: 
                print("Problem loading the weights in strict mode(try in non strict now): {}".format(e))
                try:
                    net.load_state_dict(checkpoint['state_dict'], strict =False)
                except: 
                    try: 
                        net.load_state_dict(checkpoint['model_state_dict'] )
                    except Exception as e: 
                        print("Model state dict did not work on strict mode. {}".format(e))
                        
                        net.load_state_dict(checkpoint['model_state_dict'], strict=False)
        else:
            try: 
                 net.load_state_dict(checkpoint['model_state_dict'])
            except Exception as e: 
                print("Problem loading the weights in strict mode(try in non strict now): {}".format(e))
                net.load_state_dict(checkpoint['model_state_dict'], strict =False)
           
        self.epoch = checkpoint['epoch']
        net.eval()
        print("Model and training state restored.")

        return

    # Test main methods
    # ------------------------------------------------------------------------------------------------------------------

    

    

    def slam_segmentation_test(self, net, test_loader, config, num_votes=100, debug=True, bias=0.0, weight= 50, bias_elements = [0,3, 4, 7, 19]):
        """
        Test method for slam segmentation models
        """

        ############
        # Initialize
        ############
        
        visualisation = False

        print("Save the predictions for visualisation: {}".format(visualisation))
        # Choose validation smoothing parameter (0 for no smothing, 0.99 for big smoothing)
        test_smooth = 0.5
        last_min = -0.5
        softmax = torch.nn.Softmax(1)

        # Number of classes including ignored labels
        nc_tot = test_loader.dataset.num_classes
        nc_model = net.C

        # Test saving path
        test_path = None
        report_path = None
        if config.saving:
            test_path = join('test', 'bias_{}_weight_{}'.format(bias,weight),config.saving_path.split('/')[-1])
            if not exists(test_path):
                makedirs(test_path)
            report_path = join(test_path, 'reports')
            if not exists(report_path):
                makedirs(report_path)
        if test_loader.dataset.set == 'validation':
            for folder in ['val_predictions', 'val_probs']:
                if not exists(join(test_path, folder)):
                    makedirs(join(test_path, folder))
        else:
            for folder in ['predictions', 'probs']:
                if not exists(join(test_path, folder)):
                    makedirs(join(test_path, folder))

        # Init validation container
        all_f_preds = []
        all_f_labels = []
        if test_loader.dataset.set == 'validation':
            for i, seq_frames in enumerate(test_loader.dataset.frames):
                all_f_preds.append([np.zeros((0,), dtype=np.int32) for _ in seq_frames])
                all_f_labels.append([np.zeros((0,), dtype=np.int32) for _ in seq_frames])

        #####################
        # Network predictions
        #####################

        predictions = []
        targets = []
        test_epoch = 0

        t = [time.time()]
        last_display = time.time()
        mean_dt = np.zeros(1)
        temp_sum = 0.0
        avg_counter = 0
        total_sum = 0.0
        total_counter = 0

        # Start test loop
        while True:
            print('Initialize workers')
            epoch_sum = 0.0
            epoch_counter = 0
            for i, batch in enumerate(test_loader):
                temp_sum = 0.0
                avg_counter = 0
                # New time
                t = t[-1:]
                t += [time.time()]

                if i == 0:
                    print('Done in {:.1f}s'.format(t[1] - t[0]))

                if 'cuda' in self.device.type:
                    batch.to(self.device)

                # Forward pass
                if self.generative: 
                    outputs = net.forward_generative(batch, config)
                #else: 
                #    outputs = net(batch, config)

                # Get probs and labels
                stk_probs = softmax(outputs).cpu().detach().numpy()

                lengths = batch.lengths[0].cpu().numpy()
                f_inds = batch.frame_inds.cpu().numpy()
                r_inds_list = batch.reproj_inds
                r_mask_list = batch.reproj_masks
                labels_list = batch.val_labels
                torch.cuda.synchronize(self.device)

                t += [time.time()]

                # Get predictions and labels per instance
                # ***************************************

                i0 = 0
                for b_i, length in enumerate(lengths):

                    # Get prediction
                    probs = stk_probs[i0:i0 + length]
                    proj_inds = r_inds_list[b_i]
                    proj_mask = r_mask_list[b_i]
                    frame_labels = labels_list[b_i]
                    s_ind = f_inds[b_i, 0]
                    f_ind = f_inds[b_i, 1]
                    
                    # Project predictions on the frame points
                    proj_probs = probs[proj_inds]

                    # Safe check if only one point:
                    if proj_probs.ndim < 2:
                        proj_probs = np.expand_dims(proj_probs, 0)

                    # Save probs in a binary file (uint8 format for lighter weight)
                    seq_name = test_loader.dataset.sequences[s_ind]
                    if test_loader.dataset.set == 'validation':
                        folder = 'val_probs'
                        pred_folder = 'val_predictions'
                    else:
                        folder = 'probs'
                        pred_folder = 'predictions'
                    filename = '{:s}_{:07d}.npy'.format(seq_name, f_ind)
                    filepath = join(test_path, folder, filename)
                    if exists(filepath):
                        try:
                           frame_probs_uint8 = np.load(filepath)
                        except Exception as e: 
                            print("e: {}".format(e))
                            print("filepath: {}".format(filepath))
                    else:
                        frame_probs_uint8 = np.zeros((proj_mask.shape[0], nc_model), dtype=np.uint8)
                    frame_probs = frame_probs_uint8[proj_mask, :].astype(np.float32) / 255
                    frame_probs = test_smooth * frame_probs + (1 - test_smooth) * proj_probs
                    frame_probs_uint8[proj_mask, :] = (frame_probs * 255).astype(np.uint8)
                    np.save(filepath, frame_probs_uint8)

                    
                        
                    # Save some prediction in ply format for visual
                    if test_loader.dataset.set == 'validation':

                        # Insert false columns for ignored labels
                        frame_probs_uint8_bis = frame_probs_uint8.copy()
                        for l_ind, label_value in enumerate(test_loader.dataset.label_values):
                            if label_value in test_loader.dataset.ignored_labels:
                                frame_probs_uint8_bis = np.insert(frame_probs_uint8_bis, l_ind, 0, axis=1)

                        # Predicted labels
                        tmp = frame_probs_uint8_bis.astype(np.float32)/255
                        elements = np.arange(0,20)
                        seen_classes_idx_metric = np.delete(elements, bias_elements).tolist()
                        stk_probs_bias = np.zeros(frame_probs_uint8_bis.shape)
                        stk_probs_bias[:,[ seen_classes_idx_metric]] = 1.0 * bias
                        
                        tmp = tmp - stk_probs_bias
                        #frame_preds = test_loader.dataset.label_values[np.argmax(frame_probs_uint8_bis,
                        #                                                         axis=1)].astype(np.int32)
                         
                        frame_preds =  test_loader.dataset.label_values[np.argmax(tmp,
                                                                                 axis=1)].astype(np.int32)

                        
                        

                        # keep frame preds in memory
                        #all_f_preds[s_ind][f_ind] = frame_preds
                        #all_f_labels[s_ind][f_ind] = frame_labels
                        #print(np.sum(frame_preds==0)*100.0/frame_preds.shape[0])
                        temp_sum += np.sum(frame_preds==0)*100.0/frame_preds.shape[0]
                        avg_counter += 1

                   

                    # Stack all prediction for this epoch
                    i0 += length

                # Average timing
                t += [time.time()]
                mean_dt = 0.95 * mean_dt + 0.05 * (np.array(t[1:]) - np.array(t[:-1]))

                avg_batch = temp_sum * 1.0/avg_counter
                total_sum += avg_batch
                epoch_sum += avg_batch
                total_counter += 1
                epoch_counter += 1
                current_total_avg = total_sum * 1.0 / total_counter

                # Display
                if (t[-1] - last_display) > 1.0:
                    last_display = t[-1]
                    message = 'e{:03d}-i{:04d} => {:.1f}% (timings : {:4.2f} {:4.2f} {:4.2f}) / pots {:d} => {:.1f}%, batch_avg {:3.2f}, total avg:{:3.2f}'
                    min_pot = int(torch.floor(torch.min(test_loader.dataset.potentials)))
                    pot_num = torch.sum(test_loader.dataset.potentials > min_pot + 0.5).type(torch.int32).item()
                    current_num = pot_num + (i + 1 - config.validation_size) * config.val_batch_num
                    print(message.format(test_epoch, i,
                                         100 * i / config.validation_size,
                                         1000 * (mean_dt[0]),
                                         1000 * (mean_dt[1]),
                                         1000 * (mean_dt[2]),
                                         min_pot,
                                         100.0 * current_num / len(test_loader.dataset.potentials), avg_batch, current_total_avg))


            # Update minimum od potentials
            new_min = torch.min(test_loader.dataset.potentials)
            print('Test epoch {:d}, end. Min potential = {:.1f}'.format(test_epoch, new_min))
            print("Avg. per epoch  {}".format(epoch_sum * 1.0 / epoch_counter))
            #print("Size of all pred {}".format())
            if last_min + 1 < new_min:

                # Update last_min
                last_min += 1

                if (test_loader.dataset.set == 'validation' and last_min % 1) == 0 or (last_min > num_votes):

                    #####################################
                    # Results on the whole validation set
                    #####################################

                    # Confusions for our subparts of validation set
                    Confs = np.zeros((len(predictions), nc_tot, nc_tot), dtype=np.int32)
                    for i, (preds, truth) in enumerate(zip(predictions, targets)):
                        print("in preds {}".format(preds))
                        # Confusions
                        Confs[i, :, :] = fast_confusion(truth, preds, test_loader.dataset.label_values).astype(np.int32)


                    # Show vote results
                    print('\nCompute confusion')

                    val_preds = []
                    val_labels = []
                    t1 = time.time()
                    for i, seq_frames in enumerate(test_loader.dataset.frames):
                        
                        val_preds += [np.hstack(all_f_preds[i])]
                        val_labels += [np.hstack(all_f_labels[i])]
                    print("val preds {}".format(val_preds))
                    val_preds = np.hstack(val_preds)
                    print("val preds np {}".format(val_preds))
                    print("labels preds {}".format(val_labels))

                    val_labels = np.hstack(val_labels)
                    t2 = time.time()
                    print("val preds shape {}".format(val_preds.shape))
                    print("val labels shape {}".format(val_labels.shape))

                    C_tot = fast_confusion(val_labels, val_preds, test_loader.dataset.label_values)
                    t3 = time.time()
                    print(' Stacking time : {:.1f}s'.format(t2 - t1))
                    print('Confusion time : {:.1f}s'.format(t3 - t2))

                    s1 = '\n'
                    for cc in C_tot:
                        for c in cc:
                            s1 += '{:7.0f} '.format(c)
                        s1 += '\n'
                    if debug:
                        print(s1)

                    # Remove ignored labels from confusions
                    for l_ind, label_value in reversed(list(enumerate(test_loader.dataset.label_values))):
                        if label_value in test_loader.dataset.ignored_labels:
                            print("Ignored labels {}".format(label_value))
                            C_tot = np.delete(C_tot, l_ind, axis=0)
                            C_tot = np.delete(C_tot, l_ind, axis=1)

                    # Objects IoU
                    val_IoUs = IoU_from_confusions(C_tot)

                    # Compute IoUs
                    mIoU = np.mean(val_IoUs)
                    s2 = '{:5.2f} | '.format(100 * mIoU)
                    for IoU in val_IoUs:
                        s2 += '{:5.2f} '.format(100 * IoU)
                    print(s2 + '\n')

                    # Save a report
                    report_file = join(report_path, 'report_{:04d}_gen_{}_time_{}_bias{}_weight_{}.txt'.format(int(np.floor(last_min)),self.generative, time.time(), bias, weight))
                    str = 'Report of the confusion and metrics\n'
                    str += "Time created: {}\n\n".format(time.time())
                    str += '***********************************\n\n\n'
                    str += 'Confusion matrix:\n\n'
                    str += s1
                    str += '\nIoU values:\n\n'
                    str += s2
                    str += '\n\n'
                    with open(report_file, 'w') as f:
                        f.write(str)

            test_epoch += 1

            # Break when reaching number of desired votes
            if last_min > num_votes:
                break

        return

























