# Contains all the  functions neccessary for word_representations in semantic segmentation
import pickle
import os
from pathlib import Path

def get_word_vector(dataset="s3dis"):
    abs_path = os.path.dirname(os.path.abspath(__file__))
    
    def pickle_load_w2v_glove(dataset):
        with open(os.path.join(abs_path, "{}_Glove_dict_name".format(dataset)), "rb") as file:
            Glove_name_dict = pickle.load(file)
        with open(os.path.join(abs_path, "{}_Glove_dict_num".format(dataset)), "rb") as file:
            Glove_num_dict = pickle.load(file)

        with open(os.path.join(abs_path, "{}_w2v_dict_name".format(dataset)), "rb") as file:
            w2v_name_dict = pickle.load(file)

        with open(os.path.join(abs_path, "{}_w2v_dict_num".format(dataset)), "rb") as file:
            w2v_num_dict = pickle.load(file)

        return Glove_name_dict, Glove_num_dict, w2v_name_dict, w2v_num_dict


    if dataset == "s3dis":
        #S3DIS dataset
        Glove_num_dict, w2v_num_dict, img_embd = None, None, None
        try:
            _, Glove_num_dict, _, w2v_num_dict = pickle_load_w2v_glove("s3dis")
        except Exception as e:
            print("Did not found word representations {}".format(e))

    elif dataset == "sk":
        #SemanticKITTI dataset
        Glove_num_dict, w2v_num_dict, img_embd = None, None, None
        try:
            _, Glove_num_dict, _, w2v_num_dict = pickle_load_w2v_glove("sk")
        except Exception as e:
            print("Did not found word representations {}".format(e))

    elif dataset == "scannet":
        #SemanticNet dataset
        Glove_num_dict, w2v_num_dict, img_embd = None, None, None
        try:
            _, Glove_num_dict, _, w2v_num_dict = pickle_load_w2v_glove("sn")
        except Exception as e:
            raise ValueError("W2V/GloVe files for ScanNet not available {}".format(e))
        img_embd = None


    else:
        raise ValueError('Name of dataset is not known')

    return  Glove_num_dict, w2v_num_dict, img_embd
