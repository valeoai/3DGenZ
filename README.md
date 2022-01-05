<div align="center">
  <h1> 
   Generative Zero-Shot Learning for Semantic Segmentation of 3D Point Clouds
  </h1>
Björn Michele<sup>1)</sup>,  <a href="https://boulch.eu/">Alexandre Boulch</a><sup>1)</sup>, <a href="https://sites.google.com/site/puygilles/">Gilles Puy</a><sup>1)</sup>, Maxime Bucher<sup>1)</sup> and <a href="http://imagine.enpc.fr/~marletr/">Renaud Marlet</a><sup>1)2)</sup>
  
<sup>1)</sup> [Valeo.ai](https://valeoai.github.io/blog/)  <sup>2)</sup>LIGM, Ecole des Ponts, Univ Gustave Eiffel, CNRS, Marne-la-Vallée, France
</div>

Accepted at [3DV 2021](https://3dv2021.surrey.ac.uk/papers/042.html)\
Arxiv: [Paper and Supp.](https://arxiv.org/pdf/2108.06230.pdf)  
[Poster](pres_material/3DGenZ_poster.pdf) or [Presentation](pres_material/3dGenZ_pres_10min.pdf)



**Abstract**: While there has been a number of studies on Zero-Shot Learning (ZSL) for 2D images, 
its application to 3D data is still recent and scarce, 
with just a few methods limited to classification. 
We present the first generative approach for
both ZSL and Generalized ZSL (GZSL) on 3D data, that can
handle both classification and, for the first time, semantic
segmentation. We show that it reaches or outperforms the
state of the art on ModelNet40 classification for both inductive ZSL and inductive GZSL. For semantic segmentation,
we created three benchmarks for evaluating this new ZSL
task, using S3DIS, ScanNet and SemanticKITTI. Our experiments show that our method outperforms strong baselines,
which we additionally propose for this task.

If you want to cite this work: 


```
@inproceedings{michele2021generative,
  title={Generative Zero-Shot Learning for Semantic Segmentation of {3D} Point Cloud},
  author={Michele, Bj{\"o}rn and Boulch, Alexandre and Puy, Gilles and Bucher, Maxime and Marlet, Renaud},
  booktitle={International Conference on 3D Vision (3DV)},
  year={2021}
}
```

# Code 
We provide in this repository the code and the pretrained models for the semantic segmentation tasks on SemanticKITTI and ScanNet. 


### To-Do: 
- We will add more experiments in the future (You could "watch" the repo to stay updated).


## Code  Semantic Segmentation

### Installation
Dependencies: 
Please see [requirements.txt](requirements.txt) for all needed code libraries. 
Tested with: Pytorch 1.6.0 and 1.7.1 (both Cuda 10.1). As torch-geometric is needed Pytoch >= 1.4.0 is required. 

1. Clone this repository.
2. Download and/or install the backbones (ConvPoint is also necessary for our adaption of FKAConv. More information: [ConvPoint](ConvPoint/convpoint), [FKAConv](url), [KP-Conv](https://github.com/HuguesTHOMAS/KPConv-PyTorch/blob/master/INSTALL.md)).
    - For ConvPoint:
    ```
    cd 3DGenZ/genz3d/convpoint/convpoint/knn
    python3 setup.py install --home="."
    ```
    - For FKAConv:
    ```
    cd 3DGenZ/genz3d/fkaconv
    pip install -ve . 
    ```
    - For KPConv have a look at: [INSTALL.md](3DGenZ/genz3d/kpconv/INSTALL.md)
4. Download the datasets. 
    - For an out of the box start we recommend the following folder structure. 
    ```
    ~/3DGenZ
    ~/data/scannet/
    ~/data/semantic_kitti/
    ```
4. Download the [semantic word embeddings](https://drive.google.com/file/d/11MMrgWP7OEET8W5GtRYOwKZQ6ihTQp7q/view?usp=sharing) and the [pretrained backbones](https://drive.google.com/file/d/1WyLGAYvUSGnYx0DtRZNozThFqWL7Jgi0/view?usp=sharing).
    - Place the **semantic word embeddings** in 
    ```
    3DGenZ/genz3d/word_representations/
    ```
    
    - For **SN**, the pre-trained backbone model and the config file, are placed in 
    ```
    3DGenZ/genz3d/fkaconv/examples/scannet/FKAConv_scannet_ZSL4
    ```
    The complete ZSL-trained model cpkt is placed in (create the folder if necessary)
    ```
    3DGenZ/genz3d/seg/run/scannet/
    ```
    
    - For **SK**, the pre-trained backbone-model,  the "Log-..." folder is placed in 
     ```
    3DGenZ/genz3d/kpconv/results
    ```
    And the complete ZSL-trained model ckpt is placed in 
    ```
    3DGenZ/genz3d/seg/run/sk
    ```

### Run training and evalutation
5. **Training (Classifier layer)**: In [3DGenZ/genz3d/seg/](3DGenz/genz3d/seg/) you find for each of the datasets a folder with scripts to run the generator and classificator training.(see: [SN](3DGenZ/genz3d/seg/scripts_sn),[SK](3DGenZ/genz3d/seg/scripts_sn))
    - Alternatively, you can use the pretrained models from us. 
6. **Evalutation:** Is done with the evaluation functions of the backbones. (see: [SN_eval](scannet/LightConvPoint/examples/scannet/scripts_final_eval), [KP-Conv_eval](https://github.com/HuguesTHOMAS/KPConv-PyTorch/blob/master/test_models.py))

### Backbones 
For the datasets we used  different backbones, for which we highly rely on their code basis. In order to adapt them to the ZSL setting we made the change that during the backbone training no crops of point clouds with unseen classes are shown (if there is a single unseen class

- [ConvPoint](https://github.com/aboulch/ConvPoint) [1] for the S3DIS dataset (and also partly used for the ScanNet dataset).
- [FKAConv](https://github.com/valeoai/FKAConv) [2] for the ScanNet dataset.
- [KPConv](https://github.com/HuguesTHOMAS/KPConv-PyTorch) [3] for the SemanticKITTI dataset. 

### Datasets
For semantic segmentation we did experiments on 3 datasets. 
- SemanticKITTI [4][5]. 
- S3DIS [6]. 
- ScanNet[7]. (see the [description](https://github.com/charlesq34/pointnet2/tree/master/scannet) here, to download exactly the same dataset that we used)

# Acknowledgements
For the Generator Training we use parts of the code basis of [ZS3](https://github.com/valeoai/ZS3).\
For the backbones we use the code of [ConvPoint](https://github.com/aboulch/ConvPoint), [FKAConv](https://github.com/valeoai/FKAConv) and [KPConv](https://github.com/HuguesTHOMAS/KPConv-PyTorch). 

# References
[1] Boulch, A. (2020). ConvPoint: Continuous convolutions for point cloud processing. Computers & Graphics, 88, 24-34.\
[2] Boulch, A., Puy, G., & Marlet, R. (2020). FKAConv: Feature-kernel alignment for point cloud convolution. In Proceedings of the Asian Conference on Computer Vision.\
[3] Thomas, H., Qi, C. R., Deschaud, J. E., Marcotegui, B., Goulette, F., & Guibas, L. J. (2019). Kpconv: Flexible and deformable convolution for point clouds. In Proceedings of the IEEE/CVF International Conference on Computer Vision (pp. 6411-6420).\
[4] Behley, J., Garbade, M., Milioto, A., Quenzel, J., Behnke, S., Stachniss, C., & Gall, J. (2019). Semantickitti: A dataset for semantic scene understanding of lidar sequences. In Proceedings of the IEEE/CVF International Conference on Computer Vision (pp. 9297-9307).\
[5] Geiger, A., Lenz, P., & Urtasun, R. (2012, June). Are we ready for autonomous driving? the kitti vision benchmark suite. In 2012 IEEE conference on computer vision and pattern recognition (pp. 3354-3361). IEEE.\
[6] Armeni, I., Sener, O., Zamir, A. R., Jiang, H., Brilakis, I., Fischer, M., & Savarese, S. (2016). 3d semantic parsing of large-scale indoor spaces. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (pp. 1534-1543).\
[7] Dai, A., Chang, A. X., Savva, M., Halber, M., Funkhouser, T., & Nießner, M. (2017). Scannet: Richly-annotated 3d reconstructions of indoor scenes. In Proceedings of the IEEE conference on computer vision and pattern recognition (pp. 5828-5839).

# Updates
9.12.2021 Initial Code release

# Licence
3DGenZ is released under the [Apache 2.0 license](LICENCE).

The folder 3DGenZ/genz3d/kpconv includes large parts of code taken from [KP-Conv](https://github.com/HuguesTHOMAS/KPConv-PyTorch) and is therefore distributed under the MIT Licence. See the [LICENSE](3DGenZ/genz3d/kpconv/LICENSE) for this folder. 

The folder 3DGenZ/genz3d/seg/utils also includes files taken from https://github.com/jfzhang95/pytorch-deeplab-xception and is therefore also distributed under the MIT License. See the [LICENSE](3DGenZ/genz3d/seg/utils/LICENSE) for these files. 
