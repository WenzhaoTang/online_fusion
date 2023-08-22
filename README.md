# 3DSL-SS22_OnlineFusion (Semester project at TUM)
## Project Summary

Based on a framework named NeuralRecon for real-time 3D scene reconstruction from a monocular video, we experimented with the following idea:

- Introduce different feature aggregation mechanisms to aggregate the back-projected image features. To this end, we involve two different methods, the Transformer based feature aggregation, and the Attentional aggregation, in comparison with the simplest summation method.
- Introducing a differential render, use it to render depth map of the reconstructed geometry to calculate the 2D depth loss for supervised training.
- Using the differential render to render depth map of the reconstructed geometry, calculating the robust photometric consistency loss with the neighboring input RGB image for test time optimization.

## NeuralRecon Documentation

The documentation and code repository for [NeuralRecon](https://github.com/zju3dv/NeuralRecon) are listed here for reference.

## Installation

```
# Ubuntu 18.04 and above is recommended.
sudo apt install libsparsehash-dev  
conda env create -f environment.yaml
conda activate neucon
```

Note that there is a [known issue](https://github.com/zju3dv/NeuralRecon/issues/78) with the compatibility between torchsparse(version 1.4.0) and pytorch(version 1.6.0). There is a workaround to modify the local torch sparse library source code:
```
# Modify the line 22 of /home/<USERNAME>/<CONDA PATH>/envs/neucon/lib/<PYTHON VERSION>/site-packages/torchsparse/nn/functional/downsample.py
# Change the dtype from torch.int to torch.float32
sample_stride = torch.tensor(sample_stride,
                                 dtype=torch.float32,
                                 device=coords.device).unsqueeze(dim=0)
```

For compiling the extension modules of differentiable renderer, run the `install_utils_renderer.sh` script:

```commandline
bash tools/install_utils_renderer.sh
```

## Data Preprocessing for Scannet
Download and extract ScanNet by following the instructions provided at http://www.scan-net.org/.

Then run the data preparation script to generate the ground truth TSDFs:
```commandline
python tools/tsdf_fusion/generate_gt_modified.py --data_path PATH_TO_SCANNET
```
Note that if you want to start a multi-processing procedure，set `--start_idx` and `--end_idx` for clipping the list.

## Pretrained Model on Scannet

The pretrained models with different feature aggregation methods could be downloaded from [here](https://drive.google.com/drive/folders/1q39cmg_DGuM25q9RY50JUg6mhg2Pypg2?usp=sharing).

## Trainning

### Training Phases

Same as NeuralRecon, the training is separated into two phases, which need to be switched manually.
- Phase 1 (the first 0-20 epoch), training single fragments:
```yaml
# in ./config/train.yaml, change following settings
MODEL.FUSION.FUSION_ON=False
MODEL.FUSION.FULL=False
```
- Phase 2 (the remaining 21-50 epoch), turn on GRUFusion to make the reconstruction consistent between local fragments.
```yaml
# in ./config/train.yaml, change following settings
MODEL.FUSION.FUSION_ON=True
MODEL.FUSION.FULL=True
```

### Switch feature aggregation methods
By changing the settings in train.yaml, you can switch between different feature aggregation methods:
- Summation (NeuralRecon)
```yaml
# in ./config/train.yaml, change following settings
MODEL.FP16_ON = False
MODEL.AGGREGATION_ON = False
```
- Attentional Aggregation
```yaml
# in ./config/train.yaml, change following settings
MODEL.FP16_ON = False
MODEL.AGGREGATION_ON = True
MODEL.AGGREGATION.MODE = 'attentional'
```
- Transformer Based Aggregation
```yaml
# in ./config/train.yaml, change following settings
# Turn on half precision to avoid Out of Memory issue
MODEL.FP16_ON = True
MODEL.AGGREGATION_ON = True
MODEL.AGGREGATION.MODE = 'transformer'
```

### Start Training

After determining the train.yaml, you could start training by running the script:
```commandline
bash train.sh
```
The train bash is as following. If you want to run multiple experiments in distributed manner, you could use an additional parameter --masterport.
```
#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=0,1
python -m torch.distributed.launch --nproc_per_node=2 main.py --cfg ./config/train.yaml
```

## Testing
```commandline
python main.py --cfg ./config/test.yaml
```
The generated meshes will be stored in ./results. Note that in the testing procedure, all models in the path LOGDIR that you defined in test.yaml will be used for inference. Thus, for the first 20 epochs without GRUFusion, you can't run with test.yaml that turns on the GRUFusion. You could also perform the test in two different phases.

## Evaluation

```commandline
python tools/evaluation.py --model ./results/scene_scannet_LOGDIR_fusion_eval_EPOCH
```
Note that evaluation.py uses pyrender to render depth maps from the predicted mesh for 2D evaluation. If you are using headless rendering you must also set the enviroment variable PYOPENGL_PLATFORM=osmesa (see pyrender for more details).

Some extra important parameters for evalution

- --data_path: Path to dataset.
- --gt_path: Path to raw dataset containing ground truth.
- --n_proc: Processes launched to process scenes.

### Print evaluations
You can print the results of a previous evaluation run using
```commandline
python tools/visualize_metrics.py --model ./results/scene_scannet_LOGDIR_fusion_eval_EPOCH
```

## Implementation Details
### Feature Aggregation
#### Aggregation Configuration
The configuration for aggregation method and aggregation network setting(specifically for transformer) is introduced by adding default config parameters in config/default.py. Users can config the aggregation methods they want to use in train/test.yaml as mentioned above in Training section.

#### Mixed Precision Training
By introducing a GradScaler in main.py, the parameters during training are automatically converted to Half type, which save us a lot of memory. There are still some vairables created with default float32 types, which needs to be treated carefully.

#### Viewing ray direction
We embed the viewing ray direction into the input feature. With the extrinsics provided by the dataset, we calculate the viewing camera center position in datasets/transforms.py and further use it to calculate the viewing ray direction.

#### Transformer-based feature aggregation
The implementation is in models/transformer.py, referencing to Vision Transformer implementation.

#### Attentional feature aggregation
The implementation is in models/attention.py, referencing to the Attset implementation.

### Differentiable Renderer
The differentiable renderer is adapted from [Raycaster Module of SPSG](https://github.com/angeladai/spsg/tree/master/torch/utils/raycast_rgbd). 
Given a TSDF, output depth and camera parameters, we construct the raycaster class. Firstly, we load the camera parameters into the expected format of the raycaster function. We also calculate the World2Grid matrix, this requires a .txt bounds file that contains the 3D bounding vector of the TSDF. Then, we specify the required depth dimensions and output .png depth file name. If the dataset is the processed ScanNet dataset, the default voxel size is set to 0.04 and the default output depth dimension are also set as 480x640. We then proceed to set the parameter for raycasting like thresholding distance, depth min, depth max and ray increment. For now, we have modified the parameters to adjust according to ScanNet dataset. Then, we forward pass into the renderer function to obtain the depth map tensor.
The result can be viewed using
```commandline
python3 differential_renderer.py --intrinsics <intrinsics_path> --pose <pose_path> --tsdf <tsdf_path> --out_depth <out depth path> --outh <out depth height> --outw <out depth width> --voxelsize <the voxel size in TSDF> --bounds <The input 3D bound vector .txt file of the TSDF>
```

### Robust Photometric Loss
The robust photometric loss is adapted from [Inverse Warp of Unsup-MVS](https://github.com/tejaskhot/unsup_mvs). Given a reference keypoint, the corresponding depth map at this reference view and any number of neighboring views, we can calculate the robust photometric loss. The loss calculator expects the path to reference RGB keypoint image and reference depth image. It is also important to specify number of neighboring views (M) in order to construct loss volume 'H x W x M'. Then, we need to specify the input directory for the neighboring RGB images. Along with these parameters, we require the .txt intrinsics file as well as .txt reference pose file and directory with neighboring .txt pose file. The warping procedure is calculated by inverse warping function after which we bilinearly sample to obtain the warped image. The Image is concatenated and we take average along the channel and 'M' dimension to produce a loss gradient to propagate into the Network.  
The result can be viewed using
```commandline
python3 tools/robust_photometric_loss.py --ref_rgb <Reference_Keypoint_Image> --ref_depth <Reference_Viewpoint_Depth_Image> --num_neighbours <Numbers_of_neighbors> --neighbours <Directory_with_neighboring_images> --intrinsics <Intrinsics_File> --ref_pose <Reference_Image_Pose> --neigh_pose <Directory_With_Neighboring_Viewpoint_Poses>
```


# online_fusion
