# Learning 3D Representation of Germinated Parts for View-Angle-Robust Classification

Acknowledgement: Code is borrowed from MeshInversion (thus, from ConvMesh) and Textured3DGAN.

Original repo:
* MeshInversion: [link] (https://github.com/junzhezhang/mesh-inversion)
* Textured3DGAN: [link] (https://github.com/dariopavllo/textured-3d-gan)
* ConvMesh: [link] (https://github.com/dariopavllo/convmesh)

This project attempts to perform 3D reconstruction on germinated oil palm seeds.

The README file of individual projects were also saved for more information about running the individual projects in different settings. For now the following shows how to run perform reconstruction on the AAR dataset.

## Dependencies
Required libraries:

- [Kaolin](https://github.com/NVIDIAGameWorks/kaolin) (tested on commit [e7e5131](https://github.com/NVIDIAGameWorks/kaolin/tree/e7e513173bd4159ae45be6b3e156a3ad156a3eb9)).
	(This version only works on Linux OS, not on Window OS).
- Python >= 3.6 (Tested on Python 3.7)
- PyTorch >= 1.6 (Tested 0n 1.13.1)
- CUDA >= 10.0 (tried running on without using containers in HPC that uses CUDA version 11.2, but failed; instead works in a singularity image on HPC with CUDA 11.6)
- Misc dependencies (installable using pip): `packaging`, `tensorboard` (optional).
- `requirements.txt` includes all packages that were installed except for Kaolin.

Setup if running on HPC:
The project was done in a singularity container due to the some bugs in the current CUDA runtime (version 11.2) installed in the HPC that prevents loading the model to the GPU and running it properly.
After using slurm to request GPU resources and starting a job, run the Singularity container `apps/images/pytorch-22.02-py3.sif` (assumed to be available to all users of the HPC, should be able to run properly due to the image using CUDA 11.6).
`conda` should be included in the image, use it to create a virtual environemnt with Python 3.7.
```
conda create --prefix path/to/env/env_name python=3.7
conda activate path/to/env/env_name
```
After activating the virtual environemnt, install the packages listed in `requirements.txt`, and then install Kaolin v0.1.
Install requirements:
```
pip install -r requirements.txt
```
Kaolin install (after setting up a virtual environemnt and running the installing the required libraries):
```
git clone https://github.com/NVIDIAGameWorks/kaolin --branch v0.1 --single-branch
cd kaolin
python setup.py install
```

## Dataset and pretrained setup
All are already set up. List is included just in case procedure is repeated, for copying the outputs to the right directory for the next step.
### Textured3DGAN
Mesh template:
- Pose estimation: `textured-3d-gan/mesh_templates/classes/batch1seed_train1.obj`
- Deformable mesh: `textured-3d-gan/mesh_templates/uvsphere_31rings.obj`

Dataset: (batch1seed is seedsegment)
- `textured-3d-gan/datasets/batch1ssed/test`
- `textured-3d-gan/datasets/batch1seed/train`

Trained mesh estimation model (for generating pseudo-ground truth data):
- `textured-3d-gan/checkpoints_recon/new_batch1seed_train_singletpl/checkpoint_latest.pth`

Cache:
- Remeshed mesh template
	- `textured-3d-gan/cache/remeshed_templates/singletpl/batch1seed_templates.obj`
	- `textured-3d-gan/cache/remeshed_templates/singletpl/batch1seed_templates.pth`
- Training set
	- Segmentations
		- `textured-3d-gan/cache/batch1seed_train/detections.npy`
	- Silhouette-only pose estimation
		- `textured-3d-gan/cache/batch1seed_train/camera_hypotheses_silhouette_singletpl.bin`
	- Semantic pose estimation:
		- `textured-3d-gan/cache/batch1seed_train/poses_estimated_singletpl.bin`
- Testing set:
	- Segmentations:
		- `textured-3d-gan/cache/batch1seed_test/detections.npy`
	- Silhouette-only pose estimation
		- `textured-3d-gan/cache/batch1seed_test/camera_hypotheses_silhouette_singletpl.bin`
		


### MeshInversion
Dataset:

Training set
- Dataset directory: (Copy both folder: BadSeed and GoodSeed from `AAR/seedsegment/train`)
	- `mesh-inversion/datasets/batch1seed_train/images/`
- Labels:
	- `mesh-inversion/datasets/batch1seed_train/batch1seed/image_class_labels.txt`
- Image and path list:
	- `mesh-inversion/datasets/batch1seed_train/batch1seed/images.txt`
- Cache:
	- Pre-calculated saved random camera poses (for image space discrimination)
	(not mentioned in paper how it was obtained, however with comments from the code saying to be random poses; assumed to be pre-computed to allow reproducibility of results)
		- `mesh-inversion/datasets/batch1seed_train/cache/cam_pose_train.pickle`	
	- Optimised camera poses during mesh estimation (same as the one from Textured3DGAN; which is `cache/batch1seed_train/poses_estimated_singletpl.bin`):
		- `mesh-inversion/datasets/batch1seed_train/cache/poses_metadata.npz`
	- Precomputerd FID (same as `cache/batch1seed_train/precomputed_fid_299x299.npz`):
		- `mesh-inversion/datasets/batch1seed_train/cache/precomputed_fid_299x299_train.npz`
	- Pseudo-ground truth data:
		- `mesh-inversion/datasets/batch1seed_train/pseudogt_512x512`
 
Test set
- Dataset directory: (Copy both folder: BadSeed and GoodSeed from AAR/seedsegment/test)
	- `mesh-inversion/datasets/batch1seed_test/batch1seed/images/`
- Labels:
	- `mesh-inversion/datasets/batch1seed_test/batch1seed/image_class_labels.txt`
- Image path list:
	- `mesh-inversion/datasets/batch1seed_test/batch1seed/images.txt`
- Cache:
	- Segmentations (obtained after running segmentation on the test set):
		- `mesh-inversion/datasets/batch1seed_test/cache/detections.npy`
	- Estimated camera poses:
		- `mesh-inversion/datasets/batch1seed_test/cache/poses_estimated_singletpl.bin`
	- Precomputed FID:
		-`mesh-inversion/datasets/batch1seed_test/cache/precomputed_fid_299x299_testval.npz`
	- Pre-trained GAN:
		- `mesh-inversion/checkpoints_gan/unconditional_batch1seed_train/checkpoint_latest.pth`

## Training
Segmentation
### Step 1 Image and semantic segmentation
Change `dataset_path` to the path of the dataset.
Change `image_path` to `train` to perform segmentation on the training set.
Change `save_path` to the saving path of the detections.
For performing pose estimation, copy it to the corresponding locations as described in dataset setup.



In textured-3d-gan:
- Step 0 Remeshing the mesh template for pose estimation 
```
python remesh.py --mode singletpl --gpu_ids 0 --classes batch1seed_train

```
(a pre-mesh one is already supplied, so this step is not required unless you wish to repeat the procedure)

- Step 2 (Pose estimation step 1) Silhouette-only pose estimation
```
python pose_optimization_step1.py --dataset batch1seed_train --mode singletpl --gpu_ids 0
```

- Step 3 (Pose estimation step 2) Semantic-based pose estimation
```
python pose_optimization_step2.py --dataset batch1seed_train --mode singletpl --gpu_ids 0
```

- Step 4 Train mesh estimation model
```
python run_reconstruction.py --name new_batch1seed_train_singletpl --mode singletpl --dataset batch1seed_train --gpu_ids 0 --iters 130000 --tensorboard
```

- Step 5 Generate pseudo-ground truth
```
python run_reconstruction.py --name new_batch1seed_train_singletpl --dataset batch1seed_train --batch_size 10 --generate_pseudogt --num_workers 1

```
This should output a directory of pseudo-ground truths in `textured-3d-gan/cache/batch1seed_train/pseudogot_512x512_singletpl``
Copy to `mesh-inversion/datasets/batch1seed_train` and rename it to `pseudogt_512_512`.

In mesh-inversion:
- Step 6 Train GAN
```
python run_pretraining.py --name unconditional_batch1seed_train --data_dir ./datasets/batch1seed_train --dataset batch1seed_train --gpu_ids 0 --epochs 600

```

## Testing/Inversion
- Step 1 Segment testing dataset
Change `dataset_path` to the path of the dataset.

Change `image_path` to `train` to perform segmentation on the training set.

Change `save_path` to the saving path of the detections.

For performing pose estimation/GAN inversion, copy it to the corresponding locations as described in dataset setup, which is:
`textured-3d-gan/cache/batch1seed_test/detections.npy`
`mesh-inversion/datasets/batch1seed_test/cache/detections.npy`

In textured-3d-gan:
- Step 2 (Pose estimation step 1) Silhouette-only pose estimation
```
python pose_optimization_step1.py --dataset batch1seed_test --mode singletpl --gpu_ids 0
```

- Step 3 (Pose estimation step 2) Semantic-based pose estimation
```
python pose_optimization_step2.py --dataset batch1seed_test --mode singletpl --gpu_ids 0
```
The resulting file `textured-3d-gan/cache/batch1seed_test/poses_estimated_singletpl.bin` should be copied to `mesh-inversion/datasets/batch1seed_test/cache/poses_estimated_singletpl.bin`

In mesh-inversion:
- Step 4 GAN Inversion
```
python run_inversion.py --name batch1seed_test_out --checkpoint_dir unconditional_batch1seed_train --data_dir ./datasets/batch1seed_test --dataset batch1seed_test 
```

## Evaluation
In mesh inversion:
- Calculate IoU:
```
python run_evaluation.py --name batch1seed_test_out --data_dir ./datasets/batch1seed_test --dataset batch1seed_test --checkpoint_dir unconditional_batch1seed_train --eval_option IoU
```
This will calculate the mean IoU of the reconstruction and the target image using the optimised camera poses.

- Calculate FID 1:

```
python run_evaluation.py --name batch1seed_test_out --data_dir ./datasets/batch1seed_test --dataset batch1seed_test --checkpoint_dir unconditional_batch1seed_train --eval_option FID_1
```
This will calculate the FID of the reconstruction rendered using the optimised camera pose. A .txt file detailing which [id].pth file corresponds to which image path. 
The FID of the testing set is also computed and saved.

- Calculate FID 12, render from 12 views and save the mesh as .obj file:
```
python run_evaluation.py --name batch1seed_test_out --data_dir ./datasets/batch1seed_test --dataset batch1seed_test --checkpoint_dir unconditional_batch1seed_train --eval_option FID_12
```
Each reconstruction is rendered at 12 different view, which covers azimuth from 0◦ to 360◦ at an interval of 30◦, and saved. The FID is computed between the sets of 12 renderings and the testing set.

- Calculate FID 10:
```
python run_evaluation.py --name batch1seed_test_out --data_dir ./datasets/batch1seed_test --dataset batch1seed_test --checkpoint_dir unconditional_batch1seed_train --eval_option FID_10
```
FID 10 reports the exact front view (90◦) and exact back view (270◦), which correponds to the occluded surfaces.

## Outputs
Inversion results:
`mesh-inversion/outputs/inversion_results/batch1seed_test_out`
(Note: Job was somehow automatically killed, resulting on 384 out of 401 of the testing set being reconstructed.)

Mesh of the recosntructions (in .obj file, refer to `mesh-inversion/datasets/batch1seed_test/cache/inversion_path.txt` for list of mesh corresponding to which testing image)
`mesh-inversion/outputs/mesh/mesh/batch1seed_test`

Renderings of each reconstruction from 12 different viewpoints:
`mesh-inversion/outputs/multiview_renderings_12/batch1seed_test_out`
