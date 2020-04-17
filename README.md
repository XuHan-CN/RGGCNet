
# RGGCNet: 3D Sematic Segmentation based Debris Recognition of Terrain Point Cloud

## Dataset(LPMT)
If you are interest in our dataset(LPMT), please click http://xuhan-cn.com/LPMT/dataset/HTML/data.html

## Code structure
* `./partition/*` - Partition code (Multi-attribute descriptor extraction, terrain feature division)
* `./learning/*` - Learning and testing code (Feature Extraction Network).


## Disclaimer
Our partition method is inherently stochastic. Hence, even if we provide the trained weights, it is possible that the results that you obtain differ slightly from the ones presented in the paper.

## Requirements 

*1.* Install [PyTorch] and [torchnet]
```
pip install git+https://github.com/pytorch/tnt.git@master
``` 

*2.* Install additional Python packages:
```
pip install future python-igraph tqdm transforms3d pynvrtc fastrlock cupy h5py sklearn plyfile scipy
```

*3.* Install Boost (1.63.0 or newer) and Eigen3, in Conda:
```
conda install -c anaconda boost; conda install -c omnia eigen3; conda install eigen; conda install -c r libiconv
```

*4.* Compile the ```libply_c``` and ```libcp``` libraries:
```
CONDAENV=YOUR_CONDA_ENVIRONMENT_LOCATION
cd partition/ply_c
cmake . -DPYTHON_LIBRARY=$CONDAENV/lib/libpython3.6m.so -DPYTHON_INCLUDE_DIR=$CONDAENV/include/python3.6m -DBOOST_INCLUDEDIR=$CONDAENV/include -DEIGEN3_INCLUDE_DIR=$CONDAENV/include/eigen3
make
cd ..
cd cut-pursuit
mkdir build
cd build
cmake .. -DPYTHON_LIBRARY=$CONDAENV/lib/libpython3.6m.so -DPYTHON_INCLUDE_DIR=$CONDAENV/include/python3.6m -DBOOST_INCLUDEDIR=$CONDAENV/include -DEIGEN3_INCLUDE_DIR=$CONDAENV/include/eigen3
make
```
The code was tested on Ubuntu 18 with Python 3.6 and PyTorch 1.1.

## Running the code

To extract multi-attribute descriptor and compute terrain division run:
```
python partition/partition.py --dataset LPMT --ROOT_PATH dataset --voxel_width 0.03 --reg_strength 0.03
```

To construct the super point into super point graph run:
```
python learning/LPMT_dataset.py --LPMT_PATH dataset
```

To train the network run:
```
CUDA_VISIBLE_DEVICES=0 python learning/main.py --dataset LPMT --LPMT_PATH dataset --epochs 260 --lr_steps '[30,60,150,200]' --test_nth_epoch 1 --ptn_nfeat_stn 14 --nworkers 2 --pc_attribs xyzrgbelpsvXYZ --odir "results/" --nworkers 4;
```

To test the network run:
```
CUDA_VISIBLE_DEVICES=0 python learning/main.py --dataset LPMT --LPMT_PATH dataset --epochs -1 --lr_steps '[30,60,150,200]' --test_nth_epoch 1  --ptn_nfeat_stn 14 --nworkers 2 --pc_attribs xyzrgbelpsvXYZ --odir "results/" --resume RESUME
```

To evaluate the trained model, run:
```
python learning/evaluate_LPMT.py --odir results
```
To obtain visible result, run(For example):
```
python partition/visualize.py --dataset LPMT --ROOT_PATH dataset --res_file results/predictions_test --file_path LPMT/1 --output_type igprs
```

```output_type``` defined as such:
- ```'i'``` = input rgb point cloud
- ```'g'``` = ground truth (if available), with the predefined class to color mapping
- ```'p'``` = partition, with a random color for each superpoint
- ```'r'``` = result cloud, with the predefined class to color mapping
- ```'s'``` = superedge structure of the superpoint 
