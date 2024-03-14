 # Retina-inspired Object Motion Sensitivity #10149
## Abstract
 Dynamic Vision Sensors (DVS) have emerged as a revolutionary technology with a high temporal resolution that far surpasses RGB cameras. DVS technology draws biological inspiration from photoreceptors and the initial retinal synapse. Our research showcases the potential of additional retinal functionalities to extract visual features. We provide a domain-agnostic and efficient algorithm for ego-motion compensation based on Object Motion Sensitivity (OMS), one of the multiple robust features computed within the mammalian retina. We develop a framework based on experimental neuroscience that translates OMS' biological circuitry to a low-overhead algorithm. OMS processes DVS data from dynamic scenes to perform pixel-wise object motion segmentation. Using a real and a synthetic dataset, we highlight OMS' ability to differentiate object motion from ego-motion, bypassing the need for deep networks. This paper introduces a bio-inspired computer vision method that dramatically reduces the number of parameters by a factor of 1000 compared to prior works. Our work paves the way for robust, high-speed, and low-bandwidth decision-making for in-sensor computations. 
 
## Folder Structure

- **oms/**: Contains the implementation of Object Motion Sensitivity Algorithm
- **datasets/**: Contains files required for parsing the data from EV-IMO[] and MOD[].
- **main.ipynb**: Main Jupyter notebook for running the algorithm and showcasing results.

## Requirements
```bash
conda install --file requirements.txt
```
## Data
To obtain the filtered events from both datasets in numpy format refer to this [source](https://drive.google.com/drive/folders/1yrHUqYf0rWrfxbQILzKB9_kDYWF6yekd). Maintain the directory structure for proper performance.

## Testing
The following files were taken from [SpikeMS](https://github.com/prgumd/SpikeMS) repository to preprocess the data and evaluate our algorithm under the same conditions.

- **oms/**: Contains the implementation of Object Motion Sensitivity Algorithm
- **evimo_dataset.py**: provides a class to parse the raw events of EV-IMO. *__getitem__* was modified to provide the OMS representation. Works for an specific sequence file.
- **mod_dataset.py**: provides a class to parse the raw events of MOD. *__getitem__* was modified to provide the OMS representation. Works for an specific sequence file. 
- **runner.py**: script to iterate through the validation sequences and compute the mean Intersection over Union and Detection Rate. We removed/comment all the code corresponding to their model.
- **slayerpytorch/** Modified version of SLAYER. src/loss.py calculates loss functions.

## Usage
Search for "TODO" comments in **main.ipynb** to modify the paths of the sequence files and test the algorithm with different sequences. 

### This respository will be made available upon acceptance of the manuscript. 