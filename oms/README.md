# Experiment 1 Vsim

## Parameters
Default Parameters: 
- Video 
  - Width: 240 
  - Height: 180 
  - Fps: 120
- Center_kernel_radius = 3 (arbitrary for now)
- Surround_kernel_radius =  5 (arbitrary for now)
- OMS threshold = .4 (arbitrary for now)


## Procedure
1. Optimize each process using BO comparing with one data file
2. Iterate over types of filters
   - Gaussian (neuroscience)
   - Hardware (Square ones)
   - Hardware (circle ones)

If time include CNN based kernel GKM

For each filter:
Run OMS Algorithm on the the v2e DVS data


## Metrics

SSIM 
Sparsity (bit_rate?) (spikes/frame)
Feature Similarity Index Method?

## Plots

- Fig 1. Pictures of the different kernels
- Fig 2. Table of spikes in each method vs. ground truth (outline video) (tally the num of ones)


