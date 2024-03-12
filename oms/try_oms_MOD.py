import h5py 
import numpy as np
with h5py.File('/home/mavi/iris/version_2/experiments/experiment_3_vsim/results/room/seq_room1_obj1/seq_room1_obj1_neuroscience.h5', 'r') as f:
    data = f['oms_frames'][:]
masks = np.load('/home/shared/MOD/masks.npy')
frames = np.load('/home/shared/MOD/frames.npy')
print(masks.shape, data.shape)
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman"]
## subplots for oms and masks
fig, axes = plt.subplots(1, 2)
fig.patch.set_visible(False)
axes[0].axis('off')
axes[1].axis('off')
axes[0].imshow(frames[30], cmap = 'gray_r')
axes[0].set_title('MOD events')
axes[0].set_xticks([])
axes[0].set_yticks([])
#axes[1].imshow(data[30])
#axes[1].set_title('OMS')
mask_pic = frames[30]
mask_pic[masks[30]>0] = 1
axes[1].imshow(mask_pic, cmap = 'gray_r')
axes[1].set_title('MOD events masked with GT')
axes[1].set_xticks([])
axes[1].set_yticks([])
plt.savefig('./MOD_test.png', bbox_inches = 'tight', dpi = 300)