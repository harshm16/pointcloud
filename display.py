# Paper Title : Finding Intermediate Generators using Forward Iterates and Applications
# Paper ID: 2177
#Use to covert any point cloud from xyz file to a png.

import matplotlib.pyplot as plt
import numpy as np
import torch 
def matplotlib_3d_ptcloud(filename,output_png):
    pcl = torch.FloatTensor(np.loadtxt(filename, dtype=np.float32))

    data = [pcl.detach().cpu().numpy()]
    xdata = data[0][:,0].squeeze()
    ydata = data[0][:,1].squeeze()
    zdata = data[0][:,2].squeeze()

    fig = plt.figure(figsize=(15, 15))
    ax = plt.axes(projection='3d')

    ax.scatter3D(xdata, ydata, zdata, marker='o')

    plt.savefig(output_png, bbox_inches='tight')
    # plt.show()
    

filename = "/cow.xyz"

save_file = 'cow.png'

matplotlib_3d_ptcloud(filename,save_file)
