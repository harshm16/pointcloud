# Paper Title : Finding Intermediate Generators using Forward Iterates and Applications
# Paper ID: 2177

import matplotlib.pyplot as plt
import numpy as np
import random
import torch

def delete_random_elems(input_list, n):
    to_delete = set(random.sample(range(len(input_list)), n))
    return [x for i,x in enumerate(input_list) if not i in to_delete]

def replace_points(pcl_1,pcl_2, alpha):

    clean_random_sample = random.sample(list(pcl_2.cpu().detach().numpy()),alpha)

    chopped_noisy = delete_random_elems(list(pcl_1.cpu().detach().numpy()),alpha)

    new_data = torch.zeros_like(pcl_1)
    
    new_data = clean_random_sample + chopped_noisy

    return torch.Tensor(new_data)

def replace_iterator(pcl_1,pcl_2, alpha):

    copy_noisy = torch.zeros_like(pcl_1)

    for n in range(0,pcl_1.shape[0]):
        copy_noisy[n] = replace_points(pcl_1[n],pcl_2[n],alpha)
    
    return copy_noisy