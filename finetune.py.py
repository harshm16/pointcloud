# Paper Title : Finding Intermediate Generators using Forward Iterates and Applications
# Paper ID: 2177
# Code base inherited from https://github.com/luost26/score-denoise

import os
import argparse
import torch
import torch.utils.tensorboard
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_
from tqdm.auto import tqdm

from datasets import *
from utils.misc import *
from utils.transforms import *
from utils.denoise import *
from models.denoise import *
from models.utils import chamfer_distance_unit_sphere
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import proj3d
import swap_clean_point as swap_clean

# Arguments
parser = argparse.ArgumentParser()
## Dataset and loader
parser.add_argument('--dataset_root', type=str, default='./data')
parser.add_argument('--dataset', type=str, default='PUNet')
parser.add_argument('--patch_size', type=int, default=1000)
parser.add_argument('--resolutions', type=str_list, default=['10000_poisson', '30000_poisson', '50000_poisson'])
parser.add_argument('--noise_min', type=float, default=0.005)
parser.add_argument('--noise_max', type=float, default=0.020)
parser.add_argument('--train_batch_size', type=int, default=32)
# parser.add_argument('--val_batch_size', type=int, default=64)
parser.add_argument('--num_workers', type=int, default=4)
parser.add_argument('--aug_rotate', type=eval, default=True, choices=[True, False])
## Model architecture
parser.add_argument('--supervised', type=eval, default=True, choices=[True, False])
parser.add_argument('--frame_knn', type=int, default=32)
parser.add_argument('--num_train_points', type=int, default=128)
parser.add_argument('--num_clean_nbs', type=int, default=4, help='For supervised training.')
parser.add_argument('--num_selfsup_nbs', type=int, default=8, help='For self-supervised training.')
parser.add_argument('--dsm_sigma', type=float, default=0.01)
parser.add_argument('--score_net_hidden_dim', type=int, default=128)
parser.add_argument('--score_net_num_blocks', type=int, default=4)
## Optimizer and scheduler
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--weight_decay', type=float, default=0)
parser.add_argument('--max_grad_norm', type=float, default=float("inf"))
## Training
parser.add_argument('--seed', type=int, default=2020)
parser.add_argument('--logging', type=eval, default=True, choices=[True, False])
parser.add_argument('--log_root', type=str, default='./logs')
parser.add_argument('--device', type=str, default='cuda')
parser.add_argument('--max_iters', type=int, default=50000)

# parser.add_argument('--max_iters', type=int, default=1*MILLION)
parser.add_argument('--val_freq', type=int, default=2000)
parser.add_argument('--val_upsample_rate', type=int, default=4)
parser.add_argument('--val_num_visualize', type=int, default=4)
parser.add_argument('--val_noise', type=float, default=0.015)
parser.add_argument('--ld_step_size', type=float, default=0.2)
parser.add_argument('--tag', type=str, default=None)
# parser.add_argument('--flag', type=int, default=2)

args = parser.parse_args()
seed_all(args.seed)

# Logging
if args.logging:
    log_dir = get_new_log_dir(args.log_root, prefix='D%s_' % (args.dataset), postfix='_' + args.tag if args.tag is not None else '')
    logger = get_logger('train', log_dir)
    writer = torch.utils.tensorboard.SummaryWriter(log_dir)
    ckpt_mgr = CheckpointManager(log_dir)
    log_hyperparams(writer, log_dir, args)
else:
    logger = get_logger('train', None)
    writer = BlackHole()
    ckpt_mgr = BlackHole()
logger.info(args)

# Datasets and loaders
logger.info('Loading datasets')
train_dset = PairedPatchDataset(
    datasets=[
        PointCloudDataset(
            root=args.dataset_root,
            dataset=args.dataset,
            split='train',
            resolution=resl,
            transform=standard_train_transforms(noise_std_max=args.noise_max, noise_std_min=args.noise_min, rotate=args.aug_rotate)
        ) for resl in args.resolutions
    ],
    patch_size=args.patch_size,
    patch_ratio=1.2,
    on_the_fly=True  
)
val_dset = PointCloudDataset(
        root=args.dataset_root,
        dataset=args.dataset,
        split='test',
        resolution=args.resolutions[0],
        transform=standard_train_transforms(noise_std_max=args.val_noise, noise_std_min=args.val_noise, rotate=False, scale_d=0),
    )
train_iter = get_data_iterator(DataLoader(train_dset, batch_size=args.train_batch_size, num_workers=args.num_workers, shuffle=True))

# Model
logger.info('Building model...')



ckpt_pre_trained = torch.load('/pretrained/ckpt.pt', map_location=args.device)
model = DenoiseNet(ckpt_pre_trained['args']).to(args.device)



pre_model_convp_w = ckpt_pre_trained['state_dict']['score_net.conv_p.weight']
pre_model_convp_b = ckpt_pre_trained['state_dict']['score_net.conv_p.bias']

pre_model_convout_w = ckpt_pre_trained['state_dict']['score_net.conv_out.weight']
pre_model_convout_b = ckpt_pre_trained['state_dict']['score_net.conv_out.bias']


ckpt_pre_trained['state_dict']['score_net.conv_p.weight'] = pre_model_convp_w / 2
ckpt_pre_trained['state_dict']['score_net.conv_p.bias'] = pre_model_convp_b / 2

ckpt_pre_trained['state_dict']['score_net.etau.weight'] = pre_model_convp_w / 2
ckpt_pre_trained['state_dict']['score_net.etau.bias'] = pre_model_convp_b / 2

ckpt_pre_trained['state_dict']['score_net.conv_out.weight'] = pre_model_convout_w / 2
ckpt_pre_trained['state_dict']['score_net.conv_out.bias'] = pre_model_convout_b / 2

ckpt_pre_trained['state_dict']['score_net.dtau.weight'] = pre_model_convout_w / 2
ckpt_pre_trained['state_dict']['score_net.dtau.bias'] = pre_model_convout_b / 2



model.load_state_dict(ckpt_pre_trained['state_dict'])



logger.info(repr(model))

# Optimizer and scheduler
optimizer = torch.optim.Adam(model.parameters(),
    lr=args.lr,
    weight_decay=args.weight_decay,
)


# Train, validate and test
def train(it):
    # Load data
    batch = next(train_iter)

    pcl_noisy = batch['pcl_noisy'].to(args.device)
    pcl_clean = batch['pcl_clean'].to(args.device)

    #Hyperparameter - Decides the corruption rate. Value between: 0 - 1000
    beta = 500
    pcl_average = swap_clean.replace_iterator(pcl_noisy,pcl_clean,beta)


    if it % 1000 == 0:

        print("iteration :", it)

        cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
        pdist = torch.nn.PairwiseDistance(p=2)


        print("Cos similarity conv_p.weight - etau.weight",cos(model.state_dict()['score_net.conv_p.weight'], model.state_dict()['score_net.etau.weight']).mean())
        
        print("Cos similarity conv_out.weight - dtau.weight",cos(model.state_dict()['score_net.conv_out.weight'], model.state_dict()['score_net.dtau.weight']).mean())
       
        print("Distance bw p and etau",pdist(model.state_dict()['score_net.conv_p.weight'],model.state_dict()['score_net.etau.weight']).mean())

        print("Distance bw out and dtau",pdist(model.state_dict()['score_net.conv_out.weight'],model.state_dict()['score_net.dtau.weight']).mean())


        # print("end")


    
    # Reset grad and model state
    optimizer.zero_grad()
    model.train()

    #Hyperparameter 
    alpha = 0.8
    

    # Forward
    # 1 - noisy pcl
    # 2 - Intermediate iterate
    if args.supervised:
        loss1 = model.get_supervised_loss(1, pcl_noisy=pcl_noisy, pcl_clean=pcl_clean)
        loss2 = model.get_supervised_loss(2, pcl_noisy=pcl_average, pcl_clean=pcl_clean)
        loss = (alpha)*loss1 + (1-alpha)*loss2
    else:
        # loss1 = model.get_selfsupervised_loss(pcl_noisy=pcl_noisy)
        # loss = model.get_selfsupervised_loss(pcl_noisy=pcl_average)
        # loss = (alpha)*loss1 + (1-alpha)*loss2
        print("self supervised")


    loss.backward()

    orig_grad_norm = clip_grad_norm_(model.parameters(), args.max_grad_norm)
    optimizer.step()

    # Logging
    logger.info('[Train] Iter %04d | Loss %.6f | Grad %.6f' % (
        it, loss.item(), orig_grad_norm,
    ))
    
    # storing loss

    writer.add_scalar('train/loss', loss, it)
    writer.add_scalar('train/lr', optimizer.param_groups[0]['lr'], it)
    writer.add_scalar('train/grad_norm', orig_grad_norm, it)
    writer.flush()
    return loss 

def validate(it):
    all_clean = []
    all_denoised = []
    all_denoised_avg = []
    
    #Hyperparameter
    alpha = 0.8

    for i, data in enumerate(tqdm(val_dset, desc='Validate')):
        pcl_noisy = data['pcl_noisy'].to(args.device)
        pcl_clean = data['pcl_clean'].to(args.device)

        
        pcl_average = swap_clean.replace_points(pcl_noisy,pcl_clean,500)
        pcl_average = pcl_average.to(args.device)


        pcl_denoised = patch_based_denoise(model,1, pcl_noisy, ld_step_size=args.ld_step_size)

        
        pcl_denoised_avg = patch_based_denoise(model,2, pcl_average, ld_step_size=args.ld_step_size)

        # print(pcl_denoised_avg.shape)
        all_clean.append(pcl_clean.unsqueeze(0))
        all_denoised.append(pcl_denoised.unsqueeze(0))
        all_denoised_avg.append(pcl_denoised_avg.unsqueeze(0))

    all_clean = torch.cat(all_clean, dim=0)
    all_denoised = torch.cat(all_denoised, dim=0)
    all_denoised_avg = torch.cat(all_denoised_avg, dim=0)

    avg_chamfer = chamfer_distance_unit_sphere(all_denoised, all_clean, batch_reduction='mean')[0].item()

    avg_chamfer_avg = chamfer_distance_unit_sphere(all_denoised_avg, all_clean, batch_reduction='mean')[0].item()

    combined_avg_chamfer = (alpha)*avg_chamfer + (1-alpha)*avg_chamfer_avg

    logger.info('[Val] Iter %04d | CD %.6f  ' % (it, combined_avg_chamfer))
    writer.add_scalar('val/chamfer', combined_avg_chamfer, it)
    writer.add_mesh('val/pcl', all_denoised[:args.val_num_visualize], global_step=it)
    writer.flush()

    # scheduler.step(avg_chamfer)
    return combined_avg_chamfer

# # Main loop
# logger.info('Start training...')
max_iters = 2000
PATH = 'model_update_pre_trained_model.pt'
chkp = {}


try:
    for it in range(1, args.max_iters+1):

        train(it)
        
        if it % args.val_freq == 0 or it == args.max_iters:
            cd_loss = validate(it)
            opt_states = {
                'optimizer': optimizer.state_dict(),
                # 'scheduler': scheduler.state_dict(),
            }
            ckpt_mgr.save(model, args, cd_loss, opt_states, step=it)
            # ckpt_mgr.save(model, args, 0, opt_states, step=it)
    
    torch.save(chkp, PATH)
except KeyboardInterrupt:
    logger.info('Terminating...')
