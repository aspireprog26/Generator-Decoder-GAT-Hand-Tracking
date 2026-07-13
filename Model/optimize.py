import torch
import numpy as np
import torch.nn.functional as F
from keypointdetection import HAND_SKELETON, ANGLE_JOINTS

class Losses:
    def __init__(self):
        self.bones = HAND_SKELETON
        self.angles = ANGLE_JOINTS

    def bone_dir_loss(self, stereo_aligned, target):
        loss = 0
        for parent, child in self.bones:
            bone_stereo = stereo_aligned[child] - stereo_aligned[parent]
            bone_target = target[child] - target[parent]

            bone_stereo = F.normalize(bone_stereo, dim = -1)
            bone_target = F.normalize(bone_target, dim = -1)

            loss += ((bone_stereo - bone_target) ** 2).sum(dim = -1)                          # Takes the Euclidean norm of each element and averages per bone across the batch
        return loss / len(self.bones)
    
    def bone_length_loss(self, stereo_aligned, target):
        loss = 0
        for parent, child in self.bones:
            length_stereo = torch.linalg.vector_norm(stereo_aligned[child] - stereo_aligned[parent], dim = -1)
            length_target = torch.linalg.vector_norm(target[child] - target[parent], dim = -1)
            loss += ((length_target - length_stereo) ** 2)
        return loss / len(self.bones)
    
    def angle_loss(self, stereo_aligned, target):
        loss = 0
        for parent, joint, child in self.angles:                                    # gets the angles between joints
            s1 = stereo_aligned[parent] - stereo_aligned[joint]
            s2 = stereo_aligned[child] - stereo_aligned[joint]

            t1 = target[parent] - target[joint]
            t2 = target[child] - target[joint]
            
            cos_stereo = F.cosine_similarity(s1, s2, dim = -1)
            cos_target = F.cosine_similarity(t1, t2, dim = -1)
            loss += ((cos_stereo - cos_target) ** 2)
        return loss / len(self.angles)
    
class Optimizer:
    def __init__(self, w1: float, w2: float, w3: float, lr: float, num_steps: int):
        self.w1 = w1
        self.w2 = w2
        self.w3 = w3
        self.lr = lr
        self.num_steps = num_steps

        self.R = None
        self.s = None
        self.t = None

        self.losses = Losses()

    def umeyama(self, src, dest):                                                                                           # src and dest is shape (N, 3)
        src = np.asarray(src)
        dest = np.asarray(dest)

        N = src.shape[0]
        src_mean = src.mean(axis = 0)
        dest_mean = dest.mean(axis = 0)
        
        src_centered = src - src_mean
        dest_centered = dest - dest_mean

        cov = (dest_centered.T @ src_centered) / N
        U, D, Vt = np.linalg.svd(cov)

        S = np.eye(3)
        if np.linalg.det(U) * np.linalg.det(Vt) < 0:
            S[-1, -1] = -1
        R = U @ S @ Vt

        var_src = np.mean(np.sum(src_centered ** 2, axis = 1))
        scale = np.trace(np.diag(D) @ S) / var_src
        t = dest_mean - scale * R @ src_mean
        
        self.R = R
        self.scale = scale
        self.t = t

    def transform(self, coords):
        aligned = (self.scale * (self.R @ coords.T)).T + self.t
        return aligned
    
    def inverse_transform(self, stereo_optim):
        stereo_orig = ((self.R.T @ ((stereo_optim - self.t).T / self.scale))).T
        return stereo_orig
        
    def optimize(self, stereo_coords: np.ndarray, target: np.ndarray):
        self.umeyama(stereo_coords, target)
        stereo_aligned = torch.tensor(self.transform(stereo_coords), dtype = torch.float32)    
        coords = torch.nn.Parameter(stereo_aligned.clone())   # Optimization variable
        target = torch.tensor(target, dtype = torch.float32)                                                     
        optimizer = torch.optim.Adam([coords], lr = self.lr)

        for _ in range(self.num_steps):
            # Optimization occurs in target space
            optimizer.zero_grad()
            bone_dir_loss = self.losses.bone_dir_loss(coords, target)
            bone_length_loss = self.losses.bone_length_loss(coords, target)
            bone_angle_loss = self.losses.angle_loss(coords, target)
            loss = (self.w1 * bone_dir_loss) + (self.w2 * bone_length_loss) + (self.w3 * bone_angle_loss)
            loss.backward()
            optimizer.step()

        coords_optim = coords.detach().cpu().numpy()
        stereo_optim = self.inverse_transform(coords_optim)

        # Restore original hand position (translation only to keep original depth estimate)
        PALM = [0, 1, 5, 9, 13, 17]

        orig_center = stereo_coords[PALM].mean(axis = 0)
        optim_center = stereo_optim[PALM].mean(axis = 0)

        translation = orig_center - optim_center
        stereo_optim += translation

        return stereo_optim
