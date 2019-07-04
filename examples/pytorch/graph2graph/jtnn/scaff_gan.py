import torch as th
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
# from mol_tree import Vocab, MolTree
from .g2g_encoder import G2GEncoder

class ScaffoldGAN(nn.Module):
    


