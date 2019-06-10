import dgl
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from .mol_tree import Vocab
from .nnutils import create_var, cuda, move_dgl_to_cuda
from .chemutils import set_atommap, copy_edit_mol, enum_assemble_nx, \
        attach_mols_nx, decode_stereo
from .jtnn_enc import DGLJTNNEncoder
from .jtnn_dec import DGLJTNNDecoder
from .mpn import DGLMPN
from .mpn import mol2dgl_single as mol2dgl_enc
from .jtmpn import DGLJTMPN
from .jtmpn import mol2dgl_single as mol2dgl_dec
from .line_profiler_integration import profile

import rdkit
import rdkit.Chem as Chem
from rdkit import DataStructs
from rdkit.Chem import AllChem
import copy, math

from dgl import batch, unbatch

from .g2g_encoder import G2GEncoder
from .g2g_decoder import TreeGRU, G2GDecoder

class Graph2Graph(nn.Module):
    def __init__(self, args):
        super(Graph2Graph, self).__init__()
        g1_G = self.G1(args.d_ndataG, args.d_edataG, args.d_msgG)
        g2_G = self.G2(args.d_ndataG, args.d_msgG, args.d_xG)
        g1_T = TreeGRU(args.d_ndataT, args.d_msgT, 'msg', 'f_src', 'f_dst')
        g2_T = self.G2(args.d_ndataT, args.d_msgT, args.d_xT)
        embeddings = nn.Parameter(th.rand(args.vocab_size, args.d_ndataT))
        self.encoder = G2GEncoder(embeddings, g1_G, g1_T, g2_G, g2_T,
                                  args.d_msgG, args.d_msgT, args.n_itersG, args.n_itersT)

        self.decoder = G2GDecoder(embeddings, args.d_ndataG, args.d_ndataT, args.d_xG, args.d_xT,
                                  args.d_msgT, args.d_h, args.d_ud, [args.d_ul, args.vocab_size])

        self.w1 = nn.Parameter(th.rand(args.d_xG, args.d_xG))
        self.w2 = nn.Parameter(th.rand(args.d_zG, args.d_xG))
        self.b1 = nn.Parameter(th.zeros(1, args.d_xG))

        self.w3 = nn.Parameter(th.rand(args.d_xT, args.d_xT))
        self.w4 = nn.Parameter(th.rand(args.d_zT, args.d_xT))
        self.b2 = nn.Parameter(th.zeros(1, args.d_xT))

        self.mu_G = nn.Linear(args.d_xG, args.d_zG)
        self.logvar_G = nn.Linear(args.d_xG, args.d_zG)
        self.mu_T = nn.Linear(args.d_xT, args.d_zT)
        self.logvar_T = nn.Linear(args.d_xT, args.d_zT)

    class G1(nn.Module):
        def __init__(self, d_ndata, d_edata, d_msg):
            super().__init__()
            self.w1 = nn.Parameter(th.rand(d_ndata, d_msg))
            self.w2 = nn.Parameter(th.rand(d_edata, d_msg))
            self.w3 = nn.Parameter(th.rand(d_msg, d_msg))
            self.b = nn.Parameter(th.zeros(1, d_msg))

        def forward(self, f_src, f, sum_msg):
            return F.relu(f_src @ self.w1 + f @ self.w2 + sum_msg @ self.w3 + self.b)  # Eq. (17)

    class G2(nn.Module):
        def __init__(self, d_ndata, d_msg, d_x):
            super().__init__()
            self.u1 = nn.Parameter(th.rand(d_ndata, d_x))
            self.u2 = nn.Parameter(th.rand(d_msg, d_x))
            self.b = nn.Parameter(th.zeros(1, d_x))

        def forward(self, f, sum_msg):
            return F.relu(f @ self.u1 + sum_msg @ self.u2 + self.b)  # Eq. (18)

    def forward(self, X_G, X_T, Y_G, Y_T):
        batch_size = X_G.batch_size

        self.encoder(X_G, X_T)
        self.encoder(Y_G, Y_T)

        delta_T = dgl.sum_nodes(X_T, 'x') - dgl.sum_nodes(Y_T, 'x')  # Eq. (11)
        mu_T = self.mu_T(delta_T)
        logvar_T = self.logvar_T(delta_T)
        z_T = th.exp(logvar_T) ** 0.5 * (mu_T + th.rand_like(mu_T))  # Eq. (12)
        z_T = th.repeat_interleave(z_T, th.tensor(X_T.batch_num_nodes), 0)
        x_T = X_T.ndata['x']
        x_tildeT = F.relu(x_T @ self.w1 + z_T @ self.w2 + self.b2)  # Eq. (13)

        delta_G = dgl.sum_nodes(X_G, 'x') - dgl.sum_nodes(Y_G, 'x')  # Eq. (11)
        mu_G = self.mu_G(delta_G)
        logvar_G = self.logvar_G(delta_G)
        z_G = th.exp(logvar_G) ** 0.5 * (mu_G + th.rand_like(mu_G))  # Eq. (12)
        z_G = th.repeat_interleave(z_G, th.tensor(X_G.batch_num_nodes), 0)
        x_G = X_G.ndata['x']
        x_tildeG = F.relu(x_G @ self.w3 + z_G @ self.w4 + self.b2)  # Eq. (13)

        topology_ce, label_ce = self.decoder(Y_G, Y_T, x_tildeG, x_tildeT,
                                             th.tensor(X_G.batch_num_nodes),
                                             th.tensor(X_T.batch_num_nodes))

        kl_div = -0.5 * th.sum(1 + logvar_G - mu_G ** 2 - th.exp(logvar_G)) / batch_size - \
                 0.5 * th.sum(1 + logvar_T - mu_T ** 2 - th.exp(logvar_T)) / batch_size

        return topology_ce, label_ce, kl_div
