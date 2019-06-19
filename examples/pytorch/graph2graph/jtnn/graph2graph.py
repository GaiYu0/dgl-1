import dgl
import torch as th
import torch.nn as nn
import torch.nn.functional as F

from .g2g_encoder import G2GEncoder
from .g2g_decoder import TreeGRU, G2GDecoder
from .g2g_jtmpn import g2g_JTMPN

class G1(nn.Module):
    def __init__(self, d_ndata, d_edata, d_msg):
        super().__init__()
        self.w1 = nn.Parameter(1e-3 * th.rand(d_ndata, d_msg))
        self.w2 = nn.Parameter(1e-3 * th.rand(d_edata, d_msg))
        self.w3 = nn.Parameter(1e-3 * th.rand(d_msg, d_msg))
        self.b = nn.Parameter(th.zeros(1, d_msg))

    def forward(self, f_src, f, sum_msg):
        return F.relu(f_src @ self.w1 + f @ self.w2 + sum_msg @ self.w3 + self.b)  # Eq. (17)

class G2(nn.Module):
    def __init__(self, d_ndata, d_msg, d_x):
        super().__init__()
        self.u1 = nn.Parameter(1e-3 * th.rand(d_ndata, d_x))
        self.u2 = nn.Parameter(1e-3 * th.rand(d_msg, d_x))
        self.b = nn.Parameter(th.zeros(1, d_x))

    def forward(self, f, sum_msg):
        return F.relu(f @ self.u1 + sum_msg @ self.u2 + self.b)  # Eq. (18)

class Graph2Graph(nn.Module):
    def __init__(self, args, vocab):
        super(Graph2Graph, self).__init__()
        g1_G = G1(args.d_ndataG, args.d_edataG, args.d_msgG)
        g2_G = G2(args.d_ndataG, args.d_msgG, args.d_xG)
        g1_T = TreeGRU(args.d_ndataT, args.d_msgT, 'msg', 'f_src', 'f_dst')
        g2_T = G2(args.d_ndataT, args.d_msgT, args.d_xT)
        g1_candidates_G = G1(args.d_ndataG, args.d_edataG,args.d_msgG)
        g2_candidates_G = G2(args.d_ndataG, args.d_msgG, args.d_xG)
        embeddings = nn.Parameter(1e-3 * th.rand(args.vocab_size, args.d_ndataT))
        self.encoder = G2GEncoder(embeddings, g1_G, g1_T, g2_G, g2_T,
                                  args.d_msgG, args.d_msgT, args.n_itersG, args.n_itersT)

        self.decoder = G2GDecoder(embeddings, args.d_ndataG, args.d_ndataT, args.d_xG, args.d_xT,
                                  args.d_msgT, args.d_h, args.d_ud, [args.d_ul, args.vocab_size])
        self.g2g_jtmpn = g2g_JTMPN(g1_candidates_G, g2_candidates_G, args.d_msgG, args.d_msgT, args.n_itersG)

        self.w1 = nn.Parameter(1e-3 * th.rand(args.d_xG, args.d_xG))
        self.w2 = nn.Parameter(1e-3 * th.rand(args.d_zG, args.d_xG))
        self.b1 = nn.Parameter(th.zeros(1, args.d_xG))

        self.w3 = nn.Parameter(1e-3 * th.rand(args.d_xT, args.d_xT))
        self.w4 = nn.Parameter(1e-3 * th.rand(args.d_zT, args.d_xT))
        self.b2 = nn.Parameter(th.zeros(1, args.d_xT))

        self.mu_G = nn.Linear(args.d_xG, args.d_zG)
        self.logvar_G = nn.Linear(args.d_xG, args.d_zG)
        self.mu_T = nn.Linear(args.d_xT, args.d_zT)
        self.logvar_T = nn.Linear(args.d_xT, args.d_zT)

        self.mode = "train"
        self.vocab = vocab
    
    def set_mode(self, mode):
        self.mode = mode

    
    def scoring_candidates(self, candidates_G, Y_G_vecs, candidates_batch_idx):
        """
        scoring all candidates graphs
        """
        candidates_vecs = candidates_G.ndata['x']
        device = candidates_vecs.device
        batch_idx = th.LongTensor(candidates_batch_idx).to(device)
        # ground truth mol graph vecs
        gt_G_vecs = Y_G_vecs[batch_idx]
        

        return 
    def assemble(self, candidates_G, Y_T, x_tildeG, Y_T_msgs):
        """
        candidates_G is a two-level batched graph of all candidates graphs
        assuming the training batch size is B and on average each training example
        has C candidate graphs, then candidates_G actually contain B*C small graphs

        # teacher forcing: given Y_T's structure and x_g, x_t perturb vectors, construct a P_G (predicted_graph)
        # and compute assembling loss with respect to Y_G
        Input:
        Predicted Junction Tree T_\hat

        Procedure:
        1) Find all realizations of clustering attachment (need to clarify)
        2) Apply a Graph MPN over each realization and compute the atom representations
        3) Global sum readout of atom representations to compute each realization's overall score
        4) compute the score function
        5) Write up the assembling loss score function (Equation 10)
        During training we do teacher forcing: i.e. we have the ground truth junction tree.
        """
        # jtmpn is used to decode a given junction tree as well as candidates to actual graphs
        self.g2g_jtmpn(candidates_G, Y_T, Y_T_msgs)
        candidates_score = self.scoring_candidates(candidates_vecs, x_tildeG)
        loss = self.candidates_loss(candidates_score)

        return loss




    def forward(self, batch):
        #\TODO fix self.process
        X_G, X_T, _ = self.process(batch[0])
        Y_G, Y_T, _ = self.process(batch[1])
        device=X_G.ndata['f'].device
        batch_size = X_G.batch_size
        XG_bnn = th.tensor(X_G.batch_num_nodes, device=device)
        XT_bnn = th.tensor(X_T.batch_num_nodes, device=device)

        self.encoder(X_G, X_T)
        self.encoder(Y_G, Y_T)

        delta_T = dgl.sum_nodes(X_T, 'x') - dgl.sum_nodes(Y_T, 'x')  # Eq. (11)
        mu_T = self.mu_T(delta_T)
        logvar_T = -th.abs(self.logvar_T(delta_T))  # Mueller et al.
        z_T = mu_T + th.exp(logvar_T) ** 0.5 * th.rand_like(mu_T, device=device)  # Eq. (12)
        z_T = th.repeat_interleave(z_T, XT_bnn, 0)
        x_T = X_T.ndata['x']
        x_tildeT = F.relu(x_T @ self.w1 + z_T @ self.w2 + self.b2)  # Eq. (13)
        X_T.ndata['x'] = x_tildeT

        delta_G = dgl.sum_nodes(X_G, 'x') - dgl.sum_nodes(Y_G, 'x')  # Eq. (11)
        mu_G = self.mu_G(delta_G)
        logvar_G = -th.abs(self.logvar_G(delta_G))  # Mueller et al.
        z_G = mu_G + th.exp(logvar_G) ** 0.5 * th.rand_like(mu_G, device=device)  # Eq. (12)
        z_G = th.repeat_interleave(z_G, XG_bnn, 0)
        x_G = X_G.ndata['x']
        x_tildeG = F.relu(x_G @ self.w3 + z_G @ self.w4 + self.b2)  # Eq. (13)
        X_G.ndata['x'] = x_tildeG

        topology_ce, label_ce = self.decoder(X_G, X_T, Y_G, Y_T)

        assm_loss, assm_acc = self.assemble(X_G, X_T, Y_G, Y_T)

        kl_div = -0.5 * th.sum(1 + logvar_G - mu_G ** 2 - th.exp(logvar_G)) / batch_size - \
                 0.5 * th.sum(1 + logvar_T - mu_T ** 2 - th.exp(logvar_T)) / batch_size

        return topology_ce, label_ce, kl_div

    def process(self, batch):
        device = self.mu_G.weight.device
        G = batch['mol_graph_batch']
        G.ndata['f'] = G.ndata['x'].to(device)
        G.pop_n_repr('x')
        # G.pop_n_repr('src_x')
        G.edata['f'] = G.edata['x'].to(device)
        G.pop_e_repr('x')
        for tree in batch['mol_trees']:
            n = tree.number_of_nodes()
            tree.ndata['wid'] = tree.ndata['wid'].to(device)
            tree.ndata['id'] = th.zeros(n, self.vocab.size(), device=device)
            tree.ndata['id'][th.arange(n), tree.ndata['wid']] = 1
        T = dgl.batch(batch['mol_trees'])
        return G, T
