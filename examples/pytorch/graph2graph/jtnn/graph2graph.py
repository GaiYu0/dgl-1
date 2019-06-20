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
        g1_candidates_G = G1(args.d_ndataG_dec, args.d_edataG_dec,args.d_msgG)
        g2_candidates_G = G2(args.d_ndataG_dec, args.d_msgG, args.d_xG)
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

    
    def scoring_candidates(self, c_readout, c_idx, X_embedding, X_bnn):
        """
        scoring all candidates graphs

        # could be done better with scatter_multiply
        """
        c_score = th.FloatTensor(c_readout.size()[0]).to(c_readout.device)
        X_bnn = th.cumsum(X_bnn,0)
        for c in range(c_readout.size()[0]):
            c_score[c] = th.sum(c_readout[c, :] *\
                                 X_embedding[X_bnn[c_idx[c]]:X_bnn[c_idx[c]+1],:])
        return c_score 
        

    def candidates_loss(self, c_score, Y_T):
        val_c_map = Y_T.ndata['num_cands'] > 1
        not_leaf_map = ~Y_T.ndata['is_leaf']
        val_node = th.masked_select(Y_T.nodes().to(device=val_c_map.device), val_c_map & not_leaf_map)
        cum_num_cands = [0] + list(th.cumsum(th.masked_select(Y_T.ndata['num_cands'], 
                                                    val_c_map & not_leaf_map),0))
        all_loss = []
        acc = 0
        for i, n_id in enumerate(val_node):
            cur_c_score = c_score[int(cum_num_cands[i]):int(cum_num_cands[i+1])]
            label = Y_T.ndata['label'][n_id].unsqueeze(0)
            all_loss.append(F.cross_entropy(cur_c_score.view(1, -1), label, size_average=False))
            
            # counting correct
            if th.argmax(cur_c_score) == label:
                acc += 1 
        all_loss = sum(all_loss) /Y_T.batch_size

        return all_loss, acc/len(val_node)
         
        
    
    def assemble(self, candidates_G, candidates_G_idx, Y_T, X_G_embedding, Y_T_msgs, X_G_bnn):
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
        candidates_G_readout = dgl.sum_nodes(candidates_G, 'x')
        candidates_score = self.scoring_candidates(candidates_G_readout, candidates_G_idx,
                                                   X_G_embedding, X_G_bnn)
        loss, accu = self.candidates_loss(candidates_score, Y_T)


        return loss, accu




    def forward(self, batch):
        #\TODO fix self.process
        X_G, X_T, _, _, _ = self.process(batch[0])
        Y_G, Y_T, candidates_G, Y_T_msgs, candidates_G_idx = self.process(batch[1])
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

        X_G_embedding = X_G.ndata['x']
        delta_G = dgl.sum_nodes(X_G, 'x') - dgl.sum_nodes(Y_G, 'x')  # Eq. (11)
        mu_G = self.mu_G(delta_G)
        logvar_G = -th.abs(self.logvar_G(delta_G))  # Mueller et al.
        z_G = mu_G + th.exp(logvar_G) ** 0.5 * th.rand_like(mu_G, device=device)  # Eq. (12)
        z_G = th.repeat_interleave(z_G, XG_bnn, 0)
        x_G = X_G.ndata['x']
        x_tildeG = F.relu(x_G @ self.w3 + z_G @ self.w4 + self.b2)  # Eq. (13)
        X_G.ndata['x'] = x_tildeG

        topology_ce, label_ce = self.decoder(X_G, X_T, Y_G, Y_T)

        # 0 start
        X_G_bnn = th.LongTensor([0] + X_G.batch_num_nodes)
        assm_loss, assm_acc = self.assemble(candidates_G, candidates_G_idx, Y_T, X_G_embedding, Y_T_msgs, X_G_bnn)
        kl_div = -0.5 * th.sum(1 + logvar_G - mu_G ** 2 - th.exp(logvar_G)) / batch_size - \
                 0.5 * th.sum(1 + logvar_T - mu_T ** 2 - th.exp(logvar_T)) / batch_size
        return topology_ce, label_ce, assm_loss, kl_div

    def process(self, batch):
        device = self.mu_G.weight.device
        # fetching molecular graphs
        G = batch['mol_graph_batch']
        G.ndata['f'] = G.ndata['x'].to(device)
        G.pop_n_repr('x')
        # G.pop_n_repr('src_x')
        G.edata['f'] = G.edata['x'].to(device)
        G.pop_e_repr('x')

        # fetching molecular junction trees
        for tree in batch['mol_trees']:
            n = tree.number_of_nodes()
            tree.ndata['wid'] = tree.ndata['wid'].to(device)
            tree.ndata['id'] = th.zeros(n, self.vocab.size(), device=device)
            tree.ndata['id'][th.arange(n), tree.ndata['wid']] = 1
            tree.ndata['is_leaf'] = th.ByteTensor(tree.number_of_nodes()).to(device)
            tree.ndata['num_cands'] = th.LongTensor(tree.number_of_nodes()).to(device)
            tree.ndata['label'] = th.LongTensor(tree.number_of_nodes()).to(device)
            for n_id in tree.nodes():
                tree.ndata['is_leaf'][int(n_id)] = tree.nodes_dict[int(n_id)]['is_leaf']
                tree.ndata['num_cands'][int(n_id)] = len(tree.nodes_dict[int(n_id)]['cands'])
                tree.ndata['label'][int(n_id)] = tree.nodes_dict[int(n_id)]['cands']\
                                                                .index(tree.nodes_dict[int(n_id)]['label'])
            #tree.ndata['is_leaf'] = tree.ndata['is_leaf'].to(device)
            #tree.ndata['num_cands'] = tree.ndata['num_cands'].to(device)
            #tree.ndata['label'] = tree.ndata['label'].to(device)

        T = dgl.batch(batch['mol_trees'])

        # fetching molecular's related candidate graphs

        candidates_G = batch['cand_graph_batch']
        candidates_G.ndata['f'] = candidates_G.ndata['x'].to(device)
        candidates_G.pop_n_repr('x')
        candidates_G.edata['f'] = candidates_G.edata['x'].to(device)
        candidates_G.pop_e_repr('x')

        # ground truth junction tree mapping to candidate graphs
        gt_Y_T_msgs = [batch['tree_mess_src_e'],
                       batch['tree_mess_tgt_e'],
                       batch['tree_mess_tgt_n']]

        candidates_G_idx = th.LongTensor(batch['cand_batch_idx']).to(device)
        
        return G, T, candidates_G, gt_Y_T_msgs, candidates_G_idx
