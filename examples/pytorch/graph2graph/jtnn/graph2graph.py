import copy, math
import dgl
import torch as th
import torch.nn as nn
import torch.nn.functional as F

import rdkit
import rdkit.Chem as Chem

from .chemutils import set_atommap, copy_edit_mol, enum_assemble_nx, \
     attach_mols_nx, decode_stereo
from .g2g_encoder import G2GEncoder
from .g2g_decoder import TreeGRU, G2GDecoder
from .g2g_jtmpn import g2g_JTMPN
from .g2g_jtmpn import mol2dgl_single as mol2dgl_dec

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
        
        self.embeddings = nn.Parameter(1e-3 * th.rand(args.vocab_size, args.d_ndataT))
        
        # = self.jtnn
        # d_msgG = d_msgT = hidden_size
        self.encoder = G2GEncoder(self.embeddings, g1_G, g1_T, g2_G, g2_T,
                                  args.d_msgG, args.d_msgT, args.n_itersG, args.n_itersT)

        self.decoder = G2GDecoder(self.embeddings, args.d_ndataG, args.d_ndataT, args.d_xG, args.d_xT,
                                  args.d_msgT, args.d_h, args.d_ud, [args.d_ul, args.vocab_size], vocab)
        
        # depth_G = n_itersG
        self.g2g_jtmpn = g2g_JTMPN(g1_candidates_G, g2_candidates_G, args.d_msgG, args.d_msgT, args.n_itersG)

        
        
        # sampling param
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

        # assembling param
        self.A_assm = nn.Linear(args.d_xG, args.d_xG, bias=False)
        # Not used
        # self.assm_loss = nn.CrossEntropyLoss(size_average=False)

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
        # teacher forcing: given Y_T's structure and x_g, x_t perturb vectors, construct a P_G (predicted_graph)
        # and compute assembling loss with respect to Y_G
        Input:
        Predicted Junction Tree T_\hat
        """
        # jtmpn is used to decode a given junction tree as well as candidates to actual graphs
        self.g2g_jtmpn(candidates_G, Y_T, Y_T_msgs)
        candidates_G_readout = dgl.sum_nodes(candidates_G, 'x')
        candidates_score = self.scoring_candidates(candidates_G_readout, candidates_G_idx,
                                                   X_G_embedding, X_G_bnn)
        loss, accu = self.candidates_loss(candidates_score, Y_T)

        return loss, accu

    def decode(self, x_T, x_G):
        gen_T = self.decoder.decode(x_T, x_G)
        print("##########DECODER END##############")
        if gen_T.number_of_nodes() == 1: return gen_T.nodes_dict[0]['smiles']
        # set atom map, change to 1-indexed for this sake
        leaf_count = 0
        for id in gen_T.nodes():
            id = int(id)
            gen_T.nodes_dict[id].update({'idx' : id})
            gen_T.nodes_dict[id].update({"nid": id + 1})
            #\TODO double check author's definition of neighbors
            if len(gen_T.predecessors(id).tolist()) == 1:
                gen_T.nodes_dict[id].update({'is_leaf' : True})
                leaf_count += 1
            else:
                gen_T.nodes_dict[id].update({'is_leaf' : False})
                set_atommap(gen_T.nodes_dict[id]['mol'], gen_T.nodes_dict[id]['nid'])
        self.encoder(None, gen_T)

        # \TODO understand what's going on here. Don't copy line 108 - 110 blindly
        x_G_readout = th.sum(x_G, dim=0).unsqueeze(0) # another option: average pooling
        x_G_readout = self.A_assm(x_G_readout) # bilinear transform

        cur_mol = copy_edit_mol(gen_T.nodes_dict[0]['mol'])
        # amap: atom map
        global_amap = [{}] + [{} for node in gen_T.nodes_dict]
        global_amap[1] = {atom.GetIdx() : atom.GetIdx() for atom in cur_mol.GetAtoms()}

        cur_mol = self.dfs_assemble(gen_T, x_G_readout, cur_mol, global_amap, [], 0, -1)
        
        if cur_mol is None:
            return None
        
        cur_mol = cur_mol.GetMol()
        set_atommap(cur_mol)
        cur_mol = Chem.MolFromSmiles(Chem.MolToSmiles(cur_mol))

        return Chem.MolToSmiles(cur_mol) if cur_mol is not None else None

        # Debug only
        return gen_T
    
    def dfs_assemble(self, gen_T, x_readout, cur_mol, global_amap, p_amap, cur_node_id, p_node_id):
        # p_amap: parent atom map
        # p_node: parent node
        device = gen_T.ndata['x'].device

        
        p_node_dict = gen_T.nodes_dict[p_node_id] if p_node_id != -1 else None
        fa_nid = p_node_dict['nid'] if p_node_dict is not None else -1
        cur_node_dict = gen_T.nodes_dict[cur_node_id]

        #p_node_nid = p_node_id + 1
        prev_nodes_dict = [p_node_dict] if p_node_dict is not None else []

        child_id = [v for v in gen_T.successors(cur_node_id).tolist() if gen_T.nodes_dict[v]['nid'] != fa_nid]

        child_dict = [gen_T.nodes_dict[id] for id in child_id]
        neighbors = [nei_id for nei_id in child_id if gen_T.nodes_dict[nei_id]['mol'].GetNumAtoms() > 1]
        neighbors = sorted(neighbors, key=lambda x: gen_T.nodes_dict[x]['mol'].GetNumAtoms(), reverse=True)
        singletons = [nei_id for nei_id in child_id if gen_T.nodes_dict[nei_id]['mol'].GetNumAtoms() == 1]
        neighbors = singletons + neighbors
        neighbors_dict = [gen_T.nodes_dict[id] for id in neighbors]

        # nid is only saved to amap
        cur_amap = [(fa_nid, a2, a1) for nid, a1, a2 in p_amap if nid == gen_T.nodes_dict[cur_node_id]['nid']]
        cands = enum_assemble_nx(cur_node_dict, neighbors_dict, prev_nodes_dict, cur_amap)

        if len(cands) == 0:
            return None
        cand_smiles, cand_mols, cand_amap = list(zip(*cands))
        #\TODO this is not used
        mol_tree_msg = gen_T.ndata['x']
        cands = [(candmol, gen_T, cur_node_id) for candmol in cand_mols]
        cand_graphs, atom_x, bond_x, tree_mess_src_edges, \
            tree_mess_tgt_edges, tree_mess_tgt_nodes = mol2dgl_dec(cands)
        
        gen_T_msgs = [tree_mess_src_edges, tree_mess_tgt_edges, tree_mess_tgt_nodes]
        cand_graphs = dgl.batch(cand_graphs)
        cand_graphs.ndata['f'] = atom_x.to(device)
        cand_graphs.edata['f'] = bond_x.to(device)

        
        
        # \TODO I think we don't need the line below
        # cand_graphs.edata['src_f'] = atom_x.new(bond_x.shape[0], atom_x.shape[1]).zero_()
        self.g2g_jtmpn(cand_graphs, gen_T, gen_T_msgs)
        

        cand_graphs_readout = dgl.sum_nodes(cand_graphs, 'x')
        scores = cand_graphs_readout @ x_readout.t()

        _, cand_idx = th.sort(scores, descending=True)

        #cand_idx = cand_idx.squeeze(1).tolist()

        backup_mol = Chem.RWMol(cur_mol)
        for i in range(min(cand_idx.numel(), 5)):
            cur_mol = Chem.RWMol(backup_mol)
            pred_amap = cand_amap[cand_idx[i].item()]
            new_global_amap = copy.deepcopy(global_amap)

            for nei_id, ctr_atom, nei_atom in pred_amap:
                if nei_id == fa_nid:
                    continue
                new_global_amap[nei_id][nei_atom] = new_global_amap[cur_node_id+1][ctr_atom]
            
            cur_mol = attach_mols_nx(cur_mol, child_dict, [], new_global_amap)
            new_mol = cur_mol.GetMol()
            new_mol = Chem.MolFromSmiles(Chem.MolToSmiles(new_mol))

            if new_mol is None:
                continue
            
            result = True
            for nei_node_id, nei_node in zip(child_id, child_dict):
                if nei_node['is_leaf']:
                    continue
                cur_mol = self.dfs_assemble(
                    gen_T, x_readout, cur_mol, new_global_amap, pred_amap, nei_node_id, cur_node_id
                )
                if cur_mol is None:
                    result = False
                    break
            
            if result:
                return cur_mol
        
        return None
    
    def sample_with_noise(self, x_T, x_G):
        z_T = th.rand_like(x_T, device=x_T.device)
        z_G = th.rand_like(x_G, device=x_G.device)
        x_G_tilde = F.relu(x_G @ self.w1 + z_G @ self.w2 + self.b1)
        x_T_tilde = F.relu(x_T @ self.w3 + z_T @ self.w4 + self.b2)

        return x_T_tilde, x_G_tilde

    def sample_from_diff(self, X, Y, mean_gen, var_gen, w1, w2, b):
        # \TODO I'm actually not sure whether the shape is right for batch version.
        device = X.ndata['f'].device
        X_bnn = th.tensor(X.batch_num_nodes, device=device)
        Y_bnn = th.FloatTensor(Y.batch_num_nodes).unsqueeze(1).to(device)
        # NOTICE: gaiyu->hq: implementation changed to Y - X
        delta = dgl.sum_nodes(Y, 'x') - dgl.sum_nodes(X, 'x') # Eq. (11)
        # Normalized by the size of y
        delta /= Y_bnn

        mu = mean_gen(delta)
        logvar = -th.abs(var_gen(delta)) # Mueller et al.
        z = mu + th.exp(logvar) ** 0.5 * th.rand_like(mu, device=device) # Eq. (12)
        z = th.repeat_interleave(z, X_bnn, 0)
        x = X.ndata['x']
        x_tilde = F.relu(x @ w1 + z @ w2 + b) # Eq. (13)
        #X.ndata['x'] = x_tilde
        
        return x_tilde, mu, logvar
    
    def forward(self, batch):
        #\TODO fix self.process
        X_G, X_T, _, _, _ = self.process(batch[0])
        Y_G, Y_T, candidates_G, Y_T_msgs, candidates_G_idx = self.process(batch[1])
        device=X_G.ndata['f'].device
        batch_size = X_G.batch_size
        XG_bnn = th.tensor(X_G.batch_num_nodes, device=device)
        XT_bnn = th.tensor(X_T.batch_num_nodes, device=device)

        self.encoder(X_G, X_T)
        print("#### THIS IS X_G embedding")
        print("min {}, max {}, sum {}\
              ".format(th.min(X_G.ndata['x']).item(),th.max(X_G.ndata['x']).item(),
                      th.sum(X_G.ndata['x']).item()))
        print('###### This is X_T embedding')
        print("min {}, max {}, sum {}\
              ".format(th.min(X_T.ndata['x']).item(),th.max(X_T.ndata['x']).item(),
                      th.sum(X_T.ndata['x']).item()))
        print('########')
        self.encoder(Y_G, Y_T)
        print("#### THIS IS Y_G embedding")
        print("min {}, max {}, sum {}\
              ".format(th.min(Y_G.ndata['x']).item(),th.max(Y_G.ndata['x']).item(),
                      th.sum(Y_G.ndata['x']).item()))
        print('###### This is Y_T embedding')
        print("min {}, max {}, sum {}\
              ".format(th.min(Y_T.ndata['x']).item(),th.max(Y_T.ndata['x']).item(),
                      th.sum(Y_T.ndata['x']).item()))
        print('########')

        X_G_embedding = X_G.ndata['x']
        x_G_tilde, mu_G, logvar_G = self.sample_from_diff(X_G, Y_G, self.mu_G, self.logvar_G,
                                                self.w1, self.w2, self.b1)
        x_T_tilde, mu_T, logvar_T = self.sample_from_diff(X_T, Y_T, self.mu_T, self.logvar_T,
                                                self.w3, self.w4, self.b2)
        
        X_G.ndata['x'] = x_G_tilde
        X_T.ndata['x'] = x_T_tilde

        topology_ce, label_ce, topo_acc, label_acc = self.decoder(X_G, X_T, Y_G, Y_T)

        # 0 start
        X_G_bnn = th.LongTensor([0] + X_G.batch_num_nodes)
        assm_loss, assm_acc = self.assemble(candidates_G, candidates_G_idx, Y_T, X_G_embedding, Y_T_msgs, X_G_bnn)
        kl_div = -0.5 * th.sum(1.0 + logvar_G - mu_G * mu_G - th.exp(logvar_G)) / batch_size - \
                 0.5 * th.sum(1.0 + logvar_T - mu_T * mu_G - th.exp(logvar_T)) / batch_size
        return topology_ce, label_ce, assm_loss, kl_div, topo_acc, label_acc, assm_acc

    def process(self, batch, train=True):
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

        T = dgl.batch(batch['mol_trees'])

        # fetching molecular's related candidate graphs

        if train:
            candidates_G = batch['cand_graph_batch']
            candidates_G.ndata['f'] = candidates_G.ndata['x'].to(device)
            candidates_G.pop_n_repr('x')
            candidates_G.edata['f'] = candidates_G.edata['x'].to(device)
            candidates_G.pop_e_repr('x')
            candidates_G_idx = th.LongTensor(batch['cand_batch_idx']).to(device)

        # ground truth junction tree mapping to candidate graphs
            gt_Y_T_msgs = [batch['tree_mess_src_e'],
                           batch['tree_mess_tgt_e'],
                           batch['tree_mess_tgt_n']]

            return G, T, candidates_G, gt_Y_T_msgs, candidates_G_idx
        else:
            return G, T
