import dgl
import dgl.function as fn
import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from .mol_tree_nx import DGLMolTree
from .datautils import one_hotify
from .chemutils import enum_assemble_nx, get_mol

MAX_DECODE_LEN = 100
EXPAND = 1
NEXPAND = 0

def copy_src(G, u, uv):
    src, dst = G.edges()
    G.edata[uv] = G.ndata[u][src]

def copy_dst(G, v, uv):
    src, dst = G.edges()
    G.edata[uv] = G.ndata[v][dst]

def create_node_dict(smiles, clique=[]):
    return dict(
            smiles=smiles,
            mol=get_mol(smiles),
            clique=clique,
            )

def have_slots(fa_slots, ch_slots):
    if len(fa_slots) > 2 and len(ch_slots) > 2:
        return True
    matches = []
    for i,s1 in enumerate(fa_slots):
        a1,c1,h1 = s1
        for j,s2 in enumerate(ch_slots):
            a2,c2,h2 = s2
            if a1 == a2 and c1 == c2 and (a1 != "C" or h1 + h2 >= 4):
                matches.append( (i,j) )

    if len(matches) == 0: return False

    fa_match,ch_match = list(zip(*matches))
    if len(set(fa_match)) == 1 and 1 < len(fa_slots) <= 2: #never remove atom from ring
        fa_slots.pop(fa_match[0])
    if len(set(ch_match)) == 1 and 1 < len(ch_slots) <= 2: #never remove atom from ring
        ch_slots.pop(ch_match[0])

    return True
    
def can_assemble(mol_tree, u, v_node_dict):
    u_node_dict = mol_tree.nodes_dict[u]
    u_neighbors = mol_tree.successors(u)
    u_neighbors_node_dict = [
            mol_tree.nodes_dict[_u]
            for _u in u_neighbors
            if _u in mol_tree.nodes_dict
            ]
    neis = u_neighbors_node_dict + [v_node_dict]
    for i,nei in enumerate(neis):
        nei['nid'] = i

    neighbors = [nei for nei in neis if nei['mol'].GetNumAtoms() > 1]
    neighbors = sorted(neighbors, key=lambda x:x['mol'].GetNumAtoms(), reverse=True)
    singletons = [nei for nei in neis if nei['mol'].GetNumAtoms() == 1]
    neighbors = singletons + neighbors
    cands = enum_assemble_nx(u_node_dict, neighbors)
    return len(cands) > 0


class TreeGRU(nn.Module):
    def __init__(self, d_f, d_h, h_key, f_src_key, f_dst_key):
        """
        Parameters
        ----------
        d_f : dimension of $f_i$ in Eq. (21) ~ (23)
        d_h : dimension of $h_{i j}$ in Eq. (20) (22) ~ (24)
        h_key : str
            field for $h_{k i}$ in Eq. (20)
        f_src_key : str
            field for source node data
        f_dst_key : str
            field for destination node data

        Tree GRU has been verified.
        """
        super(TreeGRU, self).__init__()

        self.wz = nn.Parameter(1e-3 * th.rand(d_f, d_h))
        self.uz = nn.Parameter(1e-3 * th.rand(d_h, d_h))
        self.bz = nn.Parameter(th.zeros(1, d_h))
        self.wr = nn.Parameter(1e-3 * th.rand(d_f, d_h))
        self.ur = nn.Parameter(1e-3 * th.rand(d_h, d_h))
        self.br = nn.Parameter(th.zeros(1, d_h))
        self.w = nn.Parameter(1e-3 * th.rand(d_f, d_h))
        self.u = nn.Parameter(1e-3 * th.rand(d_h, d_h))
        self.b = nn.Parameter(th.zeros(1, d_h))

        self.h_key = h_key
        self.f_src_key = f_src_key
        self.f_dst_key = f_dst_key

    def forward(self, G_lg, eids=None):
        """
        Parameters
        ----------
        G_lg : DGLBatchedGraph
            line graph
        eids : id of edges to update
        """
        s_message_fn = fn.copy_src(src=self.h_key, out=self.h_key)
        s_reduce_fn = fn.reducer.sum(msg=self.h_key, out='s')
        def z_apply_fn(nodes):
            f_src = nodes.data[self.f_src_key]
            s = nodes.data['s']
            z = th.sigmoid(f_src @ self.wz + s @ self.uz + self.bz)  # Eq. (21)
            return {'z' : z}
        if eids is None:
            G_lg.update_all(s_message_fn, s_reduce_fn, z_apply_fn)  # Eq. (20)
        else:
            G_lg.pull(eids, s_message_fn, s_reduce_fn, z_apply_fn)  # Eq. (20)

        def h_tilde_message_fn(edges):
            f_dst = edges.src[self.f_dst_key]
            h = edges.src[self.h_key]
            r = th.sigmoid(f_dst @ self.wr + h @ self.ur + self.br)  # Eq. (22)
            r_times_h = r * h
            return {'r_times_h' : r_times_h}
        h_tilde_reduce_fn = fn.reducer.sum(msg='r_times_h', out='sum_r_times_h')
        def h_tilde_apply_fn(nodes):
           f_src = nodes.data[self.f_src_key]
           sum_r_times_h = nodes.data['sum_r_times_h']
           h_tilde = th.tanh(f_src @ self.w + sum_r_times_h @ self.u + self.b)  # Eq. (23)
           return {'h_tilde' : h_tilde}
        if eids is None:
            G_lg.update_all(h_tilde_message_fn, h_tilde_reduce_fn, h_tilde_apply_fn)
        else:
            G_lg.pull(eids, h_tilde_message_fn, h_tilde_reduce_fn, h_tilde_apply_fn)

        def h_apply_fn(nodes):
            z = nodes.data['z']
            s = nodes.data['s']
            h_tilde = nodes.data['h_tilde']
            h = (1 - z) * s + z * h_tilde  # Eq. (24)
            return {self.h_key : h}
        if eids is None:
            G_lg.apply_nodes(h_apply_fn)
        else:
            G_lg.apply_nodes(h_apply_fn, eids)

        # TODO(gaiyu): pop_n_repr
        '''
        G_lg.pop_n_repr('s')
        G_lg.pop_n_repr('z')
        G_lg.pop_n_repr('sum_r_times_h')
        G_lg.pop_n_repr('h_tilde')
        '''

class Attention(nn.Module):
    def __init__(self, d_h=None, d_x=None, a=None):
        """
        Parameters
        ----------
        d_h : dimension of $h_t$ in Eq. (25)
        d_x : dimension of $x_i^T$ or $x_i^G$ in Eq. (25)

        Attention has been verified
        """
        super(Attention, self).__init__()
        self.a = nn.Parameter(1e-3 * th.rand(d_h, d_x)) if a is None else a

    def forward(self, h, G):
        """
        Parameters
        ----------
        h : (m, d_h)
        x : (n, d_x)
        """
        device = h.device
        bnn = th.tensor(G.batch_num_nodes, device=device)
        G.ndata['s'] = th.sum(G.ndata['x'] * th.repeat_interleave(h @ self.a, bnn, 0), 1)
        #numerical stability
        G.ndata['exp'] = th.exp(G.ndata['s'] - th.repeat_interleave(dgl.max_nodes(G, 's'), bnn))
        z = th.repeat_interleave(dgl.sum_nodes(G, 'exp'), bnn)
        a = th.unsqueeze(G.ndata['exp'] / z, 1)  # Eq. (25)
        G.ndata['a_times_x'] = a * G.ndata['x']
        ret = dgl.sum_nodes(G, 'a_times_x')
        G.pop_n_repr('s')
        G.pop_n_repr('exp')
        G.pop_n_repr('a_times_x')
        return ret
        
class G2GDecoder(nn.Module):
    def __init__(self, embeddings, d_ndataG, d_ndataT, d_xG, d_xT, d_msgT, d_h, d_ud, d_ul, vocab):
        """
        Parameters
        ----------
        embedding :
        d_ndataG : dimension of molecule's node data ($f_u$ in paper)
        d_ndataT : dimension of junction tree's node data ($f_u$ in paper)
        d_xG : dimension of molecule's node embedding ($x_u^G$ in paper)
        d_xT : dimension of junction tree's node embedding ($x_u^T$ in paper)
        d_msgG : dimension of graph's message vector in message passing
        d_msgT : dimension of junction tree's message vector in message passing
        d_h : dimension of $h_t$ in Eq. (4)
        d_ud : dimension of $u^d$ in Eq. (6)
        d_ul : dimension of $U^l$ in Eq. (9)
        """
        super().__init__()

        self.d_msgT = d_msgT
        self.d_h = d_h
        self.vocab = vocab
        self.d_ndataT = d_ndataT

        self.embeddings = embeddings
        self.tree_gru = TreeGRU(d_ndataT, d_msgT, 'msg', 'f_src', 'f_dst')

        self.w_d1 = nn.Parameter(1e-3 * th.rand(d_ndataT, d_h))
        self.w_d2 = nn.Parameter(1e-3 * th.rand(d_msgT, d_h))
        self.b_d1 = nn.Parameter(th.zeros(1, d_h))

        self.att_dT = Attention(d_h, d_xT)
        self.att_dG = Attention(d_h, d_xG)
        '''
        assert d_xT == d_xG
        self.a_d = nn.Parameter(th.rand(d_h, dx_T)
        self.att_dT = self.att_dG = Attention(self.a_d)
        '''

        self.w_d3 = nn.Parameter(1e-3 * th.rand(d_h, d_ud))
        self.w_d4 = nn.Parameter(1e-3 * th.rand(d_xT + d_xG, d_ud))
        self.b_d2 = nn.Parameter(th.zeros(1, d_ud))

        # (hq) change u_d to BCEWithLogitLoss, from 2 to 1
        self.u_d = nn.Parameter(1e-3 * th.rand(d_ud, 1))
        self.b_d3 = nn.Parameter(th.zeros(1))

        self.w_l1 = nn.Parameter(1e-3 * th.rand(d_msgT, d_ul[0]))
        self.w_l2 = nn.Parameter(1e-3 * th.rand(d_xT + d_xG, d_ul[0]))
        self.b_l1 = nn.Parameter(th.zeros(1, d_ul[0]))

        self.att_lT = Attention(d_msgT, d_xT)
        self.att_lG = Attention(d_msgT, d_xG)
        '''
        assert d_xT == d_xG
        self.a_l = nn.Parameter(th.rand(d_msgT, dx_T)
        self.att_lT = self.att_lG = Attention(self.a_l)
        '''

        self.u_l = nn.Parameter(1e-3 * th.rand(*d_ul))
        self.b_l2 = nn.Parameter(th.zeros(1, d_ul[1]))

        self.expand_loss = nn.BCEWithLogitsLoss(size_average=False)

    def forward(self, X_G, X_T, Y_G, Y_T):
        """
        Parameters
        ----------
        X_G : DGLBatchedGraph
            source molecules
        X_T : DGLBatchedGraph
            source junction trees
        Y_G : DGLBatchedGraph
            groundtruth molecules
        Y_T : DGLBatchedGraph
            groundtruth junction trees

        Returns
        -------
        topology_ce
        label_ce
        """
        device = X_G.ndata['x'].device
        bnn = th.tensor(Y_T.batch_num_edges, device=device)

        Y_T.ndata['f'] = Y_T.ndata['id'] @ self.embeddings
        # T_lg is constructed from the groundtruth tree Y_T
        T_lg = Y_T.line_graph(backtracking=False, shared=True)
        T_lg.ndata['msg'] = th.zeros(T_lg.number_of_nodes(), self.d_msgT, device=device)

        topology_ce = 0
        label_ce = 0
        roots = np.cumsum([0] + Y_T.batch_num_nodes)[:-1]
        topology_count = 0
        label_count = 0
        for i, eids in enumerate(self.dfs_order(Y_T, roots)):
            eids = eids.to(device)
            to_continue = self.to_continue(eids.cpu(), bnn.cpu())
            self.tree_gru(T_lg, eids)

            # topology prediction
            h_message_fn = fn.copy_src(src='msg', out='msg')
            h_reduce_fn = fn.reducer.sum(msg='msg', out='sum_h')
            T_lg.ndata['sum_h'] = th.zeros(Y_T.number_of_edges(), self.d_msgT, device=device)
            T_lg.pull(eids, h_message_fn, h_reduce_fn)
            f_src = T_lg.nodes[eids].data['f_src']
            sum_h = T_lg.nodes[eids].data['sum_h']
            h = F.relu(f_src @ self.w_d1 +  sum_h @ self.w_d2 + self.b_d1)  # Eq. (4)
            h_batched = h[th.cumsum(to_continue.long(), 0) - 1]
            c_dT = self.att_dT(h_batched, X_T)
            c_dG = self.att_dG(h_batched, X_G)
            c_d = th.cat([c_dT, c_dG], 1)[to_continue]  # Eq. (5) (7)
            z_d = th.relu(h @ self.w_d3 + c_d @ self.w_d4 + self.b_d2)
            p = z_d @ self.u_d + self.b_d3  # Eq. (6)
            expand = (1 - eids % 2).unsqueeze(1).float()
            #print("this is stop loss logit shape ", p.size())
            #topology_ce += F.cross_entropy(p, expand)
            topology_ce += self.expand_loss(p, expand)
            hard_expand = th.ge(p, 0).float()
            correct = th.eq(hard_expand, expand).float()
            topology_count += th.sum(correct)
            #topology_ce += F.BCEWithLogitsLoss(p, expand)
            #topology_count += (th.argmax(p, dim=1) == expand).sum().float()

            # label prediction
            msg = T_lg.nodes[eids].data['msg']
            msg_batched = msg[th.cumsum(to_continue.long(), 0) - 1]
            c_lT = self.att_lT(msg_batched, X_T)
            c_lG = self.att_lG(msg_batched, X_G)
            c_l = th.cat([c_lT, c_lG], 1)[to_continue]  # Eq. (8)
            z_l = th.relu(msg @ self.w_l1 + c_l @ self.w_l2 + self.b_l1)
            q = z_l @ self.u_l + self.b_l2  # Eq. (9)
            _, dst = Y_T.edges()
            dst = dst.to(device)
            label_ce += F.cross_entropy(q, Y_T.nodes[dst[eids]].data['wid'])
            label_count += (th.argmax(q, dim=1) == Y_T.nodes[dst[eids]].data['wid']).sum().float()
        
        topo_acc = topology_count / Y_T.number_of_edges()
        label_acc = label_count / Y_T.number_of_nodes()
        return topology_ce / (i + 1), label_ce / (i + 1), topo_acc, label_acc

    @staticmethod
    def dfs_order(forest, roots):
        ret = dgl.dfs_labeled_edges_generator(forest, roots, has_reverse_edge=True)
        for eids, label in zip(*ret):
            yield eids ^ label

    @staticmethod
    def to_continue(eids, bnn):
        """
        Parameters
        ----------
        eids : torch.tensor
            (m,)
        bnn : torch.tensor

        Returns
        -------
        """
        lower = th.unsqueeze(th.cumsum(th.cat([th.zeros(1), bnn.float()[:-1]]), 0), 0)
        upper = th.unsqueeze(th.cumsum(bnn.float(), 0), 0)
        eids = th.unsqueeze(eids.float(), 1)
        isin = (lower <= eids) & (eids < upper)
        return th.any(isin, 0)

    def decode_label(self, T, curr_nid, context, T_lg):
        # info needed from parent node
        p_slot = T.nodes_dict[curr_nid]['slot']
        device = T.ndata['f'].device

        eid = T.number_of_edges() - 1
        if len(T_lg.predecessors(eid)) == 0:
            msg = th.zeros(1, self.d_msgT).to(device)
        else:
            self.tree_gru(T_lg, eid)
            msg = T_lg.nodes[eid].data['msg']
        c_l = context # Eq. (8)
        z_l = th.relu(msg @ self.w_l1 + c_l @ self.w_l2 + self.b_l1)
        q = z_l @ self.u_l + self.b_l2  # Eq. (9)
        _, sort_wid = th.sort(q, dim=1, descending=True)
        sort_wid = sort_wid.data.squeeze()

        next_wid = None
        for wid in sort_wid[:5]:
            slot = self.vocab.get_slots(wid)
            cand_node_dict = create_node_dict(self.vocab.get_smiles(wid))

            if have_slots(p_slot,slot) and can_assemble(T, curr_nid, cand_node_dict):
                next_wid = wid
                next_slots = slot
                break
                
        if next_wid is None:
            #\TODO see how to remove node
            end_node_idx = T.nodes()[-1]
            end_lg_node_idx = T_lg.nodes()[-1]
            T.remove_nodes(end_node_idx)
            T_lg.remove_nodes(end_lg_node_idx)
            self.expand = NEXPAND
            print("fail to find correct label, backtrack instead")
        else:
            end_node_idx = T.number_of_nodes() - 1
            cur_smiles = self.vocab.get_smiles(next_wid)
            # print("cur smiles is ", cur_smiles)
            #print(" number of node in the gen tree is ", T.number_of_nodes())
            T.nodes_dict[end_node_idx] = {}
            T.nodes_dict[end_node_idx]['smiles'] = cur_smiles
            #T.nodes(end_node_idx).data['wid'] = next_wid
            T.nodes_dict[end_node_idx]['wid'] = next_wid
            T.nodes_dict[end_node_idx]['slot'] = next_slots
            T.nodes_dict[end_node_idx]['mol'] = get_mol(cur_smiles)
            T.nodes[end_node_idx].data['parent'] = th.LongTensor([curr_nid]).unsqueeze(0).to(device)
            curr_nid = end_node_idx
        
        return T, T_lg, curr_nid
    
    def soft_decode_label(self, T, curr_nid, context, T_lg, stop_val_ste):
        device = T.ndata['f'].device

        eid = T.number_of_edges() - 1
        if len(T_lg.predecessors(eid)) == 0:
            msg = th.zeros(1, self.d_msgT).to(device)
        else:
            self.tree_gru(T_lg, eid)
            msg = T_lg.nodes[eid].data['msg']
        T_lg.nodes[eid].data['msg'] *= stop_val_ste
        c_l = context # Eq. (8)
        z_l = th.relu(msg @ self.w_l1 + c_l @ self.w_l2 + self.b_l1)
        q = z_l @ self.u_l + self.b_l2  # Eq. (9)
        
        pred_prob = self.sample_softmax(q)
        end_node_idx = T.number_of_nodes() - 1
        curr_nid = end_node_idx
        T.nodes[curr_nid].data['f'] = self.soft_embedding(pred_prob)

        return T, T_lg, curr_nid
            
    def decode_backtrack(self, T, curr_nid, old_T_lg):
        device = T.ndata['f'].device
        parent = int(T.nodes[curr_nid].data['parent'].squeeze(0))

        T.add_edges([curr_nid], [parent])
        copy_src(T, 'f', 'f_src')
        copy_dst(T, 'f', 'f_dst')

        T_lg = T.line_graph(backtracking=False, shared=True)
        T_lg.ndata['msg'] = th.zeros(T_lg.number_of_nodes(), self.d_msgT, device=device)
        T_lg.ndata['sum_h'] = th.zeros(T.number_of_edges(), self.d_msgT, device=device)
        if old_T_lg is not None:
            T_lg.ndata['msg'][:-1,:] = old_T_lg.ndata['msg'][:-1, :]
            T_lg.ndata['sum_h'][:-1,:] = old_T_lg.ndata['sum_h'][:-1, :]
        
        eid = T.number_of_edges() - 1
        self.tree_gru(T_lg, eid)

        curr_nid = int(T.nodes[curr_nid].data['parent'])

        return curr_nid

    def soft_decode_backtrack(self, T, curr_nid, old_T_lg, stop_val_ste):
        device = T.ndata['f'].device
        parent = int(T.nodes[curr_nid].data['parent'].squeeze(0))

        T.add_edges([curr_nid], [parent])
        copy_src(T, 'f', 'f_src')
        copy_dst(T, 'f', 'f_dst')

        T_lg = T.line_graph(backtracking=False, shared=True)
        T_lg.ndata['msg'] = th.zeros(T_lg.number_of_nodes(), self.d_msgT, device=device)
        T_lg.ndata['sum_h'] = th.zeros(T.number_of_edges(), self.d_msgT, device=device)
        if old_T_lg is not None:
            T_lg.ndata['msg'][:-1,:] = old_T_lg.ndata['msg'][:-1, :]
            T_lg.ndata['sum_h'][:-1,:] = old_T_lg.ndata['sum_h'][:-1, :]
        
        eid = T.number_of_edges() - 1
        self.tree_gru(T_lg, eid)

        # edit edge feature here:
        T_lg.nodes[eid].data['h'] *= (1.0 - stop_val_ste) # what is new_h[0]?

        curr_nid = int(T.nodes[curr_nid].data['parent'])



    def decode_stop(self, T, curr_nid, context, old_T_lg = None):
        device = T.ndata['f'].device
        cur_label = one_hotify([T.nodes_dict[curr_nid]['wid']], 
                                pad=self.u_l.size(1))
        cur_label = cur_label.to(device)
        # use the embedding of current wid to initiate f.
        f = cur_label @ self.embeddings
        T.nodes[curr_nid].data['f'] = f

        T.add_nodes(1)
        T.add_edges([curr_nid], [T.nodes()[-1]])
        eid = T.number_of_edges() - 1
        copy_src(T, 'f', 'f_src')
        copy_dst(T, 'f', 'f_dst')

        # old_T_lg: DGL does not mantain a line graph dynamically

        T_lg = T.line_graph(backtracking=False, shared=True)
        T_lg.ndata['msg'] = th.zeros(T_lg.number_of_nodes(), self.d_msgT, device=device)
        T_lg.ndata['sum_h'] = th.zeros(T.number_of_edges(), self.d_msgT, device=device)
        if old_T_lg is not None:
            T_lg.ndata['msg'][:-1,:] = old_T_lg.ndata['msg'][:-1, :]
            T_lg.ndata['sum_h'][:-1,:] = old_T_lg.ndata['sum_h'][:-1, :]
        h_message_fn = fn.copy_src(src='msg', out='msg')
        h_reduce_fn = fn.reducer.sum(msg='msg', out='sum_h')
        T_lg.pull(eid, h_message_fn, h_reduce_fn)
        f_src = T_lg.nodes[eid].data['f_src']
        sum_h = T_lg.nodes[eid].data['sum_h']

        h = F.relu(f_src @ self.w_d1 +  sum_h @ self.w_d2 + self.b_d1)  # Eq. (4)
        c_d = context # Eq. (8), only attend to root
        z_d = th.relu(h @ self.w_d3 + c_d @ self.w_d4 + self.b_d2)
        p = z_d @ self.u_d + self.b_d3
        
        self.expand = th.argmax(p)

        if self.expand == NEXPAND:
            end_node_idx = T.nodes()[-1]
            end_lg_node_idx = T_lg.nodes()[-1]
            T.remove_nodes(end_node_idx)
            if T.number_of_nodes() == 1:
                return T, None # basecase, do not create new line graph
            pruned_T_lg = T.line_graph(backtracking=False, shared=True)
            pruned_T_lg.ndata['msg'] = th.zeros(T.number_of_edges(), self.d_msgT, device=device)
            pruned_T_lg.ndata['sum_h'] = th.zeros(T.number_of_edges(), self.d_msgT, device=device)
            pruned_T_lg.ndata['msg'] = T_lg.ndata['msg'][:-1,:]
            pruned_T_lg.ndata['sum_h']= T_lg.ndata['sum_h'][:-1,:]
            T_lg = pruned_T_lg


        return T, T_lg
    
    def soft_decode_stop(self, T, curr_nid, context, all_hiddens, old_T_lg = None):
        device = T.ndata['f'].device
        cur_label = one_hotify([T.nodes_dict[curr_nid]['wid']], 
                                pad=self.u_l.size(1))
        cur_label = cur_label.to(device)
        # use the embedding of current wid to initiate f.
        f = cur_label @ self.embeddings
        T.nodes[curr_nid].data['f'] = f

        T.add_nodes(1)
        T.add_edges([curr_nid], [T.nodes()[-1]])
        eid = T.number_of_edges() - 1
        copy_src(T, 'f', 'f_src')
        copy_dst(T, 'f', 'f_dst')

        # old_T_lg: DGL does not mantain a line graph dynamically

        T_lg = T.line_graph(backtracking=False, shared=True)
        T_lg.ndata['msg'] = th.zeros(T_lg.number_of_nodes(), self.d_msgT, device=device)
        T_lg.ndata['sum_h'] = th.zeros(T.number_of_edges(), self.d_msgT, device=device)
        if old_T_lg is not None:
            T_lg.ndata['msg'][:-1,:] = old_T_lg.ndata['msg'][:-1, :]
            T_lg.ndata['sum_h'][:-1,:] = old_T_lg.ndata['sum_h'][:-1, :]
        h_message_fn = fn.copy_src(src='msg', out='msg')
        h_reduce_fn = fn.reducer.sum(msg='msg', out='sum_h')
        T_lg.pull(eid, h_message_fn, h_reduce_fn)
        f_src = T_lg.nodes[eid].data['f_src']
        sum_h = T_lg.nodes[eid].data['sum_h']

                
        self.cur_x = f_src
        self.cur_h = sum_h


        h = F.relu(f_src @ self.w_d1 +  sum_h @ self.w_d2 + self.b_d1)  # Eq. (4)
        c_d = context # Eq. (8), only attend to root
        z_d = th.relu(h @ self.w_d3 + c_d @ self.w_d4 + self.b_d2)
        all_hiddens.append(h)
        p = z_d @ self.u_d + self.b_d3
        
        # We consider self.expand as stop score for the time being
        self.expand = th.argmax(p) #\TODO stop score binary or unary?

        stop_prob = F.hardtanh(slope * self.expand + 0.5, min_val=0, max_val=1).unsqueeze(1)
        stop_val_ste = self.expand + stop_prob - stop_prob.detach() # straight through estimator

        if self.expand == NEXPAND:
            end_node_idx = T.nodes()[-1]
            end_lg_node_idx = T_lg.nodes()[-1]
            T.remove_nodes(end_node_idx)
            if T.number_of_nodes() == 1:
                return T, None # basecase, do not create new line graph
            pruned_T_lg = T.line_graph(backtracking=False, shared=True)
            pruned_T_lg.ndata['msg'] = th.zeros(T.number_of_edges(), self.d_msgT, device=device)
            pruned_T_lg.ndata['sum_h'] = th.zeros(T.number_of_edges(), self.d_msgT, device=device)
            pruned_T_lg.ndata['msg'] = T_lg.ndata['msg'][:-1,:]
            pruned_T_lg.ndata['sum_h']= T_lg.ndata['sum_h'][:-1,:]
            T_lg = pruned_T_lg


        return T, T_lg, all_hiddens, stop_val_ste

    def soft_decode(self, x_T, x_G, gumbel, slope, temp):
        self.soft_embedding = lambda x: x @ self.embeddings
        if gumbel:
            self.sample_softmax = lambda x: F.gumbel_softmax(x, tau=temp)
        else:
            self.sample_softmax = lambda x: F.softmax(x / temp, dim=1)
        
        device = x_G.device

        # during decoding, decoder only attends to the first tree/mol node embedding
        d_context = th.cat([th.unsqueeze(x_T[0,:],0), th.unsqueeze(x_G[0,:],0)], 1) # Eq. (8)
        # init_h (message vector) is a zero vector
        z_l = th.relu(d_context @ self.w_l2 + self.b_l1)
        q = z_l @ self.u_l + self.b_l2  # Eq. (9)
        root_prob = self.sample_softmax(q)
        # we do not predict the most likely wid

        gen_T = DGLMolTree()
        gen_T.add_nodes(1)
        gen_T.ndata['f'] = self.soft_embedding(root_prob)
        gen_T.ndata['parent'] = (th.LongTensor([-1]).unsqueeze(0)).to(device)
        curr_nid = 0

        all_hiddens = []
        self.cur_x = None
        self.cur_h = None

        for i in range(MAX_DECODE_LEN):
            if i == 0:
                gen_T, gen_T_lg, all_hiddens, stop_val_ste = self.soft_decode_stop(gen_T, 
                                                                    curr_nid, d_context, all_hiddens)
            else:
                gen_T, gen_T_lg, all_hiddens, stop_val_ste = self.soft_decode_stop(gen_T, 
                                                                    curr_nid, d_context, all_hiddens, gen_T_lg)
            
            if self.expand == EXPAND:
                gen_T, gen_T_lg, curr_nid = self.soft_decode_label(gen_T, curr_nid, d_context, gen_T_lg, stop_val_ste)
            
            else: # Not expand
                if gen_T.number_of_nodes() == 1:
                    return th.cat([self.cur_x, self.cur_h], dim=1) #\TODO fix this
                else:
                    curr_nid = self.soft_decode_backtrack(gen_T, curr_nid, gen_T_lg, stop_val_ste)
        
        readout_msg_fn = fn.copy_edge('h', 'msg')
        readout_reduce_fn = fn.reducer.sum('msg', 'sum_h')
        readout_apply_fn = lambda nodes : {"r" : th.cat(nodes.data['f'], nodes.data['sum_msg'])}
        gen_T.update_all(readout_msg_fn, readout_reduce_fn, readout_apply_fn)

        return Y_T.nodes[0].data['r']





    
    def decode(self, x_T, x_G):
        """
        Parameters:
        -----------
        x_G: sampled or computed graph embedding
        x_T: sampled or computed junction tree embedding

        Returns
        -------
        """
        # Currently, we do not support batch decoding unless DGLBatchGraph
        # add mutable support

        # x_T size: num_node x feat_dim, we assume that it's a single graph instance
        #assert x_T.size(0) == 1, "batch decoding not supported, please unbatch first"

        print("###############START DECODING A MOLECULE#####################")
        device = x_G.device

        # during decoding, decoder only attends to the first tree/mol node embedding
        #\TODO This is not True!!!!!! It actually attends to the first graph.
        d_context = th.cat([th.unsqueeze(x_T[0,:],0), th.unsqueeze(x_G[0,:],0)], 1) # Eq. (8)
        # init_h (message vector) is a zero vector
        z_l = th.relu(d_context @ self.w_l2 + self.b_l1)
        q = z_l @ self.u_l + self.b_l2  # Eq. (9)

        root_score = F.softmax(q, dim=1)
        _, root_wid = th.max(root_score, dim=1)
        root_wid = root_wid.item()
        root_smiles = self.vocab.get_smiles(root_wid)
        print("root_wid is ", root_wid)
        print("root smiles is ", root_smiles)

        gen_T = DGLMolTree()
        gen_T.add_nodes(1)
        gen_T.ndata['f'] = th.FloatTensor(gen_T.number_of_nodes(), self.d_ndataT).zero_().to(device)
        gen_T.ndata['parent'] = (th.LongTensor([-1]).unsqueeze(0)).to(device)
        #gen_T.ndata['wid'] = th.LongTensor(gen_T.number_of_nodes(), 1).zero_().to(device)

        slot = self.vocab.get_slots(root_wid)
        cur_smiles = self.vocab.get_smiles(root_wid)
        curr_nid = 0
        
        gen_T.nodes_dict[curr_nid] = {'smiles':cur_smiles, 'wid':root_wid, 'slot':slot, 'mol':get_mol(cur_smiles)}
        for i in range(MAX_DECODE_LEN):
            if i == 0:
                # a line graph is initialized in the first step
                gen_T, gen_T_lg = self.decode_stop(gen_T, curr_nid, d_context)
            else:
                gen_T, gen_T_lg = self.decode_stop(gen_T, curr_nid, d_context, gen_T_lg)

            if self.expand == EXPAND:
                # print("non stop, go for label")
                gen_T, gen_T_lg, curr_nid = self.decode_label(gen_T, curr_nid, d_context, gen_T_lg)

            if self.expand == NEXPAND:
                print("backtrack!")
                if curr_nid == 0:
                    print("terminate due to early stop")
                    return gen_T
                curr_nid = self.decode_backtrack(gen_T, curr_nid, gen_T_lg)
        
        n = gen_T.number_of_nodes()
        gen_T.ndata['id'] = th.zeros(n, self.vocab.size(), device=device)
        for id in range(n):
            gen_T.ndata['id'][id, gen_T.nodes_dict[id]['wid']] = 1
        #gen_T.ndata['id'][th.arange(n), gen_T.ndata['wid']] = 1
        return gen_T







