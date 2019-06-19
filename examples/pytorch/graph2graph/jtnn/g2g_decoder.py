import dgl
import dgl.function as fn
import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F

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
    def __init__(self, embeddings, d_ndataG, d_ndataT, d_xG, d_xT, d_msgT, d_h, d_ud, d_ul):
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

        self.u_d = nn.Parameter(1e-3 * th.rand(d_ud, 2))
        self.b_d3 = nn.Parameter(th.zeros(2))

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
            expand = 1 - eids % 2
            topology_ce += F.cross_entropy(p, expand)

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

        return topology_ce / (i + 1), label_ce / (i + 1)

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
