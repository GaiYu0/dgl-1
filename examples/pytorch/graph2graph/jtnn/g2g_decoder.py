import torch as th
import torch.nn as nn
import torch.nn.functional as F

class TreeGRU(nn.Module):
    def __init__(self, in_feats, out_feats, h_key, f_src_key, f_dst_key):
        super(TreeGRU, self).__init__()

        self.wz = nn.Parameter(th.rand(in_feats, out_feats))
        self.uz = nn.Parameter(th.rand(out_feats, out_feats))
        self.bz = nn.Parameter(th.zeros(1, out_feats))
        self.wr = nn.Parameter(th.rand(in_feats, out_feats))
        self.ur = nn.Parameter(th.rand(out_feats, out_feats))
        self.br = nn.Parameter(th.zeros(1, out_feats))
        self.w = nn.Parameter(th.rand(in_feats, out_feats))
        self.u = nn.Parameter(th.rand(out_feats, out_feats))
        self.b = nn.Parameter(th.zeros(1, out_feats))

        self.h_key = h_key
        self.f_src_key = f_src_key
        self.f_dst_key = f_dst_key

    def forward(self, G_lg, eids):
        s_message_fn = fn.copy_src(src=self.h_key, out=self.h_key)
        s_reduce_fn = fn.reducer.sum(msg=self.h_key, out='s')
        G_lg.pull(eids, s_message_fn, s_reduce_fn)  # Eq. (20)

        def z_apply_fn(nodes):
            f_src = nodes.data[self.f_src_key]
            s = nodes.data['s']
            z = F.sigmoid(f_src @ self.wz + s @ self.uz + self.bz)  # Eq. (21)
            return {'z' : z}
        G_lg.apply_nodes(z_apply_fn, eids)

        def h_tilde_message_fn(nodes, edges):
            f_dst = nodes.data[self.f_dst_key]
            h = nodes.data[self.h_key]
            r = F.sigmoid(f_dst @ self.wr + h @ self.ur + self.br)  # Eq. (22)
            r_times_h = r * h
            return {'r_times_h' : r_times_h}
        h_tilde_reduce_fn = fn.reducer.sum(msg='r_times_h', out='sum_r_times_h')
        def h_tilde_apply_fn(nodes):
           f_src = nodes.data[self.f_src_key]
           sum_r_times_h = nodes.data['sum_r_times_h']
           h_tilde = F.tanh(f_src @ self.w + sum_r_times_h @ self.u + self.b)  # Eq. (23)
           return {'h_tilde' : h_tilde}
        G_lg.pull(eids, h_tilde_message_fn, h_tilde_reduce_fn, h_tilde_apply_fn)

        def h_apply_fn(nodes):
            z = nodes.data['z']
            s = nodes.data['s']
            h_tilde = nodes.data['h_tilde']
            h = (1 - z) * s + z * h_tilde  # Eq. (24)
            return {self.h_key : h}
        G_lg.apply_nodes(h_apply_fn, eids)

        G_lg.pop_n_repr('s')
        G_lg.pop_n_repr('z')
        G_lg.pop_n_repr('sum_r_times_h')
        G_lg.pop_n_repr('h_tilde')

class Attention(nn.Module):
    def __init__(self, d_h=None, d_x=None, a=None):
        super(Attention, self).__init__()
        self.a = nn.Parameter(th.rand(d_h, d_x)) if a is None else a

    def forward(self, h, x):
        """
        Parameters
        ----------
        h : (m, d_h)
        x : (n, d_x)
        """
        att = F.softmax(x @ (h @ a).t())  # Eq. (25)
        z = th.sum(att * x, 0)
        return z

def dfs_order(forest, roots):
    ret = dgl.dfs_labeled_edges_generator(forest, roots, has_reverse_edge=True)
    for eids, label in zip(*ret):
        yield eids ^ label

class G2GDecoder(nn.Module):
    def __init__(self, d_f, d_xT, d_xG, d_h, d_msg, d_zd):
        self.tree_gru = TreeGRU(d_f, d_msg, 'msg', 'f_src', 'f_dst')

        self.w_d1 = nn.Parameter(th.rand(d_f, d_h))
        self.w_d2 = nn.Parameter(th.rand(d_msg, d_h))
        self.b_d1 = nn.Parameter(th.zeros(1, d_h))

        self.att_dT = Attention(d_h, d_xT)
        self.att_dG = Attention(d_h, d_xG)
        '''
        assert d_xT == d_xG
        self.a_d = nn.Parameter(th.rand(d_h, dx_T)
        self.att_dT = self.att_dG = Attention(self.a_d)
        '''

        self.w_d3 = nn.Parameter(th.rand(d_h, d_zd))
        self.w_d4 = nn.Parameter(th.rand(d_xT + d_xG, d_zd))
        self.b_d2 = nn.Parameter(th.zeros(1, d_zd))

        self.u_d = nn.Parameter(th.rand(1, d_h))
        self.b_d3 = nn.Parameter(th.zeros(1))

        self.w_l1 = nn.Parameter(th.rand(d_msg, d_zl))
        self.w_l2 = nn.Parameter(th.rand(d_xT + d_xG, d_zl))
        self.b_l1 = nn.Parameter(th.zeros(1, d_zl))

        self.att_lT = Attention(d_msg, d_xT)
        self.att_lG = Attention(d_msg, d_xG)
        '''
        assert d_xT == d_xG
        self.a_l = nn.Parameter(th.rand(d_msg, dx_T)
        self.att_lT = self.att_lG = Attention(self.a_l)
        '''

        self.u_l = nn.Parameter(th.rand(1, d_zl))
        self.b_l2 = nn.Parameter(th.zeros(1))

    def forward(self, train):
        if train:
            self.train()
        else:
            self.test()

    def train(self, G, T):
        """
        Parameters
        ----------
        G : DGLBatchedGraph
        T : DGLBatchedGraph
        """
        roots = np.cumsum([0] + T.batch_num_nodes)[:-1]
        T_lg = T.line_graph(backtracking=False, shared=True)
        T_lg.ndata['msg'] = th.zeros(T_lg.number_of_nodes(), self.d_msg)

        for eids in dfs_order(T, roots):
            self.tree_gru(T_lg, eids)

            # topology prediction
            h_message_fn = fn.copy_src(src='h', out='h')
            h_reduce_fn = fn.reducer.sum(src='h', out='sum_h')
            T_lg.pull(h_message_fn, h_reduce_fn)
            f_src = nodes[eids].data['f_src']
            sum_h = nodes[eids].data['sum_h']
            h = F.relu(self.w_d1 @ f_src + self.w_d2 @ sum_h + self.b_d1)  # Eq. (4)
            c_tT = self.att_tT(h, T.ndata['x'])
            c_tG = self.att_tG(h, G.ndata['x'])
            c_t = th.cat([c_tT, c_tG], 1)  # Eq. (5) (7)
            z_t = th.relu(h @ self.w_d3 + c @ self.w_d4 + self.b_d2)
            p = self.u_d @ z + self.b_d3  # Eq. (6)
            expand = eids ^ 1
            topology_loss =  -(expand * th.logsigmoid(p) + (1 - expand) * th.logsigmoid(1 - p))

            # label prediction
            msg = T_lg.nodes[eids].data['h']
            c_lT = self.att_lT(msg, T.ndata['x'])
            c_lG = self.att_lG(msg, G.ndata['x'])
            c_l = th.cat([c_lT, c_lG], 1)  # Eq. (8)
            z_l = th.relu(msg @ self.w_l1 + c_l @ self.w_l2 + self.b_l1)
            q = self.u_l @ z_l + self.b_l2  # Eq. (9)
            tree_ce = -th.log_softmax(q, 1)
