import dgl.function as fn
import torch as th
import torch.nn as nn

import g2g_utils

def G2GEncoder(nn.Module):
    def __init__(self, cluster_embeddings, g1, g2, d_msg, n_iterations):
        super(G2GEncoder, self).__init__()
        self.cluster_embeddings = cluster_embeddings
        self.g1 = g1
        self.g2 = g2
        self.d_msg = d_msg
        self.n_iterations = n_iterations

    def forward(self, G, T):
        # TODO G.ndata['f'], G.edata['f'], T.ndata['f'], T.edata['f']
        g2g_utils.copy_src(G, 'f', 'f_src')
        g2g_utils.copy_src(T, 'f', 'f_src')

        G_lg = G.line_graph(backtracking=False, shared=True)
        T_lg = T.line_graph(backtracking=False, shared=True)

        G_lg.ndata['msg'] = th.zeros(G.number_of_edges(), self.d_msg)
        T_lg.ndata['msg'] = th.zeros(T.number_of_edges(), self.d_msg)

        mp_message_fn = fn.copy_src(src='msg', out='msg')
        mp_reduce_fn = fn.reducer.sum(msg='msg', out='sum_msg')
        mp_apply_fn = lambda nodes: {'msg' : self.g1(nodes.data['f_src'], \
                                                     nodes.data['f'], nodes.data['sum_msg'])}
        for i in range(self.n_iterations):
            G_lg.update_all(mp_message_fn, mp_reduce_fn, mp_apply_fn)
            T_lg.update_all(mp_message_fn, mp_reduce_fn, mp_apply_fn)

        readout_message_fn = fn.copy_edge(edge='msg', out='msg')
        readout_reduce_fn = fn.reducer.sum(msg='msg', out='sum_msg')
        readout_apply_fn = lambda nodes: {'x' : self.g2(nodes.data['f'], nodes.data['sum_msg'])}
        G.update_all(readout_message_fn, readout_reduce_fn, readout_apply_fn)
        T.update_all(readout_message_fn, readout_reduce_fn, readout_apply_fn)
