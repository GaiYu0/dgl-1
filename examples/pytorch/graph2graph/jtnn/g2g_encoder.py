import dgl.function as fn
import torch as th
import torch.nn as nn

def copy_src(G, u, uv):
    src, dst = G.edges()
    G.edata[uv] = G.ndata[u][src]
 
def copy_dst(G, v, uv):
    src, dst = G.edges()
    G.edata[uv] = G.ndata[v][dst]

class G2GEncoder(nn.Module):
    def __init__(self, embeddings, g1_G, g1_T, g2_G, g2_T, d_msgG, d_msgT, n_itersG, n_itersT):
        """
        Parameters
        ----------
        embeddings :
        g1_G : $g_1$ in Eq. (1) for molecule
        g1_T : $g_1$ in Eq. (1) for junction tree
        g2_G : $g_2$ in Eq. (2) for molecule
        g2_T : $g_2$ in Eq. (2) for junction tree
        d_msgG : dimension of graph's message vector in message passing
        d_msgT : dimension of junction tree's message vector in message passing
        n_itersG : number of iterations for graph's message passing
        n_itersT : number of iterations for graph's message passing
        """
        super(G2GEncoder, self).__init__()

        self.embeddings = embeddings
        self.g1_G = g1_G
        self.g1_T = g1_T
        self.g2_G = g2_G
        self.g2_T = g2_T

        self.d_msgG = d_msgG
        self.d_msgT = d_msgT
        self.n_itersG = n_itersG
        self.n_itersT = n_itersT

    def forward(self, G, T):
        device = G.ndata['f'].device
        
        T.ndata['f'] = T.ndata['id'] @ self.embeddings
        copy_src(G, 'f', 'f_src')
        copy_dst(G, 'f', 'f_dst')
        copy_src(T, 'f', 'f_src')
        copy_dst(T, 'f', 'f_dst')

        G_lg = G.line_graph(backtracking=False, shared=True)
        T_lg = T.line_graph(backtracking=False, shared=True)

        G_lg.ndata['msg'] = th.zeros(G.number_of_edges(), self.d_msgG, device=device)
        T_lg.ndata['msg'] = th.zeros(T.number_of_edges(), self.d_msgT, device=device)

        mp_message_fn = fn.copy_src(src='msg', out='msg')
        mp_reduce_fn = fn.reducer.sum(msg='msg', out='sum_msg')
        mp_apply_fn = lambda nodes: {'msg' : self.g1_G(nodes.data['f_src'], \
                                                        nodes.data['f'], nodes.data['sum_msg'])}
        #also part ofg1_G
        for i in range(self.n_itersG):
            G_lg.update_all(mp_message_fn, mp_reduce_fn, mp_apply_fn)

        for i in range(self.n_itersT):
            self.g1_T(T_lg)

        readout_message_fn = fn.copy_edge(edge='msg', out='msg')
        readout_reduce_fn = fn.reducer.sum(msg='msg', out='sum_msg')
        readout_apply_fn = lambda nodes: {'x' : self.g2_G(nodes.data['f'], nodes.data['sum_msg'])}
        G.update_all(readout_message_fn, readout_reduce_fn, readout_apply_fn)
        readout_apply_fn = lambda nodes: {'x' : self.g2_T(nodes.data['f'], nodes.data['sum_msg'])}
        T.update_all(readout_message_fn, readout_reduce_fn, readout_apply_fn)
