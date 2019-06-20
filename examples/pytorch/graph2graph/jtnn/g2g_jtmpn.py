import dgl.function as fn
import torch as th
import torch.nn as nn

def copy_src(G, u, uv):
    src, dst = G.edges()
    G.edata[uv] = G.ndata[u][src]
 
def copy_dst(G, v, uv):
    src, dst = G.edges()
    G.edata[uv] = G.ndata[v][dst]


class g2g_JTMPN(nn.Module):

    def __init__(self, g1_G, g2_G, d_msgG, d_msgT, n_itersG):
        super(g2g_JTMPN, self).__init__()
        self.g1_G = g1_G
        self.g2_G = g2_G
        self.d_msgG = d_msgG
        self.d_msgT = d_msgT
        self.n_itersG = n_itersG
        assert self.d_msgG == self.d_msgT, "msg dimension of G and T not match, cannot integrate JT messages"
    
    def forward(self, candidates_G, Y_T, Y_T_msgs):
        device = candidates_G.ndata['f'].device
        #\TODO(hq) need to do data preprocess for candidates_G
        copy_src(candidates_G, 'f', 'f_src')
        copy_dst(candidates_G, 'f', 'f_dst')

        # Unpacking tree messages augmentation locations
        tree_mess_src_edges, tree_mess_tgt_edges, tree_mess_tgt_nodes = Y_T_msgs 


        #augment candidates_G with junction tree message functions
        candidates_G.edata['alpha'] = th.zeros(candidates_G.number_of_edges(),
                                             self.d_msgT, device=device)
        if tree_mess_src_edges.shape[0] > 0:
            src_u, src_v = tree_mess_src_edges.unbind(1)
            tgt_u, tgt_v = tree_mess_tgt_edges.unbind(1)
            alpha = Y_T.edges[src_u, src_v].data['msg']
            candidates_G.edges[tgt_u, tgt_v].data['alpha'] = alpha
        
        candidates_G_lg = candidates_G.line_graph(backtracking=False, shared=True)
        candidates_G_lg.ndata['msg'] = th.zeros(candidates_G.number_of_edges(), self.d_msgG, device=device)

        # equation (26), (27)
        mp_message_fn = fn.copy_src(src='msg', out='msg')
        mp_reduce_fn = fn.reducer.sum(msg='msg', out='sum_msg')
        mp_apply_fn = lambda nodes: {'msg' : self.g1_G(nodes.data['f_src'], \
                                                        nodes.data['f'], nodes.data['sum_msg'] + nodes.data['alpha'])}
        
        # DEBUG
        #print("number of edges is ", candidates_G_lg.number_of_nodes())
        #print(candidates_G_lg.ndata['f_src'].size())
        #print(candidates_G_lg.ndata['f'].size())
        #print(candidates_G_lg.ndata['alpha'].size())
        #raise NotImplementedError
        for _ in range(self.n_itersG):
            candidates_G_lg.update_all(mp_message_fn, mp_reduce_fn, mp_apply_fn)

        # equation (28)
        readout_message_fn = fn.copy_edge(edge='msg', out='msg')
        readout_reduce_fn = fn.reducer.sum(msg='msg', out='sum_msg')
        readout_apply_fn = lambda nodes: {'x' : self.g2_G(nodes.data['f'], nodes.data['sum_msg'])}

        candidates_G.update_all(readout_message_fn, readout_reduce_fn, readout_apply_fn)



        

