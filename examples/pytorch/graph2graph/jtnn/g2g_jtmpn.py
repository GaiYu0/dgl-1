import dgl.function as fn
import torch as th
import torch.nn as nn
import rdkit.Chem as Chem
import dgl

ELEM_LIST = ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na', 'Ca', 'Fe', 'Al', 'I', 'B', 'K', 'Se', 'Zn', 'H', 'Cu', 'Mn', 'unknown']

ATOM_FDIM = len(ELEM_LIST) + 6 + 5 + 1
BOND_FDIM = 5 
MAX_NB = 10

def copy_src(G, u, uv):
    src, dst = G.edges()
    G.edata[uv] = G.ndata[u][src]
 
def copy_dst(G, v, uv):
    src, dst = G.edges()
    G.edata[uv] = G.ndata[v][dst]

def onek_encoding_unk(x, allowable_set):
    if x not in allowable_set:
        x = allowable_set[-1]
    return [x == s for s in allowable_set]

# Note that during graph decoding they don't predict stereochemistry-related
# characteristics (i.e. Chiral Atoms, E-Z, Cis-Trans).  Instead, they decode
# the 2-D graph first, then enumerate all possible 3-D forms and find the
# one with highest score.
def atom_features(atom):
    return (th.Tensor(onek_encoding_unk(atom.GetSymbol(), ELEM_LIST)
            + onek_encoding_unk(atom.GetDegree(), [0,1,2,3,4,5])
            + onek_encoding_unk(atom.GetFormalCharge(), [-1,-2,1,2,0])
            + [atom.GetIsAromatic()]))

def bond_features(bond):
    bt = bond.GetBondType()
    return (th.Tensor([bt == Chem.rdchem.BondType.SINGLE, bt == Chem.rdchem.BondType.DOUBLE, bt == Chem.rdchem.BondType.TRIPLE, bt == Chem.rdchem.BondType.AROMATIC, bond.IsInRing()]))

def mol2dgl_single(cand_batch):
    cand_graphs = []
    tree_mess_source_edges = [] # map these edges from trees to...
    tree_mess_target_edges = [] # these edges on candidate graphs
    tree_mess_target_nodes = []
    n_nodes = 0
    n_edges = 0
    atom_x = []
    bond_x = []

    for mol, mol_tree, ctr_node_id in cand_batch:
        n_atoms = mol.GetNumAtoms()
        n_bonds = mol.GetNumBonds()

        ctr_node = mol_tree.nodes_dict[ctr_node_id]
        ctr_bid = ctr_node['idx']
        g = dgl.DGLGraph()

        for i, atom in enumerate(mol.GetAtoms()):
            assert i == atom.GetIdx()
            atom_x.append(atom_features(atom))
        g.add_nodes(n_atoms)

        bond_src = []
        bond_dst = []
        for i, bond in enumerate(mol.GetBonds()):
            a1 = bond.GetBeginAtom()
            a2 = bond.GetEndAtom()
            begin_idx = a1.GetIdx()
            end_idx = a2.GetIdx()
            features = bond_features(bond)

            bond_src.append(begin_idx)
            bond_dst.append(end_idx)
            bond_x.append(features)
            bond_src.append(end_idx)
            bond_dst.append(begin_idx)
            bond_x.append(features)

            x_nid, y_nid = a1.GetAtomMapNum(), a2.GetAtomMapNum()
            # Tree node ID in the batch
            x_bid = mol_tree.nodes_dict[x_nid - 1]['idx'] if x_nid > 0 else -1
            y_bid = mol_tree.nodes_dict[y_nid - 1]['idx'] if y_nid > 0 else -1
            if x_bid >= 0 and y_bid >= 0 and x_bid != y_bid:
                if mol_tree.has_edge_between(x_bid, y_bid):
                    tree_mess_target_edges.append((begin_idx + n_nodes, end_idx + n_nodes))
                    tree_mess_source_edges.append((x_bid, y_bid))
                    tree_mess_target_nodes.append(end_idx + n_nodes)
                if mol_tree.has_edge_between(y_bid, x_bid):
                    tree_mess_target_edges.append((end_idx + n_nodes, begin_idx + n_nodes))
                    tree_mess_source_edges.append((y_bid, x_bid))
                    tree_mess_target_nodes.append(begin_idx + n_nodes)

        n_nodes += n_atoms
        g.add_edges(bond_src, bond_dst)
        cand_graphs.append(g)

    return cand_graphs, th.stack(atom_x), \
            th.stack(bond_x) if len(bond_x) > 0 else th.zeros(0), \
            th.LongTensor(tree_mess_source_edges), \
            th.LongTensor(tree_mess_target_edges), \
            th.LongTensor(tree_mess_target_nodes)


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



        

