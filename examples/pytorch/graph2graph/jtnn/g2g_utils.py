def copy_src(G, u, uv):
    src, dst = G.edges()
    G.edata[uv] = G.ndata[u][src]
 
def copy_dst(G, v, uv):
    src, dst = G.edges()
    G.edata[uv] = G.ndata[u][dst]
