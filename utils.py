import sys                                          
import numpy as np                                
import networkx as nx                              
import scipy.sparse as sp                           
import torch                                        
import torch.nn as nn                            
import scipy.io as sio                            
import random                                      
import dgl                                       
import os
from tqdm import tqdm
import torch.nn.functional as F
from collections import OrderedDict  


def parse_skipgram(fname):
    with open(fname) as f:                         
        toks = list(f.read().split())             
    nb_nodes = int(toks[0])                       
    nb_features = int(toks[1])                     
    ret = np.empty((nb_nodes, nb_features))       
    it = 2                                       
    for i in range(nb_nodes):                     
        cur_nd = int(toks[it]) - 1               
        it += 1                                   
        for j in range(nb_features):            
            cur_ft = float(toks[it])             
            ret[cur_nd][j] = cur_ft               
            it += 1                              
    return ret                                    


def process_tu(data, nb_nodes):
    nb_graphs = len(data)                         
    ft_size = data.num_features                    
    features = np.zeros((nb_graphs, nb_nodes, ft_size))  
    adjacency = np.zeros((nb_graphs, nb_nodes, nb_nodes))  
    labels = np.zeros(nb_graphs)                   
    sizes = np.zeros(nb_graphs, dtype=np.int32)    
    masks = np.zeros((nb_graphs, nb_nodes))      
    for g in range(nb_graphs):                     
        sizes[g] = data[g].x.shape[0]              
        features[g, :sizes[g]] = data[g].x         
        labels[g] = data[g].y[0]                   
        masks[g, :sizes[g]] = 1.0                  
        e_ind = data[g].edge_index                 
        coo = sp.coo_matrix((np.ones(e_ind.shape[1]), (e_ind[0, :], e_ind[1, :])),
                            shape=(nb_nodes, nb_nodes))  
        adjacency[g] = coo.todense()               
    return features, adjacency, labels, sizes, masks 


def micro_f1(logits, labels):
    preds = torch.round(nn.Sigmoid()(logits))      
    preds = preds.long()                          
    labels = labels.long()                         
    tp = torch.nonzero(preds * labels).shape[0] * 1.0          
    tn = torch.nonzero((preds - 1) * (labels - 1)).shape[0] * 1.0  
    fp = torch.nonzero(preds * (labels - 1)).shape[0] * 1.0    
    fn = torch.nonzero((preds - 1) * labels).shape[0] * 1.0    
    prec = tp / (tp + fp)                         
    rec = tp / (tp + fn)                           
    f1 = (2 * prec * rec) / (prec + rec)          
    return f1


def adj_to_bias(adj, sizes, nhood=1):
    nb_graphs = adj.shape[0]                       
    mt = np.empty(adj.shape)                       
    for g in range(nb_graphs):                    
        mt[g] = np.eye(adj.shape[1])               
        for _ in range(nhood):                   
            mt[g] = np.matmul(mt[g], (adj[g] + np.eye(adj.shape[1])))  
        for i in range(sizes[g]):                 
            for j in range(sizes[g]):
                if mt[g][i][j] > 0.0:              
                    mt[g][i][j] = 1.0             
    return -1e9 * (1.0 - mt)                       


def parse_index_file(filename):
    index = []                                   
    for line in open(filename):                    
        index.append(int(line.strip()))            
    return index


def sample_mask(idx, l):
    mask = np.zeros(l)                            
    mask[idx] = 1                                  
    return np.array(mask, dtype=np.bool)         


def sparse_to_tuple(sparse_mx, insert_batch=False):
    def to_tuple(mx):
        if not sp.isspmatrix_coo(mx):            
            mx = mx.tocoo()
        if insert_batch:                          
            coords = np.vstack((np.zeros(mx.row.shape[0]), mx.row, mx.col)).transpose()
            values = mx.data
            shape = (1,) + mx.shape
        else:
            coords = np.vstack((mx.row, mx.col)).transpose()
            values = mx.data
            shape = mx.shape
        return coords, values, shape

    if isinstance(sparse_mx, list):               
        for i in range(len(sparse_mx)):
            sparse_mx[i] = to_tuple(sparse_mx[i])
    else:
        sparse_mx = to_tuple(sparse_mx)
    return sparse_mx


def standardize_data(f, train_mask):
    f = f.todense()                             
    mu = f[train_mask == True, :].mean(axis=0)   
    sigma = f[train_mask == True, :].std(axis=0) 
    f = f[:, np.squeeze(np.array(sigma > 0))]    
    mu = f[train_mask == True, :].mean(axis=0)   
    sigma = f[train_mask == True, :].std(axis=0)
    f = (f - mu) / sigma                         
    return f


def preprocess_features(features):
    rowsum = np.array(features.sum(1), dtype=np.float32)  
    r_inv = np.power(rowsum, -1).flatten()       
    r_inv[np.isinf(r_inv)] = 0.                  
    r_mat_inv = sp.diags(r_inv)                  
    features = r_mat_inv.dot(features)           
    return features.todense(), sparse_to_tuple(features)  


def normalize_adj(adj):
    adj = sp.coo_matrix(adj)                     
    rowsum = np.array(adj.sum(1))                
    d_inv_sqrt = np.power(rowsum, -0.5).flatten() 
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.       
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)        
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()  


def preprocess_adj(adj):
    adj_normalized = normalize_adj(adj + sp.eye(adj.shape[0]))
    return sparse_to_tuple(adj_normalized)


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def adj_to_dict(adj, hop=1, min_len=8):
    adj = np.array(adj.todense(), dtype=np.float64)  
    num_node = adj.shape[0]                       
    adj_diff = adj                                
    if hop > 1:                                   
        for _ in range(hop - 1):
            adj_diff = adj_diff.dot(adj)
    dict = {}                                     
    for i in range(num_node):                    
        dict[i] = []
        for j in range(num_node):
            if adj_diff[i, j] > 0:               
                dict[i].append(j)
    final_dict = dict.copy()
    for i in range(num_node):                     
        while len(final_dict[i]) < min_len:
            final_dict[i].append(random.choice(dict[random.choice(dict[i])]))
    return final_dict


def dense_to_one_hot(labels_dense, num_classes):

    num_labels = labels_dense.shape[0]            
    index_offset = np.arange(num_labels) * num_classes  
    labels_one_hot = np.zeros((num_labels, num_classes))
    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1  
    return labels_one_hot


def load_mat(dataset, train_rate=0.3, val_rate=0.1):

    data = sio.loadmat("./dataset/{}.mat".format(dataset)) 
    label = data['Label'] if ('Label' in data) else data['gnd']  
    attr = data['Attributes'] if ('Attributes' in data) else data['X']  
    network = data['Network'] if ('Network' in data) else data['A']  

    adj = sp.csr_matrix(network)                 
    feat = sp.lil_matrix(attr)                  

    labels = np.squeeze(np.array(data['Class'], dtype=np.int64) - 1)  
    num_classes = np.max(labels) + 1              
    labels = dense_to_one_hot(labels, num_classes)  

    ano_labels = np.squeeze(np.array(label))     
    if 'str_anomaly_label' in data:             
        str_ano_labels = np.squeeze(np.array(data['str_anomaly_label']))
        attr_ano_labels = np.squeeze(np.array(data['attr_anomaly_label']))
    else:
        str_ano_labels = None
        attr_ano_labels = None

    num_node = adj.shape[0]                       
    num_train = int(num_node * train_rate)        
    num_val = int(num_node * val_rate)            
    all_idx = list(range(num_node))               
    random.shuffle(all_idx)                      
    idx_train = all_idx[:num_train]               
    idx_val = all_idx[num_train:num_train + num_val]  
    idx_test = all_idx[num_train + num_val:]     

    return adj, feat, labels, idx_train, idx_val, idx_test, ano_labels, str_ano_labels, attr_ano_labels


def adj_to_dgl_graph(adj):
    nx_graph = nx.from_scipy_sparse_matrix(adj)  
    dgl_graph = dgl.DGLGraph(nx_graph)            
    return dgl_graph


def generate_rwr_subgraph(dgl_graph, subgraph_size):
    all_idx = list(range(dgl_graph.number_of_nodes()))  
    reduced_size = subgraph_size - 1             
    traces = dgl.contrib.sampling.random_walk_with_restart(
        dgl_graph, all_idx, restart_prob=1, max_nodes_per_seed=subgraph_size * 3)
    subv = []                                     
    for i, trace in enumerate(traces):           
        subv.append(torch.unique(torch.cat(trace), sorted=False).tolist())  
        retry_time = 0
        while len(subv[i]) < reduced_size:      
            cur_trace = dgl.contrib.sampling.random_walk_with_restart(
                dgl_graph, [i], restart_prob=0.9, max_nodes_per_seed=subgraph_size * 5)
            subv[i] = torch.unique(torch.cat(cur_trace[0]), sorted=False).tolist()
            retry_time += 1
            if (len(subv[i]) <= 2) and (retry_time > 10):  
                subv[i] = (subv[i] * reduced_size)
        subv[i] = subv[i][:reduced_size]       
        subv[i].append(i)                     
    return subv                                
                              

def get_first_adj(dgl_graph, adj, subgraph_size):
    
    all_idx = list(range(dgl_graph.number_of_nodes()))
    subgraphs = []
    adj = np.array(adj.todense()).squeeze()

    for node_id in all_idx:
        first_adj = np.where(adj[node_id] == 1)
        first_adj = list(first_adj[0])  


        if len(first_adj) < subgraph_size - 1:
            subgraphs.append(first_adj)
            first_adj.append(node_id)
            subgraphs[node_id].extend(
                list(np.random.choice(first_adj, subgraph_size - len(first_adj) - 1, replace=True)))
        else:
            subgraphs.append(list(np.random.choice(first_adj, subgraph_size - 1, replace=False)))
        subgraphs[node_id].append(node_id)
    return subgraphs



def get_second_adj(dgl_graph, adj, subgraph_size):
    all_idx = list(range(dgl_graph.number_of_nodes()))
    subgraphs = []

    adj_2 = adj.dot(adj)
    adj = np.array(adj.todense())
    adj_2 = np.array(adj_2.todense())

    row, col = np.diag_indices_from(adj_2)
    zeros = np.zeros(adj_2.shape[0])
    adj_2[row, col] = np.array(zeros)  

    adj = adj.squeeze()
    adj_2 = adj_2.squeeze()

    for node_id in all_idx:

        first_adj = np.where(adj[node_id] == 1)
        second_adj = np.where(adj_2[node_id] != 0)

        first_adj = first_adj[0].tolist()
        second_adj = second_adj[0].tolist()


        if len(first_adj) < subgraph_size // 2:
            subgraphs.append(list(np.random.choice(first_adj, subgraph_size // 2, replace=True)))


            if len(second_adj) == 0:
                first_adj.append(node_id)
                subgraphs[node_id].extend(list(np.random.choice(first_adj, (subgraph_size - 1) // 2, replace=True)))
            elif len(second_adj) < (subgraph_size - 1) // 2:
                subgraphs[node_id].extend(list(np.random.choice(second_adj, (subgraph_size - 1) // 2, replace=True)))
            else:
                subgraphs[node_id].extend(list(np.random.choice(second_adj, (subgraph_size - 1) // 2, replace=False)))

        else:
            if len(second_adj) == 0:
                first_adj.append(node_id)
                if len(first_adj) < subgraph_size - 1:
                    subgraphs.append(list(np.random.choice(first_adj, (subgraph_size - 1), replace=True)))
                else:
                    subgraphs.append(list(np.random.choice(first_adj, (subgraph_size - 1), replace=False)))
            elif len(second_adj) < (subgraph_size - 1) // 2:
                subgraphs.append(list(np.random.choice(first_adj, subgraph_size // 2, replace=False)))
                subgraphs[node_id].extend(list(np.random.choice(second_adj, (subgraph_size - 1) // 2, replace=True)))
            else:
                subgraphs.append(list(np.random.choice(first_adj, subgraph_size // 2, replace=False)))
                subgraphs[node_id].extend(list(np.random.choice(second_adj, (subgraph_size - 1) // 2, replace=False)))


        subgraphs[node_id].append(node_id)

    return subgraphs

def get_second_adj_6(dgl_graph, adj, subgraph_size_1, subgraph_size_2):
    all_idx = list(range(dgl_graph.number_of_nodes()))
    adj = np.array(adj.todense())

    row, col = np.diag_indices_from(adj)
    zeros = np.zeros(adj.shape[0])
    adj[row, col] = np.array(zeros)
    adj = adj.squeeze()
    subgraphs_1 = []
    subgraphs_2 = []

    for target_node in all_idx:
        first_adj = np.where(adj[target_node] == 1)
        first_adj = list(first_adj[0])

        if len(first_adj) == 0:
            first_adj.append(target_node)
            subgraphs_1.append(list(np.random.choice(first_adj, subgraph_size_1, replace=True)))
            subgraphs_2.append(list(np.random.choice(first_adj, subgraph_size_2, replace=True)))
        else:
            cur_subgraph_1 = []
            cur_subgraph_2 = []
            if len(first_adj) < subgraph_size_1 - 1:
                cur_subgraph_1.extend(first_adj)
                cur_subgraph_1.extend(list(np.random.choice(first_adj, subgraph_size_1 - len(first_adj) - 1, replace=True)))
            else:
                cur_subgraph_1.extend(list(np.random.choice(first_adj, subgraph_size_1 - 1, replace=False)))


            subgraphs_1_rest = list(OrderedDict.fromkeys(cur_subgraph_1.copy()))
            subgraphs_1_rest = [i for i in subgraphs_1_rest if i != target_node]


            second_adj_list = []
            for ego_node in subgraphs_1_rest:
                second_adj = np.where(adj[ego_node] == 1)[0]
                second_adj_list.extend(second_adj)


            second_adj_rest = list(OrderedDict.fromkeys(second_adj_list))
            second_adj_rest = [i for i in second_adj_rest if i != target_node]

            cur_subgraph_2.extend(cur_subgraph_1)


            if len(second_adj_rest) == 0:
                first_adj.append(target_node)
                cur_subgraph_2.extend(list(np.random.choice(first_adj, subgraph_size_2 - subgraph_size_1, replace=True)))
            elif len(second_adj_rest) < (subgraph_size_2 - subgraph_size_1):
                cur_subgraph_2.extend(second_adj_rest)
                cur_subgraph_2.extend(list(np.random.choice(second_adj_rest, subgraph_size_2 - subgraph_size_1 - len(second_adj_rest), replace=True)))
            else:
                cur_subgraph_2.extend(list(np.random.choice(second_adj_rest, (subgraph_size_2 - subgraph_size_1), replace=False)))

            cur_subgraph_1.append(target_node)
            cur_subgraph_2.append(target_node)
            subgraphs_1.append(cur_subgraph_1)
            subgraphs_2.append(cur_subgraph_2)

    return subgraphs_1, subgraphs_2


def generate_subgraph(args, dgl_graph, A, subgraph_size_1, subgraph_size_2):
    if args.subgraph_mode == 'random':
        subgraphs_1 = generate_rwr_subgraph(dgl_graph, subgraph_size_1)
        subgraphs_2 = generate_rwr_subgraph(dgl_graph, subgraph_size_2)

    elif args.subgraph_mode == '1+1':
        subgraphs_1 = get_first_adj(dgl_graph, A, subgraph_size_1)
        subgraphs_2 = get_first_adj(dgl_graph, A, subgraph_size_2)

    elif args.subgraph_mode == '1+2':
        subgraphs_1 = get_first_adj(dgl_graph, A, subgraph_size_1)
        subgraphs_2 = get_second_adj(dgl_graph, A, subgraph_size_2)

    elif args.subgraph_mode == '1+near':
        subgraphs_1, subgraphs_2 = get_second_adj_6(dgl_graph, A, subgraph_size_1, subgraph_size_2)

    else:
        raise NotImplementedError

    return subgraphs_1, subgraphs_2


def register_topology(dataset, adj):
    MAX_HOP = 100
    topo_file = f"./dataset/topology_dist/{dataset.lower()}_padding.pt"
    exist = os.path.isfile(topo_file)

    if not exist:
        nx_graph = nx.from_scipy_sparse_matrix(adj)
        node_num = adj.shape[0]
        generator = dict(nx.shortest_path_length(nx_graph))
        topology_dist = torch.zeros((node_num, node_num))
        mask = torch.zeros((node_num, node_num)).bool()


        for i in tqdm(range(0, node_num)):
            for j in range(0, node_num):
                if j in generator[i].keys():
                    topology_dist[i][j] = generator[i][j]
                else:
                    topology_dist[i][j] = MAX_HOP
                    mask[i][j] = True  

        torch.save(topology_dist, topo_file)
    else:
        topology_dist = torch.load(topo_file)

    return topology_dist


def gen_batch_topology_dist(full_topology_dist, node_idx1, node_idx2):
    batch_size = node_idx1.shape[0]
    
    batch_subpology_dist = [
        full_topology_dist.index_select(dim=0, index=node_idx1[i]).  
        index_select(dim=1, index=node_idx2[i])  
        for i in range(batch_size) 
    ]

    batch_subpology_dist = torch.stack(batch_subpology_dist)

    return batch_subpology_dist


def matrix_diag(diagonal):
    N = diagonal.shape[-1]
    shape = diagonal.shape[:-1] + (N, N)
    device, dtype = diagonal.device, diagonal.dtype

    result = torch.zeros(shape, dtype=dtype, device=device)

    indices = torch.arange(result.numel(), device=device).reshape(shape)
    indices = indices.diagonal(dim1=-2, dim2=-1)

    result.view(-1)[indices] = diagonal

    return result


def similarity(reps1, reps2):
    reps1_unit = F.normalize(reps1, dim=-1)
    reps2_unit = F.normalize(reps2, dim=-1)

    if len(reps1.shape) == 2:
        sim_mat = torch.einsum("ik,jk->ij", [reps1_unit, reps2_unit])
    elif len(reps1.shape) == 3:
        sim_mat = torch.einsum('bik,bjk->bij', [reps1_unit, reps2_unit])
    else:
        print(f"{len(reps1.shape)} dimension tensor is not supported for this function!")


    return sim_mat


def Sinkhorn(out1, avg_out1, out2, avg_out2,
             Sinkhorn_iter_times=5, lamb=20, rescale_ratio=None):
    cost_matrix = 1 - similarity(out1, out2)
    if rescale_ratio is not None:
        cost_matrix = cost_matrix * rescale_ratio

    with torch.no_grad():
        r = torch.bmm(out1, avg_out2.transpose(1, 2))  
        r[r <= 0] = 1e-8  
        r = r / r.sum(dim=1, keepdim=True)  

        c = torch.bmm(out2, avg_out1.transpose(1, 2))  
        c[c <= 0] = 1e-8 
        c = c / c.sum(dim=1, keepdim=True)  

        P = torch.exp(-1 * lamb * cost_matrix)

        u = (torch.ones_like(c) / c.shape[1])

        for i in range(Sinkhorn_iter_times):
            v = torch.div(r, torch.bmm(P, u))  
            u = torch.div(c, torch.bmm(P.transpose(1, 2), v))  

        u = u.squeeze(dim=-1)
        v = v.squeeze(dim=-1)


        transport_matrix = torch.bmm(torch.bmm(matrix_diag(v), P), matrix_diag(u))

    assert cost_matrix.shape == transport_matrix.shape

    S = torch.mul(transport_matrix, 1 - cost_matrix).sum(dim=1).sum(dim=1, keepdim=True)
    return S



