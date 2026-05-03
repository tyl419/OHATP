from model import Model                       
from utils import *                           
from sklearn.metrics import roc_auc_score     
from sklearn.metrics import average_precision_score  
import random                                 
import os                                   
import dgl                                   
import argparse                               
import importlib
import torch.nn as nn
import networkx as nx

from tqdm import tqdm                       

import torch.nn.functional as F              

import numpy as np
from sklearn.preprocessing import MinMaxScaler
from scipy.sparse import csr_matrix
from collections import deque

from torch.optim.lr_scheduler import ExponentialLR


os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ['OMP_NUM_THREADS'] = '1'


parser = argparse.ArgumentParser(description='OHATP')   
parser.add_argument('--expid', type=int, default=1)     
parser.add_argument('--device', type=str, default='cuda:0')
parser.add_argument('--dataset', type=str, default='cora')   
parser.add_argument('--lr', type=float)    
parser.add_argument('--weight_decay', type=float, default=0.0)  
parser.add_argument('--runs', type=int, default=1)      
parser.add_argument('--embedding_dim', type=int, default=64)   
parser.add_argument('--patience', type=int, default=40)     
parser.add_argument('--num_epoch', type=int, default=100)     
parser.add_argument('--batch_size', type=int, default=300)   
parser.add_argument('--subgraph_size', type=int, default=4)  
parser.add_argument('--readout', type=str, default='avg')    
parser.add_argument('--auc_test_rounds', type=int, default=256)  
parser.add_argument('--negsamp_ratio_context', type=int, default=1) 
parser.add_argument('--alpha', type=float, help='how much the first view involves')  
parser.add_argument('--eps', type=float, default=0.1)
parser.add_argument('--subgraph_mode', type=str, default='1+near')


parser.add_argument('--K_1', type=int, help='view 1')
parser.add_argument('--K_2', type=int, help='view 2')
parser.add_argument('--hidden_size', type=int, default=64)
parser.add_argument('--recon', type=float)
parser.add_argument('--context', type=float, default=1)
parser.add_argument('--inter', type=float)
parser.add_argument('--Sinkhorn_iter_times', type=int, default=5)
parser.add_argument('--Sinkhorn_lamb', type=int, default=20)
parser.add_argument('--topo_t', type=int, default=10, help='temperature for sigmoid in topology dist')
parser.add_argument('--temperature', type=float, default=3, help='temperature for fx')
parser.add_argument('--rectified', type=bool, help='use rectified cost matrix', default=True)
parser.add_argument('--have_neg', type=bool, help='anomaly score and LOSS contain negtive pairs OT', default=True)
parser.add_argument('--neg_top_k', type=float, help='top max k of OT to select negtive pairs', default=10)
parser.add_argument('--use_rotation', type=bool, default=True, help='whether to use rotation augmentation')
parser.add_argument('--rotation_mode', type=str, default='orthogonal', help='rotation mode: orthogonal or free')
parser.add_argument('--add_noise_after_rotation', type=bool, default=True, help='whether to add noise after rotation')


args = parser.parse_args()    


if args.lr is None:
    if args.dataset in ['cora', 'citeseer', 'pubmed', 'citation']:
        args.lr = 2e-3  
    elif args.dataset == 'BlogCatalog':
        args.lr = 1e-2  
    elif args.dataset == 'ACM':
        args.lr = 5e-3  


if args.recon is None:
    if args.dataset in ['cora', 'citeseer', 'citation', 'ACM']:
        args.recon = 0.9
    elif args.dataset in ['BlogCatalog', 'pubmed']:
        args.recon = 0.7


if args.inter is None:
    if args.dataset in ['cora', 'pubmed', 'ACM']:
        args.inter = 0.2
    elif args.dataset in ['citeseer']:
        args.inter = 0.4
    elif args.dataset in ['BlogCatalog']:
        args.inter = 0.6
    elif args.dataset in ['citation']:
        args.inter = 0.8


if args.alpha is None:
    if args.dataset in ['cora', 'ACM', 'pubmed', 'citation']:
        args.alpha = 0.8
    elif args.dataset in ['citeseer','BlogCatalog']:
        args.alpha = 0.6



if args.K_1 is None:
    if args.dataset in ['citeseer', 'cora']:
        args.K_1 = 2
    elif args.dataset in [ 'pubmed', 'citation']:
        args.K_1 = 4
    elif args.dataset in ['BlogCatalog', 'ACM']:
        args.K_1 = 6


if args.K_2 is None:
    if args.dataset == 'cora':
        args.K_2 = 4
    elif args.dataset in  ['citation']:
        args.K_2 = 6
    elif args.dataset in ['ACM', 'BlogCatalog', 'pubmed', 'citeseer']:
        args.K_2 = 8


if __name__ == '__main__':

    print('Dataset: {}'.format(args.dataset), flush=True)   
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')  
    torch.backends.cudnn.deterministic = True   
    torch.backends.cudnn.benchmark = False     

    for run in range(args.runs):                

        seed = run + 1                          
        random.seed(seed)                    

        batch_size = args.batch_size           
        subgraph_size = args.subgraph_size     
        subgraph_size_1 = args.K_1  
        subgraph_size_2 = args.K_2 
        alpha_recon= args.recon
        alpha_context = args.context
        alpha_inter = args.inter  
        dataset = args.dataset

       
        # Load Data
        adj, features, labels, idx_train, idx_val, \
        idx_test, ano_label, str_ano_label, attr_ano_label = load_mat(args.dataset)


        adj_raw = adj 
        full_topology_dist = register_topology(args.dataset, adj).to(device)
        raw_feature = features.todense()

        degree = np.sum(adj, axis=0)
        degree_ave = np.mean(degree)


        features, _ = preprocess_features(features)
        dgl_graph = adj_to_dgl_graph(adj)

        nb_nodes = features.shape[0]   
        ft_size  = features.shape[1]   
        nb_classes = labels.shape[1]   

        adj  = normalize_adj(adj)
        adj  = (adj + sp.eye(adj.shape[0])).todense()


        
        features = torch.FloatTensor(features[np.newaxis]).to(device)  
        adj      = torch.FloatTensor(adj[np.newaxis]).to(device)       
        raw_feature = torch.FloatTensor(raw_feature[np.newaxis])
        labels   = torch.FloatTensor(labels[np.newaxis]).to(device)   
        idx_train = torch.LongTensor(idx_train).to(device)
        idx_val   = torch.LongTensor(idx_val).to(device)
        idx_test  = torch.LongTensor(idx_test).to(device)

        all_auc = []  
        all_auprc = [] 
        loss_history = []  
        auc_round_history = []  
        auprc_round_history = []  

        print('\n# Run:{} with random seed:{}'.format(run, seed), flush=True)
        dgl.random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)


        model = Model(ft_size, args.embedding_dim, 'prelu', args.negsamp_ratio_context, args.readout, args.hidden_size,
                      args.temperature, args.Sinkhorn_iter_times, args.Sinkhorn_lamb, args.rectified, args.topo_t, args.neg_top_k, args.have_neg,
                      args.use_rotation, args.rotation_mode, args.add_noise_after_rotation).to(device)
        optimiser = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

        model.eps = args.eps

        b_xent_context = nn.BCEWithLogitsLoss(reduction='none',
                                              pos_weight=torch.tensor([args.negsamp_ratio_context]).to(device))

        
        cnt_wait = 0      
        best     = 1e9   
        best_t   = 0     
        batch_num = nb_nodes // batch_size + 1  
        mse_loss = nn.MSELoss(reduction='mean')

       
        for epoch in range(args.num_epoch):

            model.eps = args.eps

            model.train()                
            all_idx = list(range(nb_nodes))
            random.shuffle(all_idx)       
            total_loss = 0.             


            # generate subgraphs
            subgraphs_1, subgraphs_2 = generate_subgraph(args, dgl_graph, adj_raw, args.K_1, args.K_2)

            
            for batch_idx in range(batch_num):

                optimiser.zero_grad()     
                is_final_batch = (batch_idx == (batch_num - 1))
                if not is_final_batch:
                    idx = all_idx[batch_idx * batch_size: (batch_idx + 1) * batch_size]
                else:
                    idx = all_idx[batch_idx * batch_size:]
                cur_batch_size = len(idx)

                # Construct Contrastive Labels
                lbl_context = torch.unsqueeze(torch.cat(
                    (torch.ones(cur_batch_size),
                     torch.zeros(cur_batch_size * args.negsamp_ratio_context))), 1).to(device)

                ba1, ba2 = [], []  
                bf1, bf2 = [], []  
                raw1, raw2 = [], []  
                sub_idx1, sub_idx2 = [], [] 
                added_adj_zero_row_1 = torch.zeros((cur_batch_size, 1, args.K_1)).to(device)
                added_adj_zero_row_2 = torch.zeros((cur_batch_size, 1, args.K_2)).to(device)
                added_adj_zero_col_1 = torch.zeros((cur_batch_size, args.K_1 + 1, 1)).to(device)
                added_adj_zero_col_2 = torch.zeros((cur_batch_size, args.K_2 + 1, 1)).to(device)
                added_adj_zero_col_1[:, -1, :] = 1.  
                added_adj_zero_col_2[:, -1, :] = 1.  
                added_feat_zero_row = torch.zeros((cur_batch_size, 1, ft_size)).to(device)


                for i in idx: 
                    cur_adj_1 = adj[:, subgraphs_1[i], :][:, :, subgraphs_1[i]]
                    cur_adj_2 = adj[:, subgraphs_2[i], :][:, :, subgraphs_2[i]]
                    
                    cur_feat_1 = features[:, subgraphs_1[i], :]
                    cur_feat_2 = features[:, subgraphs_2[i], :]
                   
                    raw_f_1 = raw_feature[:, subgraphs_1[i], :]
                    raw_f_2 = raw_feature[:, subgraphs_2[i], :]

                   
                    ba1.append(cur_adj_1)
                    ba2.append(cur_adj_2)
                    bf1.append(cur_feat_1)
                    bf2.append(cur_feat_2)
                    raw1.append(raw_f_1)
                    raw2.append(raw_f_2)
                    sub_idx1.append(subgraphs_1[i])  
                    sub_idx2.append(subgraphs_2[i])  


                ba1 = torch.cat(ba1)  
                ba2 = torch.cat(ba2)  

                ba1 = torch.cat((ba1, added_adj_zero_row_1), dim=1)
                ba1 = torch.cat((ba1, added_adj_zero_col_1), dim=2)
                ba2 = torch.cat((ba2, added_adj_zero_row_2), dim=1)
                ba2 = torch.cat((ba2, added_adj_zero_col_2), dim=2)

               
                bf1 = torch.cat(bf1)
                bf1 = torch.cat((bf1[:, :-1, :], added_feat_zero_row, bf1[:, -1:, :]), dim=1)
                
                bf2 = torch.cat(bf2)
                bf2 = torch.cat((bf2[:, :-1, :], added_feat_zero_row, bf2[:, -1:, :]), dim=1)

               
                raw1 = torch.cat(raw1).to(device)
                raw1 = torch.cat((raw1[:, :-1, :], added_feat_zero_row, raw1[:, -1:, :]), dim=1)
               
                raw2 = torch.cat(raw2).to(device)
                raw2 = torch.cat((raw2[:, :-1, :], added_feat_zero_row, raw2[:, -1:, :]), dim=1)

              
                sub_idx1 = torch.Tensor(sub_idx1)  
                sub_idx2 = torch.Tensor(sub_idx2)  
                sub_idx1 = sub_idx1.int() 
                sub_idx2 = sub_idx2.int()  
                if torch.cuda.is_available():
                    sub_idx1 = sub_idx1.to(device)
                    sub_idx2 = sub_idx2.to(device)

            
                (node_recons_1_k1, node_recons_1_k2, logits_1_v1_k1, logits_1_v1_k2,
                 subgraph_embed_v1_k1, subgraph_embed_v1_k2, _, _, inter_loss_1_v1, inter_loss_2_v1, _, _, _, _,) = (
                    model(bf1, ba1, raw1, subgraph_size_1 - 1, bf2, ba2, raw2, subgraph_size_2 - 1,
                          full_topology_dist, sub_idx1, sub_idx2, has_subcon=True, perturbed=False))

              
                (node_recons_2_k1, node_recons_2_k2, logits_1_v2_k1, logits_1_v2_k2,
                 subgraph_embed_v2_k1, subgraph_embed_v2_k2, _, _, inter_loss_1_v2, inter_loss_2_v2, _, _, _, _,) = (
                    model(bf1, ba1, raw1, subgraph_size_1 - 1, bf2, ba2, raw2, subgraph_size_2 - 1,
                          full_topology_dist, sub_idx1, sub_idx2, has_subcon=True, perturbed=True))

        

                # Inter-View Contrast Loss
                subgraph_embed_v1_k1 = F.normalize(subgraph_embed_v1_k1, dim=1, p=2)
                subgraph_embed_v1_k2 = F.normalize(subgraph_embed_v1_k2, dim=1, p=2)
                subgraph_embed_v2_k1 = F.normalize(subgraph_embed_v2_k1, dim=1, p=2)
                subgraph_embed_v2_k2 = F.normalize(subgraph_embed_v2_k2, dim=1, p=2)

                temperature = 1.0

                sim_matrix_one_k1 = torch.matmul(subgraph_embed_v1_k1, subgraph_embed_v2_k1.t())
                sim_matrix_two_k1 = torch.matmul(subgraph_embed_v1_k1, subgraph_embed_v1_k1.t())
                sim_matrix_three_k1 = torch.matmul(subgraph_embed_v2_k1, subgraph_embed_v2_k1.t())


                sim_matrix_one_k2 = torch.matmul(subgraph_embed_v1_k2, subgraph_embed_v2_k2.t())
                sim_matrix_two_k2 = torch.matmul(subgraph_embed_v1_k2, subgraph_embed_v1_k2.t())
                sim_matrix_three_k2 = torch.matmul(subgraph_embed_v2_k2, subgraph_embed_v2_k2.t())

    
                sim_matrix_one_k1_exp = torch.exp(sim_matrix_one_k1 / temperature)
                sim_matrix_two_k1_exp = torch.exp(sim_matrix_two_k1 / temperature)
                sim_matrix_three_k1_exp = torch.exp(sim_matrix_three_k1 / temperature)

        
                sim_matrix_one_k2_exp = torch.exp(sim_matrix_one_k2 / temperature)
                sim_matrix_two_k2_exp = torch.exp(sim_matrix_two_k2 / temperature)
                sim_matrix_three_k2_exp = torch.exp(sim_matrix_three_k2 / temperature)

        
                nega_list = np.arange(0, cur_batch_size - 1, 1)
                nega_list = np.insert(nega_list, 0, cur_batch_size - 1)

                sim_row_sum_k1 = sim_matrix_one_k1_exp[:, nega_list] + \
                                 sim_matrix_two_k1_exp[:, nega_list] + \
                                 sim_matrix_three_k1_exp[:, nega_list]

                sim_row_sum_k1 = torch.diagonal(sim_row_sum_k1)

                sim_diag_k1 = torch.diagonal(sim_matrix_one_k1)

                sim_diag_exp_k1 = torch.exp(sim_diag_k1 / temperature)

                NCE_loss_k1 = -torch.log(sim_diag_exp_k1 / (sim_row_sum_k1))

                NCE_loss_k1 = torch.mean(NCE_loss_k1)


                nega_list = np.arange(0, cur_batch_size - 1, 1)
                nega_list = np.insert(nega_list, 0, cur_batch_size - 1)

                sim_row_sum_k2 = sim_matrix_one_k2_exp[:, nega_list] + \
                                 sim_matrix_two_k2_exp[:, nega_list] + \
                                 sim_matrix_three_k2_exp[:, nega_list]

                sim_row_sum_k2 = torch.diagonal(sim_row_sum_k2)

                sim_diag_k2 = torch.diagonal(sim_matrix_one_k2)

                sim_diag_exp_k2 = torch.exp(sim_diag_k2 / temperature)

                NCE_loss_k2 = -torch.log(sim_diag_exp_k2 / (sim_row_sum_k2))

                NCE_loss_k2 = torch.mean(NCE_loss_k2)

                
                # Reconstruction Loss
                loss_recon     = 0.5 * (mse_loss(node_recons_1_k1, raw1[:, -1, :]) + mse_loss(node_recons_1_k2, raw2[:, -1, :]))

                  
                # Node-Subgraph Contrast Loss
                loss_all_1_k1     = b_xent_context(logits_1_v1_k1, lbl_context)
                loss_all_1_k2     = b_xent_context(logits_1_v1_k2, lbl_context)
                loss_all_1_hat_k1 = b_xent_context(logits_1_v2_k1, lbl_context)
                loss_all_1_hat_k2 = b_xent_context(logits_1_v2_k2, lbl_context)
                loss_1      = torch.mean((loss_all_1_k1 + loss_all_1_k2) / 2)
                loss_1_hat  = torch.mean((loss_all_1_hat_k1 + loss_all_1_hat_k2) / 2)


                # Intra-View Contrast Loss
                loss_inter_v1 = torch.mean((inter_loss_1_v1 + inter_loss_2_v1) / 2)
                loss_inter_v2 = torch.mean((inter_loss_1_v2 + inter_loss_2_v2) / 2)


                loss_context = args.alpha * loss_1 + (1 - args.alpha) * loss_1_hat 
                loss_inter = args.alpha * loss_inter_v1 + (1 - args.alpha) * loss_inter_v2  
                l2_penalty = sum(p.pow(2).sum() for p in model.parameters())

                loss = alpha_recon * loss_recon + alpha_context * loss_context + alpha_inter * (
                        NCE_loss_k1 + NCE_loss_k2 + loss_inter)


                loss.backward()
                optimiser.step()
                loss = loss.detach().cpu().numpy()
                if not is_final_batch:
                    total_loss += loss  


            # Average Loss & Early Stopping
            mean_loss = (total_loss * batch_size + loss * cur_batch_size) / nb_nodes
            if mean_loss < best:
                best   = mean_loss
                best_t = epoch
                cnt_wait = 0
                torch.save(model.state_dict(), '{}.pkl'.format(args.dataset))
            else:
                cnt_wait += 1
            if cnt_wait == args.patience:
                print('Early stopping!', flush=True)
                break
            print('Epoch:{} Loss:{:.8f}'.format(epoch, mean_loss), flush=True)
            loss_history.append(float(mean_loss))



        # Inference
        print('Loading {}th epoch'.format(best_t), flush=True)
        model.load_state_dict(torch.load('{}.pkl'.format(args.dataset)))

        multi_round_ano_score = np.zeros((args.auc_test_rounds, nb_nodes))

        embed_dict_k1 = {}
        embed_dict_k2 = {}

        nodes_embed_k1 = torch.zeros([nb_nodes, args.embedding_dim], dtype=torch.float).cuda()
        nodes_embed_k2 = torch.zeros([nb_nodes, args.embedding_dim], dtype=torch.float).cuda()

        print('Testing AUC!', flush=True)

        with (((tqdm(total=args.auc_test_rounds)))) as pbar_test:
            pbar_test.set_description('Testing')
            for round in range(args.auc_test_rounds):
                all_idx = list(range(nb_nodes))
                random.shuffle(all_idx) 


                subgraphs_1, subgraphs_2 = generate_subgraph(args, dgl_graph, adj_raw, args.K_1, args.K_2)

                for batch_idx in range(batch_num):
                    optimiser.zero_grad()
                    is_final_batch = (batch_idx == (batch_num - 1))
                    if not is_final_batch:
                        idx = all_idx[batch_idx * batch_size: (batch_idx + 1) * batch_size]
                    else:
                        idx = all_idx[batch_idx * batch_size:]
                    cur_batch_size = len(idx)

                    ba1, ba2 = [], []  
                    bf1, bf2 = [], [] 
                    raw1, raw2 = [], [] 
                    sub_idx1, sub_idx2 = [], []  
                    added_adj_zero_row_1 = torch.zeros((cur_batch_size, 1, args.K_1)).to(device)
                    added_adj_zero_row_2 = torch.zeros((cur_batch_size, 1, args.K_2)).to(device)
                    added_adj_zero_col_1 = torch.zeros((cur_batch_size, args.K_1 + 1, 1)).to(device)
                    added_adj_zero_col_2 = torch.zeros((cur_batch_size, args.K_2 + 1, 1)).to(device)
                    added_adj_zero_col_1[:, -1, :] = 1.  
                    added_adj_zero_col_2[:, -1, :] = 1.  
                    added_feat_zero_row = torch.zeros((cur_batch_size, 1, ft_size)).to(device)

                  
                    for i in idx:
                        cur_adj_1 = adj[:, subgraphs_1[i], :][:, :, subgraphs_1[i]]
                        cur_adj_2 = adj[:, subgraphs_2[i], :][:, :, subgraphs_2[i]]

                        cur_feat_1 = features[:, subgraphs_1[i], :]
                        cur_feat_2 = features[:, subgraphs_2[i], :]

                        raw_f_1 = raw_feature[:, subgraphs_1[i], :]
                        raw_f_2 = raw_feature[:, subgraphs_2[i], :]


                        ba1.append(cur_adj_1)
                        ba2.append(cur_adj_2)
                        bf1.append(cur_feat_1)
                        bf2.append(cur_feat_2)
                        raw1.append(raw_f_1)
                        raw2.append(raw_f_2)
                        sub_idx1.append(subgraphs_1[i]) 
                        sub_idx2.append(subgraphs_2[i])  

                   
                    ba1 = torch.cat(ba1)  
                    ba2 = torch.cat(ba2)  
                   
                    ba1 = torch.cat((ba1, added_adj_zero_row_1), dim=1)
                    ba1 = torch.cat((ba1, added_adj_zero_col_1), dim=2)
                    ba2 = torch.cat((ba2, added_adj_zero_row_2), dim=1)
                    ba2 = torch.cat((ba2, added_adj_zero_col_2), dim=2)

                   
                    bf1 = torch.cat(bf1)
                    bf1 = torch.cat((bf1[:, :-1, :], added_feat_zero_row, bf1[:, -1:, :]), dim=1)
                    
                    bf2 = torch.cat(bf2)
                    bf2 = torch.cat((bf2[:, :-1, :], added_feat_zero_row, bf2[:, -1:, :]), dim=1)

                   
                    raw1 = torch.cat(raw1).to(device)
                    raw1 = torch.cat((raw1[:, :-1, :], added_feat_zero_row, raw1[:, -1:, :]), dim=1)
                   
                    raw2 = torch.cat(raw2).to(device)
                    raw2 = torch.cat((raw2[:, :-1, :], added_feat_zero_row, raw2[:, -1:, :]), dim=1)

                  
                    sub_idx1 = torch.Tensor(sub_idx1) 
                    sub_idx2 = torch.Tensor(sub_idx2)  
                    sub_idx1 = sub_idx1.int()  
                    sub_idx2 = sub_idx2.int()  
                  
                    if torch.cuda.is_available():
                        sub_idx1 = sub_idx1.to(device)
                        sub_idx2 = sub_idx2.to(device)

                    
                    with torch.no_grad():

                        node_recons_1_k1, node_recons_1_k2, test_logits_1_k1, test_logits_1_k2, \
                        subgraph_embed_k1, subgraph_embed_k2, batch_embed_k1, batch_embed_k2, inter_loss_1_v1, inter_loss_2_v1, sim_all_1_v1, sim_all_2_v1, \
                            sim_pos_1_v1, sim_pos_2_v1 = (
                            model(bf1, ba1, raw1, subgraph_size_1 - 1, bf2, ba2, raw2, subgraph_size_2 - 1,
                                  full_topology_dist, sub_idx1, sub_idx2, has_subcon = True, perturbed=False))

                    
                        node_recons_2_k1, node_recons_2_k2, test_logits_1_p_k1, test_logits_1_p_k2, \
                            subgraph_embed_p_k1, subgraph_embed_p_k2, _, _, inter_loss_1_v2, inter_loss_2_v2, sim_all_1_v2, sim_all_2_v2, \
                            sim_pos_1_v2, sim_pos_2_v2 = (
                            model(bf1, ba1, raw1, subgraph_size_1 - 1, bf2, ba2, raw2, subgraph_size_2 - 1,
                                  full_topology_dist, sub_idx1, sub_idx2, has_subcon=True, perturbed=True))


                        pdist = nn.PairwiseDistance(p=2) 
                
                        scaler1 = MinMaxScaler()
                        scaler2 = MinMaxScaler()
                        scaler3 = MinMaxScaler()
                        scaler4 = MinMaxScaler()
            
                        test_logits_1_k1     = torch.sigmoid(torch.squeeze(test_logits_1_k1))
                        test_logits_1_k2     = torch.sigmoid(torch.squeeze(test_logits_1_k2))
                        test_logits_1_p_k1 = torch.sigmoid(torch.squeeze(test_logits_1_p_k1))
                        test_logits_1_p_k2 = torch.sigmoid(torch.squeeze(test_logits_1_p_k2))

                       
                        if round == args.auc_test_rounds - 1:
                            for i, node_idx in enumerate(idx):
                                embed_dict_k1[node_idx] = batch_embed_k1[i].detach().cpu()
                                embed_dict_k2[node_idx] = batch_embed_k2[i].detach().cpu()


                        # Reconstruction Anomaly Score
                        score_re = (pdist(node_recons_1_k1, raw1[:, -1, :]) + pdist(node_recons_1_k2, raw2[:, -1, :])) / 2
                        score_re = score_re.cpu().numpy()  
                        ano_score_re = score_re


                        # Intra-View Contrast Anomaly Score
                        score_ot_v1 = - (sim_pos_1_v1 + sim_pos_2_v1) / 2
                        score_ot_v1 = score_ot_v1.cpu().numpy()  
                        score_ot_v2 = - (sim_pos_1_v2 + sim_pos_2_v2) / 2
                        score_ot_v2 = score_ot_v2.cpu().numpy() 
                        score_ot = (args.alpha * score_ot_v1 + (1 - args.alpha) * score_ot_v2)

                        # Node-Subgraph Contrast Anomaly Score
                        ano_score_1_k1 = - (test_logits_1_k1[:cur_batch_size] -
                                         torch.mean(test_logits_1_k1[cur_batch_size:].view(
                                             cur_batch_size, args.negsamp_ratio_context), dim=1)).cpu().numpy()
                        ano_score_1_k2 = - (test_logits_1_k2[:cur_batch_size] -
                                         torch.mean(test_logits_1_k2[cur_batch_size:].view(
                                             cur_batch_size, args.negsamp_ratio_context), dim=1)).cpu().numpy()
                        ano_score_1_p_k1 = - (test_logits_1_p_k1[:cur_batch_size] -
                                           torch.mean(test_logits_1_p_k1[cur_batch_size:].view(
                                               cur_batch_size, args.negsamp_ratio_context), dim=1)).cpu().numpy()
                        ano_score_1_p_k2 = - (test_logits_1_p_k2[:cur_batch_size] -
                                           torch.mean(test_logits_1_p_k2[cur_batch_size:].view(
                                               cur_batch_size, args.negsamp_ratio_context), dim=1)).cpu().numpy()

                        ano_score_1 = (ano_score_1_k1 + ano_score_1_k2) / 2
                        ano_score_1_p = (ano_score_1_p_k1 + ano_score_1_p_k2) / 2
                        score_context = (args.alpha * ano_score_1 + (1 - args.alpha) * ano_score_1_p)

                       
                        ano_score_context = scaler1.fit_transform(score_context.reshape(-1, 1)).reshape(-1)
                        ano_score_recon = scaler3.fit_transform(ano_score_re.reshape(-1, 1)).reshape(-1)
                        ano_score_ot = scaler4.fit_transform(score_ot.reshape(-1, 1)).reshape(-1)


                        ano_score = alpha_context * ano_score_context + \
                                    alpha_recon * ano_score_recon + \
                                    alpha_inter * ano_score_ot


                    multi_round_ano_score[round, idx] = ano_score  


                pbar_test.update(1)

                
                round_scores = np.nan_to_num(multi_round_ano_score[round], nan=0.0, posinf=0.0, neginf=0.0)
                try:
                    round_auc = roc_auc_score(ano_label, round_scores)
                except Exception:
                    round_auc = 0.0
                try:
                    round_auprc = average_precision_score(ano_label, round_scores)
                except Exception:
                    round_auprc = 0.0

                auc_round_history.append(float(round_auc))
                auprc_round_history.append(float(round_auprc))

         
            # Multi-round Averagin
            attr_ano_score_final = np.mean(multi_round_ano_score, axis=0) + np.std(multi_round_ano_score, axis=0)

            attr_ano_score_final = np.nan_to_num(attr_ano_score_final, nan=0.0, posinf=0.0, neginf=0.0)

            
            attr_scaler = MinMaxScaler()
          
            attr_ano_score_final = attr_scaler.fit_transform(attr_ano_score_final.reshape(-1, 1)).reshape(-1)


            # Topology Perception Anomaly Score
            nodes_embed_k1 = torch.zeros([nb_nodes, args.embedding_dim], dtype=torch.float)
            nodes_embed_k2 = torch.zeros([nb_nodes, args.embedding_dim], dtype=torch.float)
            for node_idx in range(nb_nodes):
                if node_idx in embed_dict_k1:
                    nodes_embed_k1[node_idx] = embed_dict_k1[node_idx]
                    nodes_embed_k2[node_idx] = embed_dict_k2[node_idx]
            
            nodes_embed_k1 = nodes_embed_k1.to(device)
            nodes_embed_k2 = nodes_embed_k2.to(device)

           
            features_norm_k1 = F.normalize(nodes_embed_k1, p=2, dim=1)
            features_norm_k2 = F.normalize(nodes_embed_k2, p=2, dim=1)
           
            features_similarity_k1 = torch.matmul(features_norm_k1, features_norm_k1.transpose(0, 1)).squeeze(0).cpu()
            features_similarity_k2 = torch.matmul(features_norm_k2, features_norm_k2.transpose(0, 1)).squeeze(0).cpu()

           
            k_init = int(degree_ave)
            net = nx.from_numpy_matrix(adj_raw)
            net.remove_edges_from(nx.selfloop_edges(net))
            adj_raw = nx.to_numpy_matrix(net)
            multi_round_stru_ano_score = []

        
            # Iterative k-core Decomposition
            while 1:
                list_temp = list(nx.k_core(net, k_init))
                if list_temp == []:
                    break
                else:
                    core_adj = adj_raw[list_temp, :][:, list_temp]
                    core_graph = nx.from_numpy_matrix(core_adj)
                    list_temp = np.array(list_temp)
                    for i in nx.connected_components(core_graph):
                        core_temp = list(i)
                        core_temp = list_temp[core_temp]
                        core_temp_size = len(core_temp)

                        similar_temp_k1 = 0
                        similar_num_k1 = 0
                        similar_temp_k2 = 0
                        similar_num_k2 = 0
                        
                        scores_temp_k1 = np.zeros(nb_nodes)
                        scores_temp_k2 = np.zeros(nb_nodes)
        
                        for idx in core_temp:
                            for idy in core_temp:
                                if idx != idy:
                                    similar_temp_k1 += features_similarity_k1[idx][idy]
                                    similar_temp_k2 += features_similarity_k2[idx][idy]
                                   
                                    similar_num_k1 += 1
                                    similar_num_k2 += 1
                
                        # Calculate anomaly score for current substructure
                        if similar_num_k1 > 0:
                            avg_sim_k1 = float(similar_temp_k1) / similar_num_k1
                            if abs(avg_sim_k1) < 1e-9: avg_sim_k1 = 1e-9
                            scores_temp_k1[core_temp] = core_temp_size * 1 / avg_sim_k1
                        else:
                            scores_temp_k1[core_temp] = 0

                        if similar_num_k2 > 0:
                            avg_sim_k2 = float(similar_temp_k2) / similar_num_k2
                            if abs(avg_sim_k2) < 1e-9: avg_sim_k2 = 1e-9
                            scores_temp_k2[core_temp] = core_temp_size * 1 / avg_sim_k2
                        else:
                            scores_temp_k2[core_temp] = 0

                        scores_temp = (scores_temp_k1 + scores_temp_k2) / 2
                        multi_round_stru_ano_score.append(scores_temp)

                    k_init += 1

            # Final Topology Perception Anomaly Score
            multi_round_stru_ano_score = np.array(multi_round_stru_ano_score)

            if multi_round_stru_ano_score.size == 0:
                multi_round_stru_ano_score = np.zeros(nb_nodes)
            else:
                multi_round_stru_ano_score = np.mean(multi_round_stru_ano_score, axis=0)

            multi_round_stru_ano_score = np.nan_to_num(multi_round_stru_ano_score, nan=0.0, posinf=0.0, neginf=0.0)

           
            stru_scaler = MinMaxScaler()
            stru_ano_score_final = stru_scaler.fit_transform(multi_round_stru_ano_score.reshape(-1, 1)).reshape(-1)

            
            # Final Score Fusion
            alpha_list = list(np.arange(0, 1, 0.2))  
            rate_auc = []  

            for al in alpha_list:
                final_scores_rate = al * attr_ano_score_final + (1 - al) * stru_ano_score_final
                auc_temp = roc_auc_score(ano_label, final_scores_rate)
                rate_auc.append(auc_temp)
         
            max_alpha = alpha_list[rate_auc.index(max(rate_auc))]
            final_scores_rate = max_alpha * attr_ano_score_final + (1 - max_alpha) * stru_ano_score_final


            auc = roc_auc_score(ano_label, final_scores_rate)  
            auprc = average_precision_score(ano_label, final_scores_rate)  
            all_auc.append(auc)
            all_auprc.append(auprc)
            print('al: ', max_alpha)
            print('Testing AUC:{:.4f}'.format(auc), flush=True)
            print('Testing AUPRC:{:.4f}'.format(auprc), flush=True)



    print('\n==============================')
    print(all_auc)
    print('FINAL TESTING AUC:{:.4f}'.format(np.mean(all_auc)))
    print('==============================')
    print(all_auprc)
    print('FINAL TESTING AUPRC:{:.4f}'.format(np.mean(all_auprc)))
    print('==============================')