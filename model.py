import torch  
import torch.nn as nn  
import torch.nn.functional as F  
from utils import *  


class GCN(nn.Module):
    def __init__(self, in_ft, out_ft, act, bias=True):
        super(GCN, self).__init__()  
        self.fc = nn.Linear(in_ft, out_ft, bias=False)
        self.act = nn.PReLU() if act == 'prelu' else act

        if bias:  
            self.bias = nn.Parameter(torch.FloatTensor(out_ft))
            self.bias.data.fill_(0.0)
        else:
            self.register_parameter('bias', None)

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, seq, adj, sparse=False):
        seq_fts = self.fc(seq)
        if sparse: 
            out = torch.unsqueeze(
                torch.spmm(adj, torch.squeeze(seq_fts, 0)), 0)
        else: 
            out = torch.bmm(adj, seq_fts)
        if self.bias is not None:
            out += self.bias  
        return self.act(out)


class AvgReadout(nn.Module):

    def __init__(self):
        super(AvgReadout, self).__init__()  

    def forward(self, seq):
        return torch.mean(seq, 1)


class MaxReadout(nn.Module):

    def __init__(self):
        super(MaxReadout, self).__init__()

    def forward(self, seq):
        return torch.max(seq, 1).values


class MinReadout(nn.Module):

    def __init__(self):
        super(MinReadout, self).__init__()

    def forward(self, seq):
        return torch.min(seq, 1).values


class WSReadout(nn.Module):

    def __init__(self):
        super(WSReadout, self).__init__()

    def forward(self, seq, query):
        query = query.permute(0, 2, 1)
        sim = torch.matmul(seq, query)
        sim = F.softmax(sim, dim=1)
        sim = sim.repeat(1, 1, 64)
        out = torch.mul(seq, sim)
        out = torch.sum(out, 1)
        return out



class Contextual_Discriminator(nn.Module):

    def __init__(self, n_h, negsamp_round):
        super(Contextual_Discriminator, self).__init__()
        self.f_k = nn.Bilinear(n_h, n_h, 1)
        self.negsamp_round = negsamp_round 
        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Bilinear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, c, h_pl, s_bias1=None, s_bias2=None):
        scs = []  
        scs.append(self.f_k(h_pl, c)) 
        c_mi = c  
        for _ in range(self.negsamp_round):
            c_mi = torch.cat((c_mi[-2:-1, :], c_mi[:-1, :]), 0)
            scs.append(self.f_k(h_pl, c_mi))
        logits = torch.cat(tuple(scs))
        return logits


class Decoder(nn.Module):
    

    def __init__(self, n_in, n_h, hidden_size=128):
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size


        self.network1 = nn.Sequential(
            nn.Linear(n_h, self.hidden_size),  
            nn.PReLU(),  
            nn.Linear(self.hidden_size, self.hidden_size),  
            nn.PReLU(),  
            nn.Linear(self.hidden_size, n_in), 
            nn.PReLU()  
        )
       
        self.network2 = nn.Sequential(
            nn.Linear(n_h * 2, self.hidden_size),
            nn.PReLU(),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.PReLU(),
            nn.Linear(self.hidden_size, n_in),
            nn.PReLU()
        )
       
        self.network3 = nn.Sequential(
            nn.Linear(n_h * 3, self.hidden_size),
            nn.PReLU(),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.PReLU(),
            nn.Linear(self.hidden_size, n_in),
            nn.PReLU()
        )
      
        self.network4 = nn.Sequential(
            nn.Linear(n_h * 4, self.hidden_size),
            nn.PReLU(),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.PReLU(),
            nn.Linear(self.hidden_size, n_in),
            nn.PReLU()
        )
       
        self.network5 = nn.Sequential(
            nn.Linear(n_h * 5, self.hidden_size),
            nn.PReLU(),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.PReLU(),
            nn.Linear(self.hidden_size, n_in),
            nn.PReLU()
        )
        
        self.network6 = nn.Sequential(
            nn.Linear(n_h * 6, self.hidden_size),
            nn.PReLU(),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.PReLU(),
            nn.Linear(self.hidden_size, n_in),
            nn.PReLU()
        )
       
        self.network7 = nn.Sequential(
            nn.Linear(n_h * 7, self.hidden_size),
            nn.PReLU(),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.PReLU(),
            nn.Linear(self.hidden_size, n_in),
            nn.PReLU()
        )

    def forward(self, h_raw, subgraph_size):
    
        sub_size = h_raw.shape[1]
        batch_size = h_raw.shape[0]
        sub_node = h_raw[:, :sub_size - 2, :]
        input_res = sub_node.reshape(batch_size, -1)
        
        if subgraph_size == 1:
            node_recons = self.network1(input_res)  
        elif subgraph_size == 2:
            node_recons = self.network2(input_res)  
        elif subgraph_size == 3:
            node_recons = self.network3(input_res)  
        elif subgraph_size == 4:
            node_recons = self.network4(input_res)  
        elif subgraph_size == 5:
            node_recons = self.network5(input_res)  
        elif subgraph_size == 6:
            node_recons = self.network6(input_res) 
        elif subgraph_size == 7:
            node_recons = self.network7(input_res) 

        return node_recons


class Model(nn.Module):
    def __init__(self, n_in, n_h, activation, negsamp_round_context, readout, hidden_size=128,
                 temperature=0.4, Sinkhorn_iter_times=5, lamb=20, is_rectified=True, topo_t=2, neg_top_k=50,
                 have_neg=False, use_rotation=False, rotation_mode='orthogonal', add_noise_after_rotation=True):
        super(Model, self).__init__()
        self.read_mode = readout  
        self.hidden_size = hidden_size  
        self.decoder = Decoder(n_in, n_h, self.hidden_size)
        
       
        self.use_rotation = use_rotation 
        self.rotation_mode = rotation_mode 
        self.add_noise_after_rotation = add_noise_after_rotation 

        # Initialize learnable projection matrix
        if self.use_rotation:
            if rotation_mode == 'orthogonal':
                self.rotation_params_view1 = nn.Parameter(torch.randn(n_h, n_h) * 0.01)
                self.rotation_params_view2 = nn.Parameter(torch.randn(n_h, n_h) * 0.01)
            elif rotation_mode == 'free':
                self.rotation_matrix_view1 = nn.Parameter(torch.eye(n_h) + torch.randn(n_h, n_h) * 0.01)
                self.rotation_matrix_view2 = nn.Parameter(torch.eye(n_h) + torch.randn(n_h, n_h) * 0.01)
            else:
                raise ValueError(f"不支持的旋转模式: {rotation_mode}")
            
            # Adaptive noise weight
            self.noise_weight = nn.Parameter(torch.ones(n_h))

       
        self.gcn_context = GCN(n_in, n_h, activation)  

        self.temperature = temperature  
        self.Sinkhorn_iter_times = Sinkhorn_iter_times  
        self.lamb = lamb 
        self.rectified = is_rectified  
        self.topo_t = topo_t  
        self.have_neg = have_neg 
        self.neg_top_k = neg_top_k  
       

        if readout == 'max':
            self.read = MaxReadout()
        elif readout == 'min':
            self.read = MinReadout()
        elif readout == 'avg':
            self.read = AvgReadout()
        elif readout == 'weighted_sum':
            self.read = WSReadout()

       
        self.c_disc = Contextual_Discriminator(n_h, negsamp_round_context)  


    # Intra-View Contrastive Loss Calculation
    def InterViewLoss(self, h1, h2, rescale_ratio, have_neg=False, neg_top_k=50):
       
        h1_new = h1.clone()
        h2_new = h2.clone()
        
        h1_new[:, [-2, -1], :] = h1_new[:, [-1, -2], :]
        h2_new[:, [-2, -1], :] = h2_new[:, [-1, -2], :]

        h_graph_1 = h1_new[:, : -1, :]  
        h_node_1 = h1_new[:, -1, :][:, None, :] 
        
        h_graph_2 = h2_new[:, : -1, :] 
        h_node_2 = h2_new[:, -1, :][:, None, :]  

        fx = lambda x: torch.exp(x / self.temperature)

       
        if rescale_ratio is not None:
            sim_pos = Sinkhorn(
                h_graph_1, h_node_1, h_graph_2, h_node_2,  
                self.Sinkhorn_iter_times, self.lamb,  
                rescale_ratio  
            )
            loss_pos = fx(sim_pos) * 2  
        else:
            sim_pos = Sinkhorn(
                h_graph_1, h_node_1, h_graph_2, h_node_2,
                self.Sinkhorn_iter_times, self.lamb
            )
            loss_pos = fx(sim_pos)  

        if have_neg:
            neg_sim_list = []  
            loss_neg_total = 0  
            sim_neg = 0  
            batch_size = h_node_1.shape[0] 
            neg_index = list(range(batch_size))  

            for i in range((batch_size - 1)):
                neg_index.insert(0, neg_index.pop(-1))
                out1_perm = h_graph_1[neg_index].clone()
                out2_perm = h_graph_2[neg_index].clone()
                avg_out1_perm = h_node_1[neg_index].clone()
                avg_out2_perm = h_node_2[neg_index].clone()


                sim_neg1 = Sinkhorn(h_graph_1, h_node_1, out1_perm, avg_out1_perm, self.Sinkhorn_iter_times, self.lamb)
                sim_neg2 = Sinkhorn(h_graph_1, h_node_1, out2_perm, avg_out2_perm, self.Sinkhorn_iter_times, self.lamb)


                sim_neg += (sim_neg1 + sim_neg2) / 2
                loss_neg_total += fx(sim_neg1) + fx(sim_neg2)
                neg_sim_list.append(torch.squeeze(sim_neg1).detach().cpu().numpy())
                neg_sim_list.append(torch.squeeze(sim_neg2).detach().cpu().numpy())


            neg_sim = torch.tensor(np.array(neg_sim_list), requires_grad=True)  
            neg_sim = neg_sim.to(torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'))
            neg_sim = torch.sort(neg_sim, descending=False, dim=0)[0]
            loss_neg_top_k = neg_sim[:neg_top_k, :]
            loss_neg_total = torch.mean(loss_neg_top_k, dim=0)
            loss_neg_total = torch.unsqueeze(loss_neg_total, 1)

            
            loss = -torch.log((loss_pos) / (loss_neg_total + loss_pos))
            sim_neg = sim_neg / (batch_size - 1)
            sim_all = sim_pos - sim_neg
        else:
            loss = -torch.log(loss_pos)
            sim_all = sim_pos  

        return loss, sim_all, sim_pos

    
    def forward(self, seq1, adj1, raw1, size1, seq2, adj2, raw2, size2, full_topology_dist, batch_g_idx1, batch_g_idx2,
                sparse=False, msk=None, samp_bias1=None, samp_bias2=None, has_subcon=False, perturbed=False):
       

        h_1_k1 = self.gcn_context(seq1, adj1, sparse)
        h_raw_k1 = self.gcn_context(raw1, adj1, sparse)

        
        h_1_k2 = self.gcn_context(seq2, adj2, sparse)
        h_raw_k2 = self.gcn_context(raw2, adj2, sparse)

        
        if self.rectified:
            topology_dist = gen_batch_topology_dist(full_topology_dist, batch_g_idx1, batch_g_idx2)
            rescale_ratio = torch.sigmoid(topology_dist / self.topo_t)
        else:
            rescale_ratio = None

        
        # Augmentation: Projection and Noise 
        if perturbed:
            if self.use_rotation:
                if self.rotation_mode == 'orthogonal':
                    skew1 = (self.rotation_params_view1 - self.rotation_params_view1.T) / 2
                    n_h = skew1.shape[0]
                    I = torch.eye(n_h, device=skew1.device, dtype=skew1.dtype)
                    rotation_matrix_1 = torch.matmul(I + skew1, torch.inverse(I - skew1))
                    
                    skew2 = (self.rotation_params_view2 - self.rotation_params_view2.T) / 2
                    rotation_matrix_2 = torch.matmul(I + skew2, torch.inverse(I - skew2))
                    
                elif self.rotation_mode == 'free': 
                    rotation_matrix_1 = self.rotation_matrix_view1
                    rotation_matrix_2 = self.rotation_matrix_view2
                
                
                h_1_k1_rotated = torch.matmul(h_1_k1, rotation_matrix_1.T)  
                h_1_k2_rotated = torch.matmul(h_1_k2, rotation_matrix_2.T)  
                
                if self.add_noise_after_rotation:
                    adaptive_scale = torch.sigmoid(self.noise_weight).view(1, 1, -1)  
                    
                    noise1_k1 = torch.rand_like(h_1_k1_rotated)
                    h_1_k1 = h_1_k1_rotated + self.eps * adaptive_scale * torch.sign(h_1_k1_rotated) * F.normalize(noise1_k1, dim=-1)
                    
                    noise1_k2 = torch.rand_like(h_1_k2_rotated)
                    h_1_k2 = h_1_k2_rotated + self.eps * adaptive_scale * torch.sign(h_1_k2_rotated) * F.normalize(noise1_k2, dim=-1)
                else:
                    h_1_k1 = h_1_k1_rotated
                    h_1_k2 = h_1_k2_rotated
            else:
                noise1_k1 = torch.rand_like(h_1_k1).to(h_1_k1.device)
                h_1_k1 += self.eps * torch.sign(h_1_k1) * F.normalize(noise1_k1, dim=-1)

                noise1_k2 = torch.rand_like(h_1_k2).to(h_1_k2.device)
                h_1_k2 += self.eps * torch.sign(h_1_k2) * F.normalize(noise1_k2, dim=-1)



        node_recons_k1 = self.decoder(h_raw_k1, size1)
        node_recons_k2 = self.decoder(h_raw_k2, size2)


        if self.read_mode != 'weighted_sum':  
            c_k1 = self.read(h_1_k1[:, :-1, :])
            h_mv_k1 = h_1_k1[:, -1, :]

            c_k2 = self.read(h_1_k2[:, :-1, :])
            h_mv_k2 = h_1_k2[:, -1, :]


        else:  
            c_k1 = self.read(h_1_k1[:, :-1, :], h_1_k1[:, -2:-1, :])
            h_mv_k1 = h_1_k1[:, -1, :]

            c_k2 = self.read(h_1_k2[:, :-1, :], h_1_k2[:, -2:-1, :])
            h_mv_k2 = h_1_k2[:, -1, :]



        ret1_k1 = self.c_disc(c_k1, h_mv_k1, samp_bias1, samp_bias2)
        ret1_k2 = self.c_disc(c_k2, h_mv_k2, samp_bias1, samp_bias2)


        if has_subcon:
            if self.rectified:
                Inter_loss_1, sim_all_1, sim_pos_1 = self.InterViewLoss(h_1_k1, h_1_k2, rescale_ratio, self.have_neg,
                                                                        self.neg_top_k)
                Inter_loss_2, sim_all_2, sim_pos_2 = self.InterViewLoss(h_1_k2, h_1_k1, rescale_ratio.transpose(1, 2),
                                                                        self.have_neg, self.neg_top_k)
            else:
                Inter_loss_1, sim_all_1, sim_pos_1 = self.InterViewLoss(h_1_k1, h_1_k2, None, self.have_neg,
                                                                        self.neg_top_k)
                Inter_loss_2, sim_all_2, sim_pos_2 = self.InterViewLoss(h_1_k2, h_1_k1, None, self.have_neg,
                                                                        self.neg_top_k)
        else:
            Inter_loss_1, sim_all_1, sim_pos_1 = None, None, None
            Inter_loss_2, sim_all_2, sim_pos_2 = None, None, None


        return (node_recons_k1, node_recons_k2, ret1_k1, ret1_k2, c_k1, c_k2, h_mv_k1, h_mv_k2,
                Inter_loss_1, Inter_loss_2, sim_all_1, sim_all_2, sim_pos_1, sim_pos_2)

