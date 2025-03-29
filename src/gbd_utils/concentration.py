import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class GeneralGraphConcentrationModule(object):
    def __init__(self, eta, concentration_strategy=None):
        super().__init__()
        self.eta = eta
        self.concentration_strategy = concentration_strategy
    
    def get_threshold_list(self, dataset, threshold_list=None):
        if threshold_list is not None:
            return threshold_list
        
        if self.concentration_strategy == 'deg':
            if dataset == 'comm20':
                threshold_list = [0.8, 0.4]
            elif dataset == 'ego': 
                # TODO
                threshold_list = [0.8, 0.4]
            elif dataset == 'planar':
                # TODO
                threshold_list = [0.8, 0.4]
            elif dataset == 'sbm': 
                # TODO
                threshold_list = [0.8, 0.4]     
        elif self.concentration_strategy == 'central':
            #TODO
            if dataset == 'comm20':
                threshold_list = [0.8, 0.4]
            elif dataset == 'ego': 
                # TODO
                threshold_list = [0.8, 0.4]
            elif dataset == 'planar':
                # TODO
                threshold_list = [0.8, 0.4]
            elif dataset == 'sbm': 
                # TODO
                threshold_list = [0.8, 0.4]  
        elif self.concentration_strategy == 'betweenness':
            #TODO
            if dataset == 'comm20':
                threshold_list = [0.8, 0.4]
            elif dataset == 'ego': 
                # TODO
                threshold_list = [0.8, 0.4]
            elif dataset == 'planar':
                # TODO
                threshold_list = [0.8, 0.4]
            elif dataset == 'sbm': 
                # TODO
                threshold_list = [0.8, 0.4]  
        else:
            raise ValueError('Wrong concentration strategy')

        return threshold_list
    
    def get_value(self, dataset, node_feat=None, adj=None):
        if self.concentration_strategy == 'deg':
            # get degree
            value = adj.sum((-1, -2))
            value = value / torch.max(value, dim=-1, keepdim=True)[0]
        else:
            raise ValueError('Wrong concentration strategy')
        
        return value
    
    def get_eta_x(self, eta_x, value=None, threshold_list=None):
        '''Concentration modulation for eta'''

        if (value is not None) and (threshold_list is not None):
            modulation_eta = 10. * torch.ones_like(value)
            assert (eta_x.size(0) == len(threshold_list) + 1)

            for level in range(len(threshold_list)):
                if level == 0:
                    mask = value >  threshold_list[level]
                elif level == len(threshold_list) - 1:
                    mask = value <= threshold_list[level]
                else:
                    mask = (value > threshold_list[level]) & (value <= threshold_list[level - 1])

                modulation_eta[mask] = eta_x[level]
        else:
            modulation_eta = eta_x

        return modulation_eta

    def get_eta_e(self, num_node, eta_e, eta_x, follow_x=True):
        if follow_x:
            modulation_eta = eta_x.expand(-1, -1, num_node)
        else:
            # modulation_eta = eta_e * torch.ones_like(eta_x.expand(-1, -1, num_node))
            raise NotImplementedError('Not implemented yet')

        return modulation_eta
    

    
class MoleculeGraphConcentrationModule(object):
    def __init__(self, eta, concentration_strategy=None):
        super().__init__()
        self.eta = eta
        self.concentration_strategy = concentration_strategy

    def get_threshold_list(self, dataset, node_feat=None, adj=None, concentration_strategy=None, threshold_list=None):
        return None
    
    def get_value(self, dataset, node_feat=None, adj=None):
        if self.concentration_strategy == 'atom':
            value = node_feat
        else:
            raise ValueError('Wrong concentration strategy')
        
        return value
    
    def get_eta_x(self, eta_x, value=None, threshold_list=None):

        modulation_eta = torch.ones_like(value[..., 0])

        for atom_type in range(value.size(-1)):
            modulation_eta[value[..., atom_type] == 1] = eta_x[atom_type]

        return modulation_eta
    
    def get_eta_e(self, num_nodes, eta_e, eta_x, value):
        modulation_eta = eta_e[-1] * torch.ones_like(eta_x.expand(-1, -1, num_nodes))

        carbon_pos = (value[..., 0] == 1)
        nitrogen_pos = (value[..., 1] == 1)
        oxygen_pos = (value[..., 2] == 1)
        fluorine_pos = (value[..., 3] == 1)

        # bonds between C and C
        cc_pos = (carbon_pos.unsqueeze(-1) * carbon_pos.unsqueeze(-2))
        modulation_eta[cc_pos] = eta_e[0]

        # bonds between C and N
        cn_pos = (carbon_pos.unsqueeze(-1) * nitrogen_pos.unsqueeze(-2))
        modulation_eta[cn_pos] = eta_e[1]

        # bonds between C and O
        co_pos = (carbon_pos.unsqueeze(-1) * oxygen_pos.unsqueeze(-2))
        modulation_eta[co_pos] = eta_e[1]

        # bonds between C and F
        cf_pos = (carbon_pos.unsqueeze(-1) * fluorine_pos.unsqueeze(-2))
        modulation_eta[cf_pos] = eta_e[2]

        # sym
        upper_triangular_mask = torch.zeros_like(modulation_eta)
        indices = torch.triu_indices(row=modulation_eta.size(1), col=modulation_eta.size(2), offset=1)
        upper_triangular_mask[:, indices[0], indices[1]] = 1

        modulation_eta = modulation_eta * upper_triangular_mask
        modulation_eta = (modulation_eta + torch.transpose(modulation_eta, 1, 2))
        modulation_eta[torch.where(modulation_eta==0.)] = eta_e[3]
    
        return modulation_eta
    
    def get_eta_x_sample(self, eta, value, eta_from=None):

        modulation_eta = torch.ones_like(value[..., 0], device=value.device)

        indices = torch.arange(modulation_eta.size(1)).unsqueeze(0).expand(modulation_eta.size(0), -1).to(modulation_eta.device)

        carbon_num = eta_from[..., 0].unsqueeze(-1)
        current_num = torch.zeros_like(carbon_num)
        mask = indices < current_num + carbon_num
        modulation_eta[mask] = eta[0]
        current_num += carbon_num

        nitrogen_num = eta_from[..., 1].unsqueeze(-1)
        mask = torch.logical_and(indices >= current_num, indices < (current_num + nitrogen_num))
        modulation_eta[mask] = eta[1]
        current_num += nitrogen_num

        oxygen_num = eta_from[..., 2].unsqueeze(-1)
        mask = torch.logical_and(indices >= current_num, indices < (current_num + oxygen_num))
        modulation_eta[mask] = eta[2]
        current_num += oxygen_num

        fluorine_num = eta_from[..., 3].unsqueeze(-1)
        mask = torch.logical_and(indices >= current_num, indices < (current_num + fluorine_num))
        modulation_eta[mask] = eta[3]
        current_num += fluorine_num

        return modulation_eta
    
    def get_eta_e_sample(self, num_nodes, batch_size, eta_e, eta_from, eta_x):

        modulation_eta = torch.ones_like(eta_x.expand(-1, -1, num_nodes), device=eta_x.device)

        for g_id in range(batch_size):
            carbon_num = eta_from[g_id, ..., 0].sum().to(torch.long)
            nitrogen_num = eta_from[g_id, ..., 1].sum().to(torch.long)
            oxygen_num = eta_from[g_id, ..., 2].sum().to(torch.long)
            fluorine_num = eta_from[g_id, ..., 3].sum().to(torch.long)

            C_end_pos = carbon_num
            N_end_pos = C_end_pos + nitrogen_num
            O_end_pos = N_end_pos + oxygen_num
            F_end_pos = O_end_pos + fluorine_num

            # other bonds
            modulation_eta[g_id] = eta_e[3] * modulation_eta[g_id]

            # bonds between C and C
            modulation_eta[g_id, :C_end_pos, :C_end_pos] = eta_e[0]

            # bonds between C and N
            modulation_eta[g_id, :C_end_pos, C_end_pos: N_end_pos] = eta_e[1]

            # bonds between C and O
            modulation_eta[g_id, :C_end_pos, N_end_pos: O_end_pos] = eta_e[1]

            # bonds between C and F
            modulation_eta[g_id, :C_end_pos, O_end_pos: F_end_pos] = eta_e[2]


        # sym
        upper_triangular_mask = torch.zeros_like(modulation_eta)
        indices = torch.triu_indices(row=modulation_eta.size(1), col=modulation_eta.size(2), offset=1)
        upper_triangular_mask[:, indices[0], indices[1]] = 1

        modulation_eta = modulation_eta * upper_triangular_mask
        modulation_eta = (modulation_eta + torch.transpose(modulation_eta, 1, 2))
        modulation_eta[torch.where(modulation_eta==0.)] = eta_e[3]

        return modulation_eta

    
class CustomGraphConcentrationModule(object):
    def __init__(self, eta, concentration_modulation=None):
        super().__init__()
        self.eta = eta
        self.concentration_modulation = concentration_modulation
    
    def get_threshold_list(self, dataset, node_feat=None, adj=None, concentration_strategy=None, threshold_list=None):
        if threshold_list is not None:
            value = None
            return threshold_list, value
        
        if concentration_strategy == 'your strategy':
            threshold_list, value = None, None
            pass
        else:
            raise ValueError('Wrong concentration strategy')

        return threshold_list, value
    
    def get_eta_x(self, eta_x, value
                  , threshold_list=None):
        '''Concentration modulation for eta'''
        modulation_eta = 10. * torch.ones_like(eta_x)
        return modulation_eta

    def get_eta_e(self, num_node, eta_e, eta_x, follow_x=True):
        if follow_x:
            modulation_eta = eta_x.expand(-1, -1, num_node)
        else:
            # modulation_eta = eta_e * torch.ones_like(eta_x.expand(-1, -1, num_node))
            raise NotImplementedError('Not implemented yet')

        return modulation_eta

