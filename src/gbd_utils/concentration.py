import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class GeneralGraphConcentrationModule(object):
    def __init__(self, eta, concentration_modulation=None):
        super().__init__()
        self.eta = eta
        self.concentration_modulation = concentration_modulation
    
    def get_threshold_list(self, dataset, threshold_list=None):
        if threshold_list is not None:
            return threshold_list
        
        if self.concentration_strategy == 'deg':
            if dataset == 'gdss-comm20':
                threshold_list = [0.8, 0.4]
            elif dataset == 'gdss-ego': 
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
            if dataset == 'gdss-comm20':
                threshold_list = [0.8, 0.4]
            elif dataset == 'gdss-ego': 
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
            if dataset == 'gdss-comm20':
                threshold_list = [0.8, 0.4]
            elif dataset == 'gdss-ego': 
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
    
    def get_eta_x(self, eta_x, value=None, threshold_list=None):
        '''Concentration modulation for eta'''
        modulation_eta = 10. * torch.ones_like(eta_x)

        if (value is not None) and (threshold_list is not None):
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
    def __init__(self, eta, concentration_modulation=None):
        super().__init__()
        self.eta = eta
        self.concentration_modulation = concentration_modulation

    def get_threshold_list(self, dataset, node_feat=None, adj=None, concentration_strategy=None, threshold_list=None):
        pass

    def get_eta_atom(self, E, eta, deg=None):
        pass

    def get_eta_bond(self, E, X, eta, eta_x):
        pass

    
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

