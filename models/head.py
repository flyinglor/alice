import torch
import torch.nn as nn
import utils

from utils import trunc_normal_
import torch.nn.functional as F

class CSyncBatchNorm(nn.SyncBatchNorm):
    def __init__(self,
                 *args,
                 with_var=False,
                 **kwargs):
        super(CSyncBatchNorm, self).__init__(*args, **kwargs)
        self.with_var = with_var

    def forward(self, x):
        # center norm
        self.training = False
        if not self.with_var:
            self.running_var = torch.ones_like(self.running_var)
        normed_x = super(CSyncBatchNorm, self).forward(x)
        # udpate center
        self.training = True
        _ = super(CSyncBatchNorm, self).forward(x)
        return normed_x

class PSyncBatchNorm(nn.SyncBatchNorm):
    def __init__(self,
                 *args,
                 bunch_size,
                 **kwargs):
        procs_per_bunch = min(bunch_size, utils.get_world_size())
        assert utils.get_world_size() % procs_per_bunch == 0
        n_bunch = utils.get_world_size() // procs_per_bunch
        #
        ranks = list(range(utils.get_world_size()))
        print('---ALL RANKS----\n{}'.format(ranks))
        rank_groups = [ranks[i*procs_per_bunch: (i+1)*procs_per_bunch] for i in range(n_bunch)]
        print('---RANK GROUPS----\n{}'.format(rank_groups))
        process_groups = [torch.distributed.new_group(pids) for pids in rank_groups]
        bunch_id = utils.get_rank() // procs_per_bunch
        process_group = process_groups[bunch_id]
        print('---CURRENT GROUP----\n{}'.format(process_group))
        super(PSyncBatchNorm, self).__init__(*args, process_group=process_group, **kwargs)

class CustomSequential(nn.Sequential):
    bn_types = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.SyncBatchNorm)

    def forward(self, input):
        for module in self:
            dim = len(input.shape)
            if isinstance(module, self.bn_types) and dim > 2:
                perm = list(range(dim - 1)); perm.insert(1, dim - 1)
                inv_perm = list(range(dim)) + [1]; inv_perm.pop(1)
                input = module(input.permute(*perm)).permute(*inv_perm)
            else:
                input = module(input)
        return input

class DINOHead(nn.Module):
    def __init__(self, in_dim, out_dim, norm=None, act='gelu', last_norm=None, 
                 nlayers=3, hidden_dim=2048, bottleneck_dim=256, norm_last_layer=True, **kwargs):
        super().__init__()
        norm = self._build_norm(norm, hidden_dim)
        last_norm = self._build_norm(last_norm, out_dim, affine=False, **kwargs)
        act = self._build_act(act)
        self.in_dim = in_dim
        nlayers = max(nlayers, 1)
        if nlayers == 1:
            if bottleneck_dim > 0:
                self.mlp = nn.Linear(in_dim, bottleneck_dim)
            else:
                self.mlp = nn.Linear(in_dim, out_dim)
        else:
            layers = [nn.Linear(in_dim, hidden_dim)]
            if norm is not None:
                layers.append(norm)
            layers.append(act)
            for _ in range(nlayers - 2):
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                if norm is not None:
                    layers.append(norm)
                layers.append(act)
            if bottleneck_dim > 0:
                layers.append(nn.Linear(hidden_dim, bottleneck_dim))
            else:
                layers.append(nn.Linear(hidden_dim, out_dim))
            self.mlp = CustomSequential(*layers)
        self.apply(self._init_weights)
        
        if bottleneck_dim > 0:
            self.last_layer = nn.utils.weight_norm(nn.Linear(bottleneck_dim, out_dim, bias=False))
            self.last_layer.weight_g.data.fill_(1)
            if norm_last_layer:
                self.last_layer.weight_g.requires_grad = False
        else:
            self.last_layer = None

        self.last_norm = last_norm

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.mlp(x)
        if self.last_layer is not None:
            x = nn.functional.normalize(x, dim=-1, p=2)
            x = self.last_layer(x)
        if self.last_norm is not None:
            x = self.last_norm(x)
        return x

    def _build_norm(self, norm, hidden_dim, **kwargs):
        if norm == 'bn':
            norm = nn.BatchNorm1d(hidden_dim, **kwargs)
        elif norm == 'syncbn':
            norm = nn.SyncBatchNorm(hidden_dim, **kwargs)
        elif norm == 'csyncbn':
            norm = CSyncBatchNorm(hidden_dim, **kwargs)
        elif norm == 'psyncbn':
            norm =  PSyncBatchNorm(hidden_dim, **kwargs)
        elif norm == 'ln':
            norm = nn.LayerNorm(hidden_dim, **kwargs)
        else:
            assert norm is None, "unknown norm type {}".format(norm)
        return norm

    def _build_act(self, act):
        if act == 'relu':
            act = nn.ReLU()
        elif act == 'gelu':
            act = nn.GELU()
        else:
            assert False, "unknown act type {}".format(act)
        return act

    
class AliceHead(DINOHead):

    def __init__(self, *args, patch_out_dim=512, norm=None, act='gelu', last_norm=None, 
                 nlayers=3, hidden_dim=2048, bottleneck_dim=256, norm_last_layer=True, 
                 shared_head=False, **kwargs):
        
        super(AliceHead, self).__init__(*args,
                                        norm=norm,
                                        act=act,
                                        last_norm=last_norm,
                                        nlayers=nlayers,
                                        hidden_dim=hidden_dim,
                                        bottleneck_dim=bottleneck_dim,
                                        norm_last_layer=norm_last_layer, 
                                        **kwargs)

        if not shared_head:
            if bottleneck_dim > 0:
                self.last_layer2 = nn.utils.weight_norm(nn.Linear(bottleneck_dim, patch_out_dim, bias=False))
                self.last_layer2.weight_g.data.fill_(1)
                if norm_last_layer:
                    self.last_layer2.weight_g.requires_grad = False
            else:
                self.mlp2 = nn.Linear(hidden_dim, patch_out_dim)
                self.last_layer2 = None

            self.last_norm2 = self._build_norm(last_norm, patch_out_dim, affine=False, **kwargs)
        else:
            if bottleneck_dim > 0:
                self.last_layer2 = self.last_layer
            else:
                self.mlp2 = self.mlp[-1]
                self.last_layer2 = None

            self.last_norm2 = self.last_norm
        
        self.mlp3 = nn.Linear(384, self.in_dim)
        self.mlp4 = nn.Linear(192, self.in_dim)

    def forward(self, cls_token, encoder, decoder):
        encoder = encoder.flatten(2).transpose(1, 2) # B L C
        decoder = decoder.flatten(2).transpose(1, 2)
        encoder_new = self.mlp3(encoder)
        decoder_new = self.mlp4(decoder)
        
        # if len(x_patch.shape) == 2:
        #     return super(Att_iBOTHead, self).forward(x_patch, cls_token)
        
        # x = torch.cat([cls_token1_new.unsqueeze(1), cls_token2_new.unsqueeze(1), x_patch], dim=1) # B 1+L C
        x = torch.cat([cls_token.unsqueeze(1), encoder_new], dim=1)
        xx = torch.cat([cls_token.unsqueeze(1), decoder_new], dim=1)
        if self.last_layer is not None:
            x, xx = self.mlp(x), self.mlp(xx)
            x, xx = nn.functional.normalize(x, dim=-1, p=2), nn.functional.normalize(xx, dim=-1, p=2)
            x1 = self.last_layer(x[:, 0])
            x2 = self.last_layer2(x[:, 1:])
            x3 = self.last_layer2(xx[:, 1:])
        else:
            x, xx = self.mlp[:-1](x), self.mlp[:-1](xx)
            x1 = self.mlp[-1](x[:, 0])
            x2 = self.mlp2(x[:, 1:])
            x3 = self.mlp2(xx[:, 1:])
        
        if self.last_norm is not None:
            x1 = self.last_norm(x1)
            x2 = self.last_norm2(x2)
            x3 = self.last_norm2(x3)
        
        return x1, x2, x3    

class ClassificationHead2FC(nn.Module):
    def __init__(self, input_dim, num_classes=3, cls=False):
        super(ClassificationHead2FC, self).__init__()
        self.linear1 = nn.Linear(input_dim, input_dim)
        self.ln1 = nn.LayerNorm(input_dim, eps=1e-6)
        self.relu1 = nn.ReLU()
        
        self.avg_pool = nn.AvgPool1d(kernel_size=8, stride=1)
        self.fc = nn.Linear(input_dim, num_classes)

        self.cls = cls

    def forward(self, x):
        x = self.linear1(x)  # [batch_size, seq_len, hidden_dim] torch.Size([8, 8, 512])
        x = self.ln1(x)  # [batch_size, seq_len, hidden_dim]
        x = self.relu1(x)  # [batch_size, seq_len, hidden_dim]

        if not self.cls:
            x = x.transpose(1, 2)  # [batch_size, hidden_dim, seq_len]
            x = self.avg_pool(x)  # [batch_size, hidden_dim, 1]  
            # Remove the singleton dimension: [batch_size, hidden_dim]
            x = x.squeeze(-1)
            
        x = self.fc(x)  # [batch_size, output_dim]
        return F.softmax(x, dim=1)

class ClassificationHead3FC(nn.Module):
    def __init__(self, input_dim, num_classes=3, cls=False):
        super(ClassificationHead3FC, self).__init__()
        self.linear1 = nn.Linear(input_dim, 256)  # Reduce from 512 to 256
        self.ln1 = nn.LayerNorm(256, eps=1e-6)
        self.relu1 = nn.ReLU()

        self.linear2 = nn.Linear(256, 256)  # Retain dimensionality at 256
        self.ln2 = nn.LayerNorm(256, eps=1e-6)
        self.relu2 = nn.ReLU()

        self.linear3 = nn.Linear(256, num_classes)  # Final output layer

        self.cls = cls

    def forward(self, x):
        x = self.linear1(x)  # [batch_size, seq_len, 256]
        x = self.ln1(x)  # [batch_size, seq_len, 256]
        x = self.relu1(x)  # [batch_size, seq_len, 256]

        x = self.linear2(x)  # [batch_size, seq_len, 256]
        x = self.ln2(x)  # [batch_size, seq_len, 256]
        x = self.relu2(x)  # [batch_size, seq_len, 256]

        if not self.cls:
            x = x.transpose(1, 2)  # [batch_size, hidden_dim, seq_len]
            x = nn.AvgPool1d(kernel_size=x.shape[-1])(x)  # Global AvgPool
            x = x.squeeze(-1)  # [batch_size, hidden_dim]

        x = self.linear3(x)  # [batch_size, num_classes]
        return F.softmax(x, dim=1)

class ClassificationHeadCLS(nn.Module):
    def __init__(self, input_dim, num_classes=3):
        super(ClassificationHeadCLS, self).__init__()
        self.linear1 = nn.Linear(input_dim, 256)  # Reduce from 512 to 256
        self.ln1 = nn.LayerNorm(256, eps=1e-6)
        self.relu1 = nn.ReLU()

        self.linear2 = nn.Linear(256, 256)  # Retain dimensionality at 256
        self.ln2 = nn.LayerNorm(256, eps=1e-6)
        self.relu2 = nn.ReLU()

        self.linear3 = nn.Linear(256, num_classes)  # Final output layer

    def forward(self, x):
        x = self.linear1(x)  # [batch_size, seq_len, 256]
        x = self.ln1(x)  # [batch_size, seq_len, 256]
        x = self.relu1(x)  # [batch_size, seq_len, 256]

        x = self.linear2(x)  # [batch_size, seq_len, 256]
        x = self.ln2(x)  # [batch_size, seq_len, 256]
        x = self.relu2(x)  # [batch_size, seq_len, 256]

        x = self.linear3(x)  # [batch_size, num_classes]
        return F.softmax(x, dim=1)