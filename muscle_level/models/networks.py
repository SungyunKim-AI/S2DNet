import math
from collections import OrderedDict
from functools import partial
from typing import Dict, Optional, Union

import torch
from einops import repeat
from torch import nn

from .models_utils import Adaptive_Spectral_Block, Block, trunc_normal_


class BimodalMAE(nn.Module):
    """
    :param input_adapters: Dictionary of task -> input adapters
    :param output_adapters: Optional dictionary of task -> output adapters
    :param num_global_tokens: Number of additional global tokens to add (like cls tokens), default is 1
    :param dim_tokens: Dimension of encoder tokens
    :param depth: Depth of encoder
    :param num_heads: Number of attention heads
    :param mlp_ratio: MLP hidden dim ratio
    :param qkv_bias: Set to False to disable bias
    :param drop_rate: Dropout after MLPs and Attention
    :param attn_drop_rate: Attention matrix drop rate
    :param drop_path_rate: DropPath drop rate
    :param norm_layer: Type of normalization layer
    """

    def __init__(self,
                 input_adapters: Dict[str, nn.Module],
                 output_adapters: Dict[str, nn.Module],
                 num_global_tokens: int = 1,
                 dim_tokens_enc: int = 192,
                 depth_ms: int = 4,
                 depth_mc: int = 4,
                 num_heads: int = 3,
                 mlp_ratio: float = 4.0,
                 qkv_bias: bool = True,
                 drop_rate: float = 0.0,
                 attn_drop_rate: float = 0.0,
                 drop_path_rate: float = 0.0,
                 out_indices_t: list = [1,3],
                 out_indices_f: list = [1,3],
                 out_indices_fusion: list = [1],
                 norm_layer: nn.Module = partial(nn.LayerNorm, eps=1e-6),
                 apply_asb: bool = True):
        super().__init__()

        # Initialize input and output adapters
        for adapter in input_adapters.values():
            adapter.init(dim_tokens=dim_tokens_enc)
        self.input_adapters = nn.ModuleDict(input_adapters)

        if output_adapters is not None:
            for adapter in output_adapters.values():
                adapter.init(dim_tokens_enc=dim_tokens_enc)
            self.output_adapters = nn.ModuleDict(output_adapters)
        else:
            self.output_adapters = None

        # Additional learnable tokens that can be used by encoder to process/store global information
        self.num_global_tokens = num_global_tokens
        self.global_tokens = nn.Parameter(torch.zeros(1, num_global_tokens, dim_tokens_enc))
        trunc_normal_(self.global_tokens, std=0.02)

        # Adaptive Spectral Block
        if apply_asb:
            self.adaptive_spectral_block = Adaptive_Spectral_Block(dim=1)
        else:
            self.adaptive_spectral_block = None

        # modality specific encoder
        dpr_ms = [x.item() for x in torch.linspace(0, drop_path_rate, depth_ms)]  # stochastic depth decay rule
        self.encoder_ms_t = nn.Sequential(*[
            Block(dim=dim_tokens_enc, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                  drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr_ms[i], norm_layer=norm_layer)
            for i in range(depth_ms)
        ])

        self.ms_t_norm = norm_layer(dim_tokens_enc)

        self.encoder_ms_f = nn.Sequential(*[
            Block(dim=dim_tokens_enc, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                  drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr_ms[i], norm_layer=norm_layer)
            for i in range(depth_ms)
        ])

        self.ms_f_norm = norm_layer(dim_tokens_enc)

        # modality collaborative encoder
        dpr_mc = [x.item() for x in torch.linspace(0, drop_path_rate, depth_mc)]  # stochastic depth decay rule
        self.encoder_mc = nn.Sequential(*[
            Block(dim=dim_tokens_enc, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                  drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr_mc[i], norm_layer=norm_layer)
            for i in range(depth_mc)
        ])

        out_norm = [
            norm_layer(dim_tokens_enc)
            for _ in range(2)
        ]
        self.out_norm = torch.nn.ModuleList(out_norm)

        self.out_indices_t = out_indices_t
        self.out_indices_f = out_indices_f
        self.out_indices_fusion = out_indices_fusion

        self.apply(self._init_weights)
        for name, m in self.named_modules():
            if isinstance(m, nn.Linear):
                if 'qkv' in name:
                    # treat the weights of Q, K, V separately
                    val = math.sqrt(6. / float(m.weight.shape[0] // 3 + m.weight.shape[1]))
                    nn.init.uniform_(m.weight, -val, val)
                elif 'kv' in name:
                    # treat the weights of K, V separately
                    val = math.sqrt(6. / float(m.weight.shape[0] // 2 + m.weight.shape[1]))
                    nn.init.uniform_(m.weight, -val, val)

            if isinstance(m, nn.Conv1d):
                if '.proj' in name:
                    # From MAE, initialize projection like nn.Linear (instead of nn.Conv1d)
                    w = m.weight.data
                    nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def get_num_layers(self):
        return len(self.encoder_ms_t) + len(self.encoder_mc)

    @torch.jit.ignore
    def no_weight_decay(self):
        no_wd_set = {'global_tokens'}

        for task, adapter in self.input_adapters.items():
            if hasattr(adapter, 'no_weight_decay'):
                to_skip = adapter.no_weight_decay()
                to_skip = set([f'input_adapters.{task}.{name}' for name in to_skip])
                no_wd_set = no_wd_set | to_skip

        if self.output_adapters is not None:
            for task, adapter in self.output_adapters.items():
                if hasattr(adapter, 'no_weight_decay'):
                    to_skip = adapter.no_weight_decay()
                    to_skip = set([f'output_adapters.{task}.{name}' for name in to_skip])
                    no_wd_set = no_wd_set | to_skip

        return no_wd_set

    def generate_random_masks(self,
                              input_tokens: Dict[str, torch.Tensor],
                              num_encoded_tokens_t: int,
                              num_encoded_tokens_f: int,
                              ):
        """
        Sample a total of num_encoded_tokens from different tasks using Dirichlet sampling.

        :param input_tokens: Dictionary of tensors to sample num_encoded_tokens from
        :param num_encoded_tokens: Number of tokens to select.
        """
        B = list(input_tokens.values())[0].shape[0]
        num_tokens = list(input_tokens.values())[0].shape[1]
        device = list(input_tokens.values())[0].device

        # Use noise to shuffle arange
        noise = torch.rand(B, num_tokens, device=device)  # noise in [0, 1]
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        ids_keep = ids_shuffle[:, :num_encoded_tokens_t]
        ids_mask = ids_shuffle[:, num_encoded_tokens_t:]
        mask = torch.ones([B, num_tokens], device=device)
        mask[:, :num_encoded_tokens_t] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        noise1 = torch.rand(B, int(0.5 * num_tokens), device=device)  # noise in [0, 1]
        ids_shuffle1 = torch.argsort(noise1, dim=1)  # ascend: small is keep, large is remove
        ids_restore1 = torch.argsort(ids_shuffle1, dim=1)
        ids_keep1 = ids_shuffle1[:, :num_encoded_tokens_f]
        ids_mask1 = ids_shuffle1[:, num_encoded_tokens_f:]
        mask1 = torch.ones([B, int(0.5 * num_tokens)], device=device)
        mask1[:, :num_encoded_tokens_f] = 0
        # unshuffle to get the binary mask
        mask1 = torch.gather(mask1, dim=1, index=ids_restore1)
        # Convert to dict
        mask_dict = {'time': mask, 'freq': mask1}
        ids_keep_dict = {'time': ids_keep, 'freq': ids_keep1}
        ids_restore_dict = {'time': ids_restore, 'freq': ids_restore1}
        ids_mask_dict = {'time': ids_mask, 'freq': ids_mask1}

        return mask_dict, ids_keep_dict, ids_restore_dict, ids_mask_dict

    def generate_input_info(self, input_task_tokens, image_size):
        input_info = OrderedDict()
        i = 0
        input_info['tasks'] = {}
        for domain, tensor in input_task_tokens.items():
            num_tokens = tensor.shape[1]
            d = {
                'num_tokens': num_tokens,
                'has_1d_posemb': False,  # TODO: Modify when adding non-2D tasks
                'start_idx': i,
                'end_idx': i + num_tokens,
            }
            i += num_tokens
            input_info['tasks'][domain] = d

        input_info['image_size'] = image_size
        input_info['num_task_tokens'] = i
        input_info['num_global_tokens'] = self.num_global_tokens

        return input_info


    def forward(self,
                x: Union[Dict[str, torch.Tensor], torch.Tensor],
                num_encoded_tokens_t: int = 25,
                num_encoded_tokens_f: int = 12):

        if 'time' in x:
            B, C, W = x['time'].shape
            if self.adaptive_spectral_block is not None:
                time, freq = self.adaptive_spectral_block(x['time'])
                x = {'time': time, 'freq':freq}
        else:
            B, C, W = list(x.values())[0].shape  # TODO: Deal with case where not all have same shape

        # Encode selected inputs to tokens
        input_task_tokens = {
            domain: self.input_adapters[domain](tensor)
            for domain, tensor in x.items()
            if domain in self.input_adapters
        }

        input_info1 = OrderedDict()
        input_info2 = OrderedDict()
        input_info1['image_size'] = W
        input_info1['num_task_tokens'] = input_task_tokens['time'].shape[1]
        input_info2['image_size'] = int(0.5 * W)
        input_info2['num_task_tokens'] = input_task_tokens['freq'].shape[1]

        # Generating masks
        mask, ids_keep, ids_restore, ids_mask = self.generate_random_masks(
            input_task_tokens,
            num_encoded_tokens_t,
            num_encoded_tokens_f
        )

        # Apply mask
        vis_tokens_t = torch.gather(input_task_tokens['time'], dim=1,
                                    index=ids_keep['time'].unsqueeze(-1).repeat(1, 1,
                                                                                 input_task_tokens['time'].shape[2]))
        vis_tokens_f = torch.gather(input_task_tokens['freq'], dim=1,
                                    index=ids_keep['freq'].unsqueeze(-1).repeat(1, 1,
                                                                                 input_task_tokens['freq'].shape[2]))

        # Add global tokens to input tokens
        cls_tokens_t = repeat(self.global_tokens, '() n d -> b n d', b=B)
        cls_tokens_f = repeat(self.global_tokens, '() n d -> b n d', b=B)

        input_tokens_t = torch.cat([cls_tokens_t, vis_tokens_t], dim=1)
        input_tokens_f = torch.cat([cls_tokens_f, vis_tokens_f], dim=1)

        res_t = []
        for i, layer in enumerate(self.encoder_ms_t):
            input_tokens_t = layer(input_tokens_t)
            if i in self.out_indices_t:
                res_t.append(input_tokens_t[:, 1:])

        res_f = []
        for i, layer in enumerate(self.encoder_ms_f):
            input_tokens_f = layer(input_tokens_f)
            if i in self.out_indices_f:
                res_f.append(input_tokens_f[:, 1:])

        input_tokens_t = self.ms_t_norm(input_tokens_t)
        input_tokens_f = self.ms_f_norm(input_tokens_f)

        cls_tokens = input_tokens_t[:, :1, :] + input_tokens_f[:, :1, :]

        encoder_tokens_mc = torch.cat([input_tokens_t[:, 1:, :], input_tokens_f[:, 1:, :]], dim=1)
        encoder_tokens_mc = torch.cat([cls_tokens, encoder_tokens_mc], dim=1)

        for i, layer in enumerate(self.encoder_mc):
            encoder_tokens_mc = layer(encoder_tokens_mc)
            if i in self.out_indices_fusion:
                res_t.append(encoder_tokens_mc[:, 1:num_encoded_tokens_t+1])
                res_f.append(encoder_tokens_mc[:, num_encoded_tokens_t+1:])

        e2d_t = self.out_norm[0](encoder_tokens_mc[:, 1:num_encoded_tokens_t+1])
        e2d_f = self.out_norm[1](encoder_tokens_mc[:, num_encoded_tokens_t+1:])
        preds = {
            'time': self.output_adapters['time'](
                encoder_tokens=e2d_t,
                input_info=input_info1,
                ids_restore=ids_restore['time'],
                inter_input=res_t
            ),
            'freq': self.output_adapters['freq'](
                encoder_tokens=e2d_f,
                input_info=input_info2,
                ids_restore=ids_restore['freq'],
                inter_input=res_f
            ),
        }

        return preds, mask


class BimodalMAE_Classifier(nn.Module):
    """
    Classifier class for Multi-Instance Learning
    Input: bag data of shape (B, 50, C, W)
    """

    def __init__(self,
                 input_adapters: Dict[str, nn.Module],
                 output_adapters: Optional[Dict[str, nn.Module]],
                 num_global_tokens: int = 1,
                 dim_tokens_enc: int = 192,
                 depth_ms: int = 4,
                 depth_mc: int = 4,
                 num_heads: int = 3,
                 mlp_ratio: float = 4.0,
                 qkv_bias: bool = True,
                 drop_rate: float = 0.0,
                 attn_drop_rate: float = 0.0,
                 drop_path_rate: float = 0.0,
                 norm_layer: nn.Module = partial(nn.LayerNorm, eps=1e-6),
                 apply_asb: bool = True):
        super().__init__()

        # Initialize input and output adapters
        for adapter in input_adapters.values():
            adapter.init(dim_tokens=dim_tokens_enc)
        self.input_adapters = nn.ModuleDict(input_adapters)
        if output_adapters is not None:
            for adapter in output_adapters.values():
                adapter.init(dim_tokens_enc=dim_tokens_enc)
            self.output_adapters = nn.ModuleDict(output_adapters)
        else:
            self.output_adapters = None

        # Additional learnable tokens that can be used by encoder to process/store global information
        self.num_global_tokens = num_global_tokens
        self.global_tokens = nn.Parameter(torch.zeros(1, num_global_tokens, dim_tokens_enc))
        trunc_normal_(self.global_tokens, std=0.02)

        # Adaptive Spectral Block
        if apply_asb:
            self.adaptive_spectral_block = Adaptive_Spectral_Block(dim=1)
        else:
            self.adaptive_spectral_block = None

        # modality specific encoder
        dpr_ms = [x.item() for x in torch.linspace(0, drop_path_rate, depth_ms)]  # stochastic depth decay rule
        self.encoder_ms_t = nn.Sequential(*[
            Block(dim=dim_tokens_enc, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                  drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr_ms[i], norm_layer=norm_layer)
            for i in range(depth_ms)
        ])

        self.ms_t_norm = norm_layer(dim_tokens_enc)

        self.encoder_ms_f = nn.Sequential(*[
            Block(dim=dim_tokens_enc, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                  drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr_ms[i], norm_layer=norm_layer)
            for i in range(depth_ms)
        ])

        self.ms_f_norm = norm_layer(dim_tokens_enc)

        # modality collaborative encoder
        dpr_mc = [x.item() for x in torch.linspace(0, drop_path_rate, depth_mc)]  # stochastic depth decay rule
        self.encoder_mc = nn.Sequential(*[
            Block(dim=dim_tokens_enc, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                  drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr_mc[i], norm_layer=norm_layer)
            for i in range(depth_mc)
        ])


        self.apply(self._init_weights)
        for name, m in self.named_modules():
            if isinstance(m, nn.Linear):
                if 'qkv' in name:
                    # treat the weights of Q, K, V separately
                    val = math.sqrt(6. / float(m.weight.shape[0] // 3 + m.weight.shape[1]))
                    nn.init.uniform_(m.weight, -val, val)
                elif 'kv' in name:
                    # treat the weights of K, V separately
                    val = math.sqrt(6. / float(m.weight.shape[0] // 2 + m.weight.shape[1]))
                    nn.init.uniform_(m.weight, -val, val)

            if isinstance(m, nn.Conv1d):
                if '.proj' in name:
                    # From MAE, initialize projection like nn.Linear (instead of nn.Conv1d)
                    w = m.weight.data
                    nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def get_num_layers(self):
        return len(self.encoder_ms_t) + len(self.encoder_mc)

    
    @torch.jit.ignore
    def no_weight_decay(self):
        no_wd_set = {'global_tokens'}

        for task, adapter in self.input_adapters.items():
            if hasattr(adapter, 'no_weight_decay'):
                to_skip = adapter.no_weight_decay()
                to_skip = set([f'input_adapters.{task}.{name}' for name in to_skip])
                no_wd_set = no_wd_set | to_skip

        if self.output_adapters is not None:
            for task, adapter in self.output_adapters.items():
                if hasattr(adapter, 'no_weight_decay'):
                    to_skip = adapter.no_weight_decay()
                    to_skip = set([f'output_adapters.{task}.{name}' for name in to_skip])
                    no_wd_set = no_wd_set | to_skip

        return no_wd_set

    def forward(
        self, 
        x: Union[Dict[str, torch.Tensor], torch.Tensor],
        norm_dict: Dict[str, torch.Tensor], 
        return_embed=False,
        **kwargs
    ):
        """
        MIL forward pass - batched processing
        Input: {'time': (B, 50, C, W), 'freq': (B, 50, C, W//2)}
        Output: compatible with output_adapters
        """
        if 'time' in x:
            B, num_instances, C, W = x['time'].shape
            if self.adaptive_spectral_block is not None:
                time, freq = self.adaptive_spectral_block(x['time'])
                x = {'time': time, 'freq':freq}
        else:
            B, num_instances, C, W = list(x.values())[0].shape

        # Reshape to batch all instances together: (B, 50, C, W) -> (B*50, C, W)
        batched_data = {
            domain: tensor.view(B * num_instances, C, tensor.shape[-1])
            for domain, tensor in x.items()
            if domain in self.input_adapters
        }

        # Encode selected inputs to tokens (B*50 instances at once)
        input_task_tokens = {
            domain: self.input_adapters[domain](tensor)
            for domain, tensor in batched_data.items()
            if domain in self.input_adapters
        }

        # Add global tokens to input tokens
        cls_tokens_t = repeat(self.global_tokens, '() n d -> b n d', b=B * num_instances)
        cls_tokens_f = repeat(self.global_tokens, '() n d -> b n d', b=B * num_instances)

        input_tokens_t = torch.cat([cls_tokens_t, input_task_tokens['time']], dim=1)
        input_tokens_f = torch.cat([cls_tokens_f, input_task_tokens['freq']], dim=1)

        # Transformer forward pass (MS-MC) - process all instances at once
        encoder_tokens_t = self.ms_t_norm(self.encoder_ms_t(input_tokens_t))
        encoder_tokens_f = self.ms_f_norm(self.encoder_ms_f(input_tokens_f))

        # cls_tokens = self.cls_fc(torch.cat([encoder_tokens_ecg[:, :1, :], encoder_tokens_pcg[:, :1, :]], dim=2))
        cls_tokens = encoder_tokens_t[:, :1, :] + encoder_tokens_f[:, :1, :]

        encoder_tokens_mc = torch.cat([encoder_tokens_t[:, 1:, :], encoder_tokens_f[:, 1:, :]], dim=1)
        encoder_tokens_mc = torch.cat([cls_tokens, encoder_tokens_mc], dim=1)
        encoder_tokens_mc = self.encoder_mc(encoder_tokens_mc)

        # Use global token (cls token) as instance features
        instance_features = encoder_tokens_mc[:, 0, :]  # (B*50, dim_tokens)
        
        if norm_dict is not None:
            # Apply denormalization to Transformer features (when mean, std are provided)
            mean_batched = norm_dict['mean'].view(B * num_instances, -1)  # (B*50, 1)
            std_batched = norm_dict['std'].view(B * num_instances, -1)    # (B*50, 1)
            
            # Expand to dim_tokens dimension: (B*50, 1) -> (B*50, dim_tokens)
            mean_expanded = mean_batched.expand(-1, instance_features.shape[-1])  # (B*50, dim_tokens)
            std_expanded = std_batched.expand(-1, instance_features.shape[-1])     # (B*50, dim_tokens)
            
            instance_features = instance_features * std_expanded + mean_expanded

        # Reshape back to bag format: (B*50, dim_tokens) -> (B, 50, dim_tokens)
        instance_features = instance_features.view(B, num_instances, -1)

        return_attention_weights = kwargs.get("return_attention_weights", False)
        preds = {
            domain: adapter(
                encoder_tokens=instance_features,
                return_embed=return_embed,
                return_attention_weights=return_attention_weights,
            )
            for domain, adapter in self.output_adapters.items()
        }
        return preds

        # preds = {
        #     domain: adapter(encoder_tokens=instance_features, return_embed=return_embed)  # pass (B, bag_size, dim_tokens)
        #     for domain, adapter in self.output_adapters.items()
        # }
        # return preds
