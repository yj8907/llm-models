
from megatron.core.transformer.transformer_layer import TransformerLayer, TransformerLayerSubmodules

from megatron.core.transformer.transformer_config import TransformerConfig
from typing import Any, Dict, Optional, Union
from megatron.core.process_groups_config import ProcessGroupCollection

import torch

class BidirTransformerLayer(TransformerLayer):

    def __init__(self,
        config: TransformerConfig,
        submodules: TransformerLayerSubmodules,
        layer_number: int = 1,
        hidden_dropout: Optional[float] = None,
        pg_collection: Optional[ProcessGroupCollection] = None,
        vp_stage: Optional[int] = None
    ):
        super().__init__(
            config = config, 
            submodules = submodules,
            layer_number = layer_number,
            hidden_dropout = hidden_dropout,
            pg_collection = pg_collection,
            vp_stage = vp_stage
        )

        self.is_final_layer = layer_number == config.num_layers


    def forward(self, *args, **kwargs):
        """
        Perform a forward pass through the transformer layer.

        This method calls the core computation of a transformer layer, including
        self-attention, cross-attention (if applicable), and feed-forward operations.
        """
        # Remove 'dynamic_inference_decode_only' from kwargs if present
        # this is only used to uniquely identify decode and non-decode cuda graph
        # runners in the cuda graph manager
        kwargs.pop("dynamic_inference_decode_only", None)

        if 'hidden_states' not in kwargs:
            hidden_states = args[0]
            other_args = args[1:]
            if self.layer_number == 1:
                hidden_states = torch.cat((hidden_states, hidden_states), dim=1)
            
            packed_seq_params = kwargs.pop("packed_seq_params", None)

            hidden_states, context = self._forward_attention(
                hidden_states,
                *other_args,
                packed_seq_params=packed_seq_params,
                **kwargs,
            )
        else:
            if self.layer_number == 1:
                kwargs['hidden_states'] = torch.cat((kwargs['hidden_states'], kwargs['hidden_states']), dim=1)
            hidden_states, context = self._forward_attention(*args, **kwargs)

        output = self._forward_mlp(
            hidden_states,
            kwargs.get("inference_context", None),
            padding_mask=kwargs.get("padding_mask", None),
        )

        if self.is_final_layer and isinstance(output, torch.Tensor):
            output, _ = torch.chunk(output, 2, dim=1)

        return output, context