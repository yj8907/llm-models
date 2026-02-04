



import torch

from functools import partial
from typing import List, Optional, Tuple
from megatron.core import parallel_state
from megatron.training import inprocess_restart
from megatron.core.datasets.blended_megatron_dataset_builder import BlendedMegatronDatasetBuilder
from megatron.core.datasets.gpt_dataset import GPTDataset, GPTDatasetConfig, MockGPTDataset
from megatron.core.enums import ModelType
from megatron.core.models.gpt import GPTModel
from megatron.core.rerun_state_machine import get_rerun_state_machine
from megatron.core.utils import get_attr_wrapped_model, StragglerDetector
from megatron.core.tokenizers.text.utils.build_tokenizer import build_tokenizer
from megatron.training import get_args, get_timers, get_tokenizer, print_rank_0
from megatron.training.utils import (
    get_batch_on_this_cp_rank,
    get_batch_on_this_tp_rank,
    get_blend_and_blend_per_split,
    is_first_or_last_pipeline_stage,
)
from megatron.training.datasets.sft_dataset import SFTDataset
from model_provider import model_provider
from gpt_builders import gpt_builder

from megatron.training import get_args, get_timers, get_tokenizer, pretrain, print_rank_0

from megatron.core.models.gpt import GPTModel
from megatron.core.models.gpt.gpt_layer_specs import (
    get_gpt_decoder_block_spec,
    get_gpt_layer_local_spec,
    get_gpt_layer_with_transformer_engine_spec,
    get_gpt_mtp_block_spec,
)
from megatron.core.models.gpt.heterogeneous.heterogeneous_layer_specs import (
    get_gpt_heterogeneous_layer_spec,
)
from megatron.core.transformer.spec_utils import import_module
from megatron.training import get_args, print_rank_0
from megatron.training.arguments import core_transformer_config_from_args
from megatron.training.yaml_arguments import core_transformer_config_from_yaml

import megatron.legacy.model  # isort: skip


try:
    from megatron.post_training.arguments import add_modelopt_args
    from megatron.post_training.loss_func import loss_func as loss_func_modelopt

    has_nvidia_modelopt = True
except ImportError:
    has_nvidia_modelopt = False

stimer = StragglerDetector()


import time
# The earliest we can measure the start time.
_TRAIN_START_TIME = time.time()

# Startup timestamps for tracking program initialization phases
_STARTUP_TIMESTAMPS = {
    'program_start': None,  # Set by entry script before imports
    'main_entry': None,     # Set by entry script at start of __main__
    'pretrain_entry': None, # Set at top of pretrain()
}


def set_startup_timestamps(program_start=None, main_entry=None):
    """Set startup timestamps from the entry script.

    Call this after imports but before calling pretrain() to register
    the program start time and main entry time.

    Args:
        program_start: Timestamp captured at very start of program, before any imports.
        main_entry: Timestamp captured right after entering __main__ block.
    """
    global _TRAIN_START_TIME, _STARTUP_TIMESTAMPS
    if program_start is not None:
        _TRAIN_START_TIME = program_start
        _STARTUP_TIMESTAMPS['program_start'] = program_start
    if main_entry is not None:
        _STARTUP_TIMESTAMPS['main_entry'] = main_entry


from collections import defaultdict
import copy
import dataclasses
from datetime import datetime, timedelta
import functools
import gc
import inspect
import logging
import math
import os
import sys
from contextlib import nullcontext
from typing import Any, Optional, Dict

import torch.distributed

from megatron.core.optimizer.distrib_optimizer import DistributedOptimizer
from .log_handler import CustomHandler

# Make default logging level INFO, but filter out all log messages not from MCore.
logging.basicConfig(handlers=[CustomHandler()], level=logging.INFO)
from .theoretical_memory_usage import report_theoretical_memory

_LEGACY_TRAIN_START_TIME = time.time() # NOTE(asolergi-nv): Legacy timestamp

import torch

try:
    from megatron.rl import rl_utils
    has_rl_utils = True
except ImportError:
    has_rl_utils = False
from megatron.rl.parallel_utils import build_inference_pg_collection
try:
    from modelopt.torch.distill.plugins.megatron import (
        get_tensor_shapes_adjust_fn_for_distillation,
    )

    has_nvidia_modelopt = True
except ImportError:
    has_nvidia_modelopt = False

try:
    from nvidia_resiliency_ext.inprocess import CallWrapper
except ImportError:
    CallWrapper = type(None)


from megatron.core import mpu, tensor_parallel
from megatron.core.models.gpt.experimental_attention_variant_module_specs import (
    is_linear_attention_variant,
)
from megatron.core.utils import (
    check_param_hashes_across_dp_replicas,
    get_attr_wrapped_model,
    get_model_config,
    get_pg_size,
    get_pg_rank,
    StragglerDetector,
)
from megatron.core.fp8_utils import correct_amax_history_if_needed
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.pipeline_parallel.utils import (
    is_pp_first_stage,
    is_pp_last_stage,
    is_vp_first_stage,
    is_vp_last_stage,
)
from megatron.core.optimizer import get_standard_config_overrides
from megatron.training.checkpointing import load_checkpoint
from megatron.training.checkpointing import save_checkpoint, save_grads
from megatron.training.checkpointing import checkpoint_exists
from megatron.training.checkpointing import get_loaded_iteration
from megatron.core.full_cuda_graph import FullCudaGraphWrapper
from megatron.core.transformer.cuda_graphs import TECudaGraphHelper
from megatron.core.transformer.enums import CudaGraphScope
from megatron.core.transformer.module import Float16Module
from megatron.core.distributed import DistributedDataParallelConfig, TorchFullyShardedDataParallelConfig
from megatron.core.distributed import DistributedDataParallel as DDP
from megatron.core.distributed.fsdp.mcore_fsdp_adapter import FullyShardedDataParallel as megatron_FSDP
from megatron.core.optimizer.optimizer import param_group_identifier_keys

from megatron.core.optimizer.qk_clip import clip_qk

try:
    from megatron.core.distributed import TorchFullyShardedDataParallel as torch_FSDP

    HAVE_FSDP2 = True
except ImportError:
    HAVE_FSDP2 = False

from megatron.core.distributed import finalize_model_grads
from megatron.core.enums import ModelType
from megatron.core.optimizer import get_megatron_optimizer, AdamOptimizerConfig, SGDOptimizerConfig, OptimizerConfig, ParamKey
from megatron.core.optimizer.muon import get_megatron_muon_optimizer
from megatron.core.rerun_state_machine import (
    get_rerun_state_machine,
    destroy_rerun_state_machine,
    RerunDataIterator,
    RerunMode,
)
from megatron.training.initialize import initialize_megatron
from megatron.training.initialize import write_args_to_tensorboard
from megatron.training.initialize import set_jit_fusion_options
from megatron.training.utils import get_batch_on_this_cp_rank, get_batch_on_this_tp_rank
from megatron.training.datasets.data_samplers import build_pretraining_data_loader
from megatron.core.datasets.data_schedule import HybridCPDataLoaderWrapper
from megatron.core.optimizer_param_scheduler import OptimizerParamScheduler
from megatron.core.transformer.moe import upcycling_utils
from megatron.core.transformer.moe.moe_utils import track_moe_metrics, clear_aux_losses_tracker
from megatron.core.transformer.experimental_attention_variant.dsa import DSAIndexerLossLoggingHelper
from megatron.core.transformer.multi_token_prediction import MTPLossLoggingHelper
from megatron.core.parallel_state import (
    destroy_global_memory_buffer,
    destroy_global_symmetric_memory_buffer,
    destroy_model_parallel,
    update_pg_timeout
)
from megatron.core.inference.unified_memory import create_unified_mempool
from megatron.core.resharding.refit import swap_model_weights

from megatron.core.pipeline_parallel import get_forward_backward_func
from megatron.core.num_microbatches_calculator import (
    destroy_num_microbatches_calculator,
    get_current_global_batch_size,
    get_current_running_global_batch_size,
    get_num_microbatches,
    update_num_microbatches
)

from .async_utils import maybe_finalize_async_save
from .utils import (
    append_to_progress_log,
    calc_params_l2_norm,
    check_adlr_autoresume_termination,
    logical_and_across_model_parallel_group,
    reduce_max_stat_across_model_parallel_group,
    is_last_rank,
    print_rank_0,
    print_rank_last,
    report_memory,
    unwrap_model,
    update_use_dist_ckpt,
    to_empty_if_meta_device,
)
from .global_vars import (
    destroy_global_vars,
    get_args,
    get_signal_handler,
    get_timers,
    get_tensorboard_writer,
    get_wandb_writer,
    get_one_logger,
    get_tokenizer,
    get_energy_monitor,
)
from . import one_logger_utils
from .dgrad_logging import enable_dgrad_logging, disable_dgrad_logging, save_dgrads

from . import ft_integration

stimer = StragglerDetector()

from megatron.core.msc_utils import MultiStorageClientFeature, open_file



if __name__ == "__main__":

    # Temporary for transition to core datasets
    train_valid_test_datasets_provider.is_distributed = True

    # Optionally enable inprocess restart on pretrain
    pretrain, store = inprocess_restart.maybe_wrap_for_inprocess_restart(pretrain)

    pretrain(
        train_valid_test_datasets_provider,
        partial(model_provider, gpt_builder),
        ModelType.encoder_or_decoder,
        forward_step,
        args_defaults={'tokenizer_type': 'GPT2BPETokenizer'},
        extra_args_provider=add_modelopt_args if has_nvidia_modelopt else None,
        store=store,
    )
