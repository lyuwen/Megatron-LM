import contextlib
from typing import Iterator, List, Union
from megatron.core.pipeline_parallel.or_schedule_core import (
    generate_1f1b_schedule,
    generate_s6_mb12_test_schedule,
    generate_s16_mb32_test_schedule,
    generate_s16_mb32_test_schedule_aa,
    generate_s16_mb32_test_schedule_ab,
    generate_communication_masks,
    generate_communication_masks_zerocross,    
    generate_communication_masks_1f1b_zerocross,
    parse_schedule
)

import os
import time
import torch
from torch.autograd.variable import Variable

from megatron.core import parallel_state
from megatron.core.enums import ModelType
from megatron.core.utils import (
    get_model_config,
    get_model_type,
    get_model_xattn,
    drain_embedding_wgrad_compute,
    get_attr_wrapped_model,
)
from megatron.core.pipeline_parallel.schedules import (
    forward_step,
    backward_step,
    get_tensor_shapes,
    deallocate_output_tensor,
    check_first_val_step,
    clear_embedding_activation_buffer,
    finish_embedding_wgrad_compute,
    recv_forward,
    recv_backward,
    send_forward,
    send_backward,
)
from megatron.core.transformer.cuda_graphs import create_cudagraphs
from dataclasses import dataclass
from megatron.training import get_args

@dataclass
class OrPipelineParams:
    schedule_type: str  # 可以是 '1f1b', 's16mb32_feelthepluse' 等

def or_pipelining(
    *,
    forward_step_func,
    data_iterator: Union[Iterator, List[Iterator]],
    model: Union[torch.nn.Module, List[torch.nn.Module]],
    num_microbatches: int,
    seq_length: int,
    micro_batch_size: int,
    decoder_seq_length: int = None,
    forward_only: bool = False,
    collect_non_loss_data: bool = False,
    first_val_step: bool = None,
    iteration: int = 0,
):
    args = get_args()
    pipeline_params = OrPipelineParams(
        schedule_type=args.schedule_type
    )

    time_begin = time.perf_counter()
    
    #print("NCCL_BUFFSIZE =", os.environ.get("NCCL_BUFFSIZE", "default (32MB)"))
    #print(f'time_begin------ert===== = {time_begin:.6f}')

    schedule_visualble = args.schedule_visualble_path is not None and iteration >= args.schedule_visual_iter_start and iteration <= args.schedule_visual_iter_end
    num_rank  = parallel_state.get_pipeline_model_parallel_world_size()
    this_rank = parallel_state.get_pipeline_model_parallel_rank()
    global_rank = torch.distributed.get_rank()

    if schedule_visualble:
        pipeline_rank_offset = global_rank % args.expert_model_parallel_size
        if pipeline_rank_offset == 0:
            pipeline_rank = global_rank // args.expert_model_parallel_size
    if global_rank == 0:
        print(f'Pipeline schedule type: {pipeline_params.schedule_type}')
    
    #frog
    if pipeline_params.schedule_type == '1f1b':
        data_schedule = generate_1f1b_schedule(num_rank, num_microbatches)
    elif pipeline_params.schedule_type == 's16mb32_ori' : 
        data_schedule = generate_s16_mb32_test_schedule()
    elif pipeline_params.schedule_type == 's16mb32_aa' : 
        data_schedule = generate_s16_mb32_test_schedule_aa()
    elif pipeline_params.schedule_type == 's16mb32_ab' : 
        data_schedule = generate_s16_mb32_test_schedule_ab()
    elif pipeline_params.schedule_type == 's6mb12_ori' :
        data_schedule = generate_s6_mb12_test_schedule()
    else:
        assert False 

    F_time = [1.0 for _ in range(num_rank)]
    B_time = [3.0 for _ in range(num_rank)]
    time_table = parse_schedule(data_schedule, F_time, B_time)

    masks = generate_communication_masks_zerocross(data_schedule, time_table)
    #masks = generate_communication_masks_1f1b_zerocross(data_schedule) 
    #masks = generate_communication_masks_zerocross(data_schedule)
    #masks = generate_communication_masks(data_schedule)

    data_schedule_mask_sendF = masks['sendF']
    data_schedule_mask_sendB = masks['sendB']
    data_schedule_mask_recvF = masks['recvF']
    data_schedule_mask_recvB = masks['recvB']

    if schedule_visualble:
        log_msgs = []
        id_forward = 0
        id_backward = 0
        
        if global_rank == 0:
            log_msg = f'iteration {iteration}'
            log_msgs.append(log_msg)
    
    if isinstance(model, list):
        assert (
            len(model) == 1
        ), "non-interleaved pipeline-parallel schedule does not support model chunking"
        model = model[0]
    if isinstance(data_iterator, list):
        assert (
            len(data_iterator) == 1
        ), "non-interleaved pipeline-parallel schedule does not support model chunking"
        data_iterator = data_iterator[0]

    config = get_model_config(model)
    if config.overlap_p2p_comm:
        raise ValueError(
            "Non-interleaved pipeline parallelism does not support overlapping p2p communication"
        )

    # Needed only when gradients are finalized in M-Core
    if config.finalize_model_grads_func is not None and not forward_only:
        embedding_module = clear_embedding_activation_buffer(config, model)

    if config.timers is not None:
        config.timers('forward-backward', log_level=1).start(barrier=config.barrier_with_L1_time)

    # Disable async grad reductions
    no_sync_func = config.no_sync_func
    if no_sync_func is None:
        no_sync_func = contextlib.nullcontext
    no_sync_context = None

    def disable_grad_sync():
        """Disable asynchronous grad reductions"""
        nonlocal no_sync_context
        if no_sync_context is None:
            no_sync_context = no_sync_func()
            no_sync_context.__enter__()

    def enable_grad_sync():
        """Enable asynchronous grad reductions"""
        nonlocal no_sync_context
        if no_sync_context is not None:
            no_sync_context.__exit__(None, None, None)
            no_sync_context = None

    disable_grad_sync()

    # Compute number of warmup microbatches.
    num_warmup_microbatches = (
        parallel_state.get_pipeline_model_parallel_world_size()
        - parallel_state.get_pipeline_model_parallel_rank()
        - 1
    )
    num_warmup_microbatches = min(num_warmup_microbatches, num_microbatches)
    #adjust warmup method
    #num_warmup_microbatches += 4
    if torch.distributed.get_rank() == 0:
        print(f"---- nwarm batch = {num_warmup_microbatches} all_microbatches = {num_microbatches} ----")
    num_microbatches_remaining = num_microbatches - num_warmup_microbatches

    # Checkpoint the activations of partial Transformer layers in a number of micro-batches
    # within the maximum outstanding micro-batch backpropagations.
    # Micro-batches with the ids less than 'num_microbatches_with_partial_activation_checkpoints'
    # checkpoint partial Transformer layers (or skip checkpointing) and
    # the rest of micro-batches within a window of micro-batches checkpoint
    # all Transformer layers. The window of micro-batches is set by the maximum
    # outstanding backpropagations and becomes smaller at later pipeline stages.
    # Please refer the appendix C in https://arxiv.org/pdf/2205.05198.pdf
    max_outstanding_backprops = None
    if config.num_microbatches_with_partial_activation_checkpoints is not None:
        max_outstanding_backprops = num_warmup_microbatches + 1
    #设成none 懒得管这个先
    #max_outstanding_backprops = None
    #checkpoint_activations_microbatch = None

    model_type = get_model_type(model)
    encoder_decoder_xattn = get_model_xattn(model)

    rank = parallel_state.get_pipeline_model_parallel_rank()
    recv_tensor_shapes = get_tensor_shapes(
        rank=rank - 1,
        model_type=model_type,
        seq_length=seq_length,
        micro_batch_size=micro_batch_size,
        decoder_seq_length=decoder_seq_length,
        config=config,
        encoder_decoder_xattn=encoder_decoder_xattn,
    )
    send_tensor_shapes = get_tensor_shapes(
        rank=rank,
        model_type=model_type,
        seq_length=seq_length,
        micro_batch_size=micro_batch_size,
        decoder_seq_length=decoder_seq_length,
        config=config,
        encoder_decoder_xattn=encoder_decoder_xattn,
    )

    # Input, output tensors only need to be saved when doing backward passes
    input_tensors = None
    input_tensors_mid = None
    output_tensors = None
    output_tensors_mid = None
    
    input_tensor_grads_mid = None
    output_tensor_grads_mid = None

    total_num_tokens = torch.tensor(0, dtype=torch.int).cuda()

    if not forward_only:
        input_tensors = []
        input_tensors_mid = []
        output_tensors = []
        output_tensors_mid = []
        output_tensor_grads_mid = []
        input_tensor_grads_mid = []

    forward_data_store = []

    #monkey
    if False:
        print('nothing ')
    else:
        cacunum_perrank = num_microbatches*2
        
        #init recv
        for _flag in range(data_schedule_mask_recvF[this_rank][0]):
            input_tensor = recv_forward(recv_tensor_shapes, config)
            input_tensors_mid.append(input_tensor)
        for _flag in range(data_schedule_mask_recvB[this_rank][0]):
            output_tensor_grads_mid.append(recv_backward(send_tensor_shapes, config))

        for i in range(cacunum_perrank):
            #print(f' ---- i = {i} ----- ')
            microbatch_id = data_schedule[this_rank][i][1]
            if max_outstanding_backprops is not None:
                checkpoint_activations_microbatch = (
                    i % max_outstanding_backprops
                    >= config.num_microbatches_with_partial_activation_checkpoints
                    )
            else:
                checkpoint_activations_microbatch = None
                
            if data_schedule[this_rank][i][0] == 'F':
                #input_tensor = recv_forward(recv_tensor_shapes, config)
                #input_tensors_mid.append(input_tensor)
                input_tensor = input_tensors_mid.pop(0)
                #------------>>>
                if schedule_visualble:
                    if pipeline_rank_offset == 0:
                        time_current = time.perf_counter()
                        time_point   = time_current - time_begin
                        log_msg = f'S {pipeline_rank} F {id_forward} begin {time_point:.6f}'
                        log_msgs.append(log_msg)
                #id_forward += 1
                #with open(file_path, 'a') as f:
                #     f.write(log_msg + '\n')
                #---------------
                #print(f'[stable occur forward ] rank = {this_rank} i = {i} gmm = {gmm}  --------')
                #print(f'[forward occur] checkpoint_actmicro = {checkpoint_activations_microbatch} ---- param = {config.num_microbatches_with_partial_activation_checkpoints}')
                output_tensor, num_tokens = forward_step(
                    forward_step_func,
                    data_iterator,
                    model,
                    num_microbatches,
                    input_tensor,
                    forward_data_store,
                    config,
                    collect_non_loss_data,
                    checkpoint_activations_microbatch,
                    check_first_val_step(
                        first_val_step, forward_only, (i == 0) and (num_warmup_microbatches == 0)
                    ),
                    current_microbatch=microbatch_id,
                    encoder_decoder_xattn=encoder_decoder_xattn,
                )
                #print(f'[stable finish forward ] rank = {this_rank} i = {i} gmm = {gmm}  --------')
                #------------>>>
                if schedule_visualble:
                    if pipeline_rank_offset == 0:
                        time_current = time.perf_counter()
                        time_point   = time_current - time_begin
                        log_msg = f'S {pipeline_rank} F {id_forward} end {time_point:.6f}'
                        log_msgs.append(log_msg)
                        id_forward += 1
                #with open(file_path, 'a') as f:
                #     f.write(log_msg + '\n')
                #---------------
                total_num_tokens += num_tokens.item()
                output_tensors_mid.append(output_tensor);

                #output_tensor = output_tensors_mid.pop(0);
                #send_forward(output_tensor, send_tensor_shapes, config)
                #input_tensors.append(input_tensor)
                #output_tensors.append(output_tensor)
                #deallocate_output_tensor(output_tensor[0], config.deallocate_pipeline_outputs)

            elif data_schedule[this_rank][i][0] == 'B':
                #output_tensor_grad = output_tensor_grads_mid.pop(0);
                # Pop input_tensor and output_tensor from the start of the list for
                # the backward pass.
                input_tensor = input_tensors.pop(0)
                output_tensor = output_tensors.pop(0)

                # Enable grad sync for the last microbatch in the batch if the full
                # backward pass completes in the 1F1B stage.
                # rank 0 的最后一个操作一定是反向
                #if num_warmup_microbatches == 0 and last_iteration:
                #    if config.grad_sync_func is None or rank == 0:
                if this_rank == 0 and i == cacunum_perrank - 1:
                    enable_grad_sync()

                #------------>>>
                if schedule_visualble:
                    if pipeline_rank_offset == 0:
                        time_current = time.perf_counter()
                        time_point   = time_current - time_begin
                        log_msg = f'S {pipeline_rank} B {id_backward} begin {time_point:.6f}'
                        log_msgs.append(log_msg)
                #id_backward += 1
                #with open(file_path, 'a') as f:
                #     f.write(log_msg + '\n')
                #---------------
                
                #print(f'[stable occur backward ] rank = {this_rank} i = {i} gmm = {gmm}  --------')
                input_tensor_grads_mid.append( backward_step(
                    input_tensor, output_tensor, output_tensor_grads_mid.pop(0), model_type, config
                ))
                #print(f'[stable finish backward ] rank = {this_rank} i = {i} gmm = {gmm}  --------')
                
                #------------>>>
                if schedule_visualble:
                    if pipeline_rank_offset == 0:
                        time_current = time.perf_counter()
                        time_point   = time_current - time_begin
                        log_msg = f'S {pipeline_rank} B {id_backward} end {time_point:.6f}'
                        log_msgs.append(log_msg)
                        id_backward += 1
                #with open(file_path, 'a') as f:
                #     f.write(log_msg + '\n')
                #---------------
                if i == cacunum_perrank-1:
                    input_tensor = None

            else:
                assert False
            
            for _flag in range(data_schedule_mask_sendF[this_rank][i]):
                output_tensor = output_tensors_mid.pop(0);
                send_forward(output_tensor, send_tensor_shapes, config)
                input_tensors.append(input_tensor)
                output_tensors.append(output_tensor)
                deallocate_output_tensor(output_tensor[0], config.deallocate_pipeline_outputs)
            if i < cacunum_perrank - 1:
                for _flag in range(data_schedule_mask_recvF[this_rank][i+1]):
                    input_tensor = recv_forward(recv_tensor_shapes, config)
                    input_tensors_mid.append(input_tensor)

            for _flag in range(data_schedule_mask_sendB[this_rank][i]):
                send_backward(input_tensor_grads_mid.pop(0), recv_tensor_shapes, config)
            if i < cacunum_perrank - 1:
                for _flag in range(data_schedule_mask_recvB[this_rank][i+1]):
                    output_tensor_grads_mid.append(recv_backward(send_tensor_shapes, config))

        # Launch any remaining grad reductions.
        if no_sync_context is not None:
            enable_grad_sync()
            if config.grad_sync_func is not None:
                config.grad_sync_func(model.parameters())

        if config.finalize_model_grads_func is not None and not forward_only:

            # If defer_embedding_wgrad_compute is enabled we need to do the
            # weight gradient GEMM's here.
            finish_embedding_wgrad_compute(config, embedding_module)

            # Finalize model grads (perform full grad all-reduce / reduce-scatter for
            # data parallelism, layernorm all-reduce for sequence parallelism, and
            # embedding all-reduce for pipeline parallelism).
            config.finalize_model_grads_func(
                [model], total_num_tokens if config.calculate_per_token_loss else None
            )

        if config.timers is not None:
            config.timers('forward-backward').stop()

        if hasattr(config, 'enable_cuda_graph') and config.enable_cuda_graph:
            create_cudagraphs()

    if schedule_visualble:
        if global_rank == (num_rank-args.expert_model_parallel_size):
            iteration += 1
            log_msg = f'iteration {iteration}'
            log_msgs.append(log_msg)
        with open(args.schedule_visualble_path, 'a') as f:
            for msg in log_msgs:
                f.write(msg + '\n')
                
    return forward_data_store
    """Your custom pipeline parallel schedule implementation.
    
    Args:
        forward_step_func: Function to perform forward step
        data_iterator: Iterator over the data
        model: The model to train
        num_microbatches: Number of microbatches
        seq_length: Sequence length
        micro_batch_size: Size of each microbatch
        decoder_seq_length: Length of decoder sequence (if applicable)
        forward_only: Whether to only perform forward pass
        collect_non_loss_data: Whether to collect non-loss data
        first_val_step: Whether this is the first validation step
        
    Returns:
        forward_data_store: Dictionary containing forward pass data
    """
    return forward_data_store 

#  加这段代码到ifib开头
#    from megatron.core.pipeline_parallel.or_schedule import or_pipelining
#    if True:
#        or_pipelining_result = or_pipelining(
#            forward_step_func=forward_step_func,
#            data_iterator=data_iterator,
#            model=model,
#            num_microbatches=num_microbatches,
#            seq_length=seq_length,
#            micro_batch_size=micro_batch_size,
#            decoder_seq_length=decoder_seq_length,
#            forward_only=forward_only,
#            collect_non_loss_data=collect_non_loss_data,
#            first_val_step=first_val_step,
#        )
#        return or_pipelining_result 
