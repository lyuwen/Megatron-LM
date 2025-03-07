import logging

from megatron.training.global_vars import get_args


logger = logging.getLogger(__name__)

_global_throughput_average_bins = []

_benchmark_target_achieved = 0

_benchmark_should_exit = False


def reset_benchmark():
    global _global_throughput_average_bins
    global _benchmark_target_achieved
    _global_throughput_average_bins = []
    _benchmark_target_achieved = 0


def record_throughput(throughput, iteration):
    args = get_args()
    global _global_throughput_average_bins
    global _benchmark_target_achieved
    global _benchmark_should_exit
    if args.num_steps_average_throughput:
        _global_throughput_average_bins.append(throughput)
        _global_throughput_average_bins = _global_throughput_average_bins[-args.num_steps_average_throughput:]
        averaged_throughput = sum(_global_throughput_average_bins) / len(_global_throughput_average_bins)
        if args.benchmark_target_tflops is not None and (args.benchmark_check_begins <= iteration <= args.benchmark_check_begins):
            _benchmark_target_achieved += averaged_throughput >= args.benchmark_target_tflops
        if iteration == args.benchmark_check_ends:
            _benchmark_should_exit = args.benchmark_pass_action == ["continue", "stop"][min(1, _benchmark_target_achieved)]
        return averaged_throughput
    else:
        return None

def benchmark_should_exit(raise=True):
    global _benchmark_should_exit
    if _benchmark_should_exit:
        msg = "Benchmark exit condition has been met, will exit the training soon."
        if raise:
            raise RuntimeError(msg)
        logger.info(msg)
    return _benchmark_should_exit
