import argparse
import collections
import copy
import json
import math
import os
import re
import sys
import time
import traceback
from functools import partial
from typing import Any, List, Tuple, Union

import torch
import torch.distributed as dist

from ..constants import DS_INFERENCE, DS_INFERENCE_BLOOM_FP16, DS_INFERENCE_BLOOM_INT8, DS_ZERO, HF_ACCELERATE, DS_INFERENCE_BLOOM_INT8_SELF, DS_INFERNECE_BLOOM_FP16_SELF


# used for benchmarks
dummy_input_sentences = [
    "DeepSpeed is a machine learning framework",
    "He is working on",
    "He has a",
    "He got all",
    "Everyone is happy and I can",
    "The new movie that got Oscar this year",
    "In the far far distance from our galaxy,",
    "Peace is the only way",
]


def get_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()

    group = parser.add_argument_group(title="model")
    group.add_argument(
        "--deployment_framework", type=str, choices=[HF_ACCELERATE, DS_INFERENCE, DS_ZERO], default=HF_ACCELERATE
    )
    group.add_argument(
        "--model_name",
        type=str,
        required=True,
        help="model name to use",
    )
    group.add_argument(
        "--model_class",
        type=str,
        required=True,
        help="model class to use",
    )
    group.add_argument("--dtype", type=str, required=True, choices=["bf16", "fp16", "int8"], help="dtype for model")
    group.add_argument(
        "--generate_kwargs",
        type=str,
        default='{"min_length": 100, "max_new_tokens": 100, "do_sample": false}',
        help="generate parameters. look at https://huggingface.co/docs/transformers/v4.21.1/en/main_classes/text_generation#transformers.generation_utils.GenerationMixin.generate to see the supported parameters",
    )
    group.add_argument("--max_input_length", type=int, help="max input length")
    group.add_argument("--max_batch_size", type=int, help="max supported batch size")
    group.add_argument(
        "--cuda_visible_devices", nargs="*", type=int, default=list(range(8)), help="number of GPUs to use"
    )

    return parser


def parse_args(parser: argparse.ArgumentParser) -> argparse.Namespace:
    args = parser.parse_args()

    args.dtype = get_torch_dtype(args.dtype)
    args.generate_kwargs = json.loads(args.generate_kwargs)
    args.use_pre_sharded_checkpoints = args.model_name in [DS_INFERENCE_BLOOM_FP16, DS_INFERENCE_BLOOM_INT8, DS_INFERENCE_BLOOM_INT8_SELF,DS_INFERNECE_BLOOM_FP16_SELF]

    return args


def run_rank_n(func: partial, barrier: bool = False, rank: int = 0, other_rank_output: Any = None) -> Any:
    # runs function on only process with specified rank
    if dist.is_initialized():
        if dist.get_rank() == rank:
            output = func()
            if barrier:
                dist.barrier()
            return output
        else:
            if barrier:
                dist.barrier()
            return other_rank_output
    else:
        return func()


def print_rank_n(*values, rank: int = 0) -> None:
    # print on only process with specified rank
    if dist.is_initialized():
        if dist.get_rank() == rank:
            print(*values)
    else:
        print(*values)


def get_torch_dtype(dtype_str: str) -> torch.dtype:
    if dtype_str == "bf16":
        return torch.bfloat16
    elif dtype_str == "fp16":
        return torch.float16
    elif dtype_str == "int8":
        return torch.int8


def get_str_dtype(dtype_str: torch.dtype) -> str:
    if dtype_str == torch.bfloat16:
        return "bf16"
    elif dtype_str == torch.float16:
        return "fp16"
    elif dtype_str == torch.int8:
        return "int8"


def get_dummy_batch(batch_size: int, input_sentences: List[str] = None) -> List[str]:
    if input_sentences is None:
        input_sentences = copy.deepcopy(dummy_input_sentences)

    if batch_size > len(input_sentences):
        input_sentences *= math.ceil(batch_size / len(input_sentences))
    input_sentences = input_sentences[:batch_size]

    return input_sentences


def get_num_tokens_to_generate(max_new_tokens: int, allowed_max_new_tokens: int) -> int:
    if max_new_tokens is None:
        return allowed_max_new_tokens
    else:
        return min(max_new_tokens, allowed_max_new_tokens)


def run_and_log_time(execs: Union[List[partial], partial]) -> Tuple[Union[List[Any], Any], float]:
    # runs a function / list of functions and times them
    start_time = time.time()

    if type(execs) == list:
        results = []
        for f in execs:
            results.append(f())
    else:
        results = execs()

    time_elapsed = time.time() - start_time
    return results, time_elapsed


def pad_ids(arrays, padding, max_length=-1):
    # does left padding
    if max_length < 0:
        max_length = max(list(map(len, arrays)))

    arrays = [[padding] * (max_length - len(array)) + array for array in arrays]

    return arrays


def get_exception_response(query_id: int, method: str, debug: bool = False):
    e_type, e_message, e_stack_trace = sys.exc_info()
    response = {"error": str(e_type.__name__), "message": str(e_message), "query_id": query_id, "method": method}

    if debug:
        trace_back = traceback.extract_tb(e_stack_trace)

        # Format stacktrace
        stack_trace = []
        for trace in trace_back:
            stack_trace.append(
                "File : {}, Line : {}, Func.Name : {}, Message : {}".format(trace[0], trace[1], trace[2], trace[3])
            )

        response["stack_trace"] = stack_trace

    return response

def fetch_hostfile(hostfile_path):
    if not os.path.isfile(hostfile_path):
        logger.warning("Unable to find hostfile, will proceed with training "
                       "with local resources only.")
        return None

    # e.g., worker-0 slots=16
    with open(hostfile_path, 'r') as fd:
        hostfile_text = fd.readlines()

    return  _parse_hostfile(hostfile_text)


def _parse_hostfile(hostfile_lines):
    # Regex matches one or more non-whitespace characters (\S+) at the start of
    # the line, followed by one or more whitespace characters (\s+), followed
    # by the string "slots=", followed by one or more digits (\d+).
    pattern = r'^(\S+)\s+slots=(\d+)'

    resource_pool = collections.OrderedDict()
    for line in hostfile_lines:
        line = line.strip()
        match = re.search(pattern, line)
        if line.startswith("#") or line == "":
            # hostfile comment or empty line, ignore
            continue
        elif match:
            host = match.group(1)
            num_slots = int(match.group(2))
            if host in resource_pool:
                logger.error(f"Bad hostfile text: {hostfile_lines}")
                raise ValueError(
                    f"Hostfile contains multiple entries for {host}, unable to proceed with launching"
                )
            resource_pool[host] = num_slots
        else:
            logger.error(f"Bad hostfile text: {hostfile_lines}")
            raise ValueError(
                "Hostfile contains a bad entry: {line}, unable to proceed with launching"
            )

    if len(resource_pool) == 0:
        logger.error(f"Bad hostfile text: {hostfile_lines}")
        raise ValueError(
            "Hostfile is empty or not formatted correctly, unable to proceed with launching."
        )

    return resource_pool
