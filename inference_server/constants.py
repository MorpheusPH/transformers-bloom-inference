# inference method (args.deployment_framework)
HF_ACCELERATE = "hf_accelerate"
DS_INFERENCE = "ds_inference"
DS_ZERO = "ds_zero"

# model weights
DS_INFERENCE_BLOOM_FP16 = "microsoft/bloom-deepspeed-inference-fp16"
DS_INFERENCE_BLOOM_INT8 = "microsoft/bloom-deepspeed-inference-int8"
DS_INFERNECE_BLOOM_FP16_SELF = "/home/ubuntu/.cache/bloom-fp16-shards"
DS_INFERENCE_BLOOM_INT8_SELF = "/home/ubuntu/.cache/bloom-int8-shards"

# GRPC_MAX_MSG_SIZE = 2**30  # 1GB
