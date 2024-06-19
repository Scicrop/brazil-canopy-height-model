import torch
def list_gpus():
    num_gpus = torch.cuda.device_count()
    print(f"Available GPUs: {num_gpus}")
    device_ids = []
    for i in range(num_gpus):
        print(f"GPU id: {i}, Name: {torch.cuda.get_device_name(i)}")
        device_ids.append(i)
    return device_ids