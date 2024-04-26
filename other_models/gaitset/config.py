conf = {
    "CUDA_VISIBLE_DEVICES": "0",
    "data": {
        'dataset_path': "",
        'resolution': '64',
        'dataset': 'gait',
        'pid_shuffle': False,
    },
    "model": {
        'hidden_dim': 256,
        'batch_size': (10, 12),
        'restore_iter': 20000,
        'total_iter': 80000,
        'margin': 0.2,
        'num_workers': 4,
        'frame_num': 1,
        'model_name': 'gait',
    },
}
