from pathlib import Path

def get_config():
    return {
        'image_size': 28,
        'in_channels': 3,
        'trans_in_channels': 64,
        'num_epochs': 80,
        'class_size': 10,
        'learning_rate': 3e-4,
        'patch_size': 7,
        'd_model': 512,
        'layers': 8,
        'heads': 32,
        'mlp_dim': 2048,
        'batch_size': 32,
        'data_dir': 'testing',
        'model_folder': 'weights',
        'model_basename': 'tmodel_',
        'preload': None,
        'experiment_name': 'runs/tmodel'
    }

def get_weights_file_path(config, epoch: str) -> str:
    model_folder = config['model_folder']
    model_basename = config['model_basename']
    model_filename = f"{model_basename}{epoch}.pt"
    return str(Path('.') / model_folder / model_filename)