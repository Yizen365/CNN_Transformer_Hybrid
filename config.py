from pathlib import Path

def get_config():
    return {
        'image_size': 224,
        'in_channels': 3,
        'num_epochs': 20,
        'class_size': 3,
        'learning_rate': 10**-4,
        'patch_size': 16,
        'embedding_dim': 256,
        'layers': 4,
        'heads': 4,
        'mlp_dim': 512,
        'batch_size': 32,
        'data_dir': 'data',
        'num_classes': 3,
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