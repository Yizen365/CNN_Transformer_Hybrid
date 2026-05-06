from pathlib import Path

def get_config():
    return {
        'image_size': 224,
        'in_channels': 3,
        'num_epochs': 160,
        'cnn_unfreeze_epoch': 10,
        'class_size': 3,
        'learning_rate': 3e-4,
        'fine_tune_lr': 1e-5,
        'patch_size': 16,
        'd_model': 512,
        'layers': 4,
        'heads': 16,
        'mlp_dim': 2048,
        'batch_size': 32,
        'data_dir': 'lung_image_sets',
        'model_folder': 'weights',
        'model_basename': 'tmodel_',
        'preload': 'best',
        'experiment_name': 'runs/tmodel'
    }

def get_weights_file_path(config, epoch: str) -> str:
    model_folder = config['model_folder']
    model_basename = config['model_basename']
    model_filename = f"{model_basename}{epoch}.pt"
    return str(Path('.') / model_folder / model_filename)