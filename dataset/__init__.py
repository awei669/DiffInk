from .vae_dataset import TrainDataset, ValDataset
from .transform import Transform
from torch.utils.data import DataLoader, DistributedSampler

def build_datasets_and_loaders(config):
    transform = Transform(data_fixed_length=config.get("data_fixed_length", 1000),
                          prob=config.get("aug_prob", 0.5))

    train_dataset = TrainDataset(config["train_file"], text_file=config["text_file"], writer_file=config["writer_file"], transform=transform)
    val_dataset = ValDataset(config["val_file"], text_file=config["text_file"], writer_file=None, transform=None)

    config["num_text_embedding"] = len(train_dataset.text_cache) + 1 # 0 for padding
    config["num_writer"] = len(train_dataset.writer_cache)

    train_batch_size = config.get("train_batch_size", 1)
    val_batch_size = config.get("val_batch_size", 50)

    train_loader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True, num_workers=8,
                              collate_fn=TrainDataset.collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=val_batch_size, shuffle=True, num_workers=8,
                            collate_fn=ValDataset.collate_fn)

    return train_loader, val_loader, config

def build_test_datasets_and_loaders(config):
    val_dataset = ValDataset(config["val_file"], text_file=config["text_file"], writer_file=config["writer_file"], transform=None)

    config["num_text_embedding"] = len(val_dataset.text_cache) + 1 # 0 for padding

    val_batch_size = config.get("val_batch_size", 50)

    val_loader = DataLoader(val_dataset, batch_size=val_batch_size, shuffle=True, num_workers=8,
                            collate_fn=ValDataset.collate_fn)

    return val_loader, config

def build_datasets_and_loaders_ddp(config, ddp=False):
    transform = Transform(data_fixed_length=config.get("data_fixed_length", 1000),
                          prob=config.get("aug_prob", 0.5))

    train_dataset = TrainDataset(config["train_file"], text_file=config["text_file"], writer_file=config["writer_file"], transform=transform)
    val_dataset = ValDataset(config["val_file"], text_file=config["text_file"], writer_file=None, transform=None)

    config["num_text_embedding"] = len(train_dataset.text_cache) + 1  # 0 for padding
    config["num_writer"] = len(train_dataset.writer_cache)

    train_batch_size = config.get("train_batch_size", 1)
    val_batch_size = config.get("val_batch_size", 50)

    if ddp:
        train_sampler = DistributedSampler(train_dataset)
        shuffle = False
    else:
        train_sampler = None
        shuffle = True

    train_loader = DataLoader(train_dataset, batch_size=train_batch_size, sampler=train_sampler,
                              shuffle=shuffle, num_workers=8, collate_fn=TrainDataset.collate_fn)

    val_loader = DataLoader(val_dataset, batch_size=val_batch_size, shuffle=False, num_workers=8,
                            collate_fn=ValDataset.collate_fn)

    return train_loader, val_loader, config