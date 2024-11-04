import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import tqdm


def collect_feature(data_loader: DataLoader, classifier: nn.Module,
                    device: torch.device, max_num_features=None) -> torch.Tensor:
    """
    Fetch data from `data_loader`, and then use `feature_extractor` to collect features

    Args:
        data_loader (torch.utils.data.DataLoader): Data loader.
        feature_extractor (torch.nn.Module): A feature extractor.
        device (torch.device)
        max_num_features (int): The max number of features to return

    Returns:
        Features in shape (min(len(data_loader), max_num_features * mini-batch size), :math:`|\mathcal{F}|`).
    """
    classifier.eval()
    all_features_c = []
    all_features_t = []
    with torch.no_grad():
        for i, data in enumerate(tqdm.tqdm(data_loader)):
            if max_num_features is not None and i >= max_num_features:
                break
            inputs = data[0].to(device)
            feature_c, feature_t = classifier.forward_analysis(inputs)
            all_features_c.append(feature_c)
            all_features_t.append(feature_t)
    return torch.cat(all_features_c, dim=0), torch.cat(all_features_t, dim=0)

def collect_feature0(data_loader: DataLoader, feature_extractor: nn.Module,
                    device: torch.device, max_num_features=None) -> torch.Tensor:
    """
    Fetch data from `data_loader`, and then use `feature_extractor` to collect features
    Args:
        data_loader (torch.utils.data.DataLoader): Data loader.
        feature_extractor (torch.nn.Module): A feature extractor.
        device (torch.device)
        max_num_features (int): The max number of features to return
    Returns:
        Features in shape (min(len(data_loader), max_num_features * mini-batch size), :math:`|\mathcal{F}|`).
    """
    feature_extractor.eval()
    all_features = []
    with torch.no_grad():
        for i, data in enumerate(tqdm.tqdm(data_loader)):
            if max_num_features is not None and i >= max_num_features:
                break
            inputs = data[0].to(device)
            feature = feature_extractor(inputs).cpu()
            all_features.append(feature)
    return torch.cat(all_features, dim=0)