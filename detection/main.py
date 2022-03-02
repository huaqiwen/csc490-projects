import os
import random
from typing import Dict, Optional

import numpy as np
import torch
from pylab import plt
from tqdm import tqdm

from detection.dataset import PandasetDataset, custom_collate
from detection.metrics.evaluator import Evaluator
from detection.model import DetectionModel, DetectionModelConfig
from detection.modules.loss_function import DetectionLossFunction
from detection.pandaset.dataset import DETECTING_CLASSES
from detection.pandaset.util import LabelClass
from detection.utils.visualization import visualize_detections

torch.multiprocessing.set_sharing_strategy('file_system')

def overfit(
    data_root: str,
    output_root: str,
    seed: int = 42,
    num_iterations: int = 500,
    log_frequency: int = 100,
    learning_rate: float = 1e-4,
) -> None:
    """Overfit detector to one frame of the Pandaset dataset.

    Args:
        data_root: The root directory of the Pandaset dataset.
        output_root: The root directory to output visualizations and checkpoints.
        seed: A fixed random seed for reproducibility.
        num_iterations: The number of iterations to run overfitting for.
        log_frequency: The number of training iterations between logs/visualizations.
        learning_rate: The learning rate for training the detection model.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.random.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.random.manual_seed(seed)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    os.makedirs(output_root, exist_ok=True)
    
    # setup model for all classes
    model_config = DetectionModelConfig()
    class_models = {class_: DetectionModel(model_config).to(device) for class_ in DETECTING_CLASSES}

    # setup data
    dataset = PandasetDataset(data_root, model_config)
    dataloader = torch.utils.data.DataLoader(dataset, collate_fn=custom_collate)

    # setup loss function and optimizer for all classes
    class_loss_fns = {class_: DetectionLossFunction(model_config.loss) 
        for class_ in DETECTING_CLASSES}
    class_optimizers = {class_: torch.optim.Adam(class_models[class_].parameters(), lr=learning_rate) 
        for class_ in DETECTING_CLASSES}

    # start training
    bev_lidar, class_bev_targets, class_labels = next(iter(dataloader))
    bev_lidar = bev_lidar.to(device)

    for class_ in class_bev_targets.keys():
        print(f"Overfitting class: {class_.value}")

        # define targets and labels for the current class
        bev_targets, labels = class_bev_targets[class_], class_labels[0][class_]
        bev_targets = bev_targets.to(device)

        # define model, loss_fn, and optimizer for the current class
        model = class_models[class_]
        loss_fn, optimizer = class_loss_fns[class_], class_optimizers[class_]

        for idx in tqdm(range(num_iterations)):
            model.train()
            predictions = model(bev_lidar)
            loss, loss_metadata = loss_fn(predictions, bev_targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # inference on the training example, and save vis results
            if (idx + 1) % log_frequency == 0:
                print(
                    f"[{idx}/{num_iterations}]: "
                    f"Loss - {loss.item():.4f} "
                    f"Heatmap Loss - {loss_metadata.heatmap_loss.item():.4f} "
                    f"Offset Loss - {loss_metadata.offset_loss.item():.4f} "
                    f"Size Loss - {loss_metadata.size_loss.item():.4f} "
                    f"Heading Loss - {loss_metadata.heading_loss.item():.4f} "
                )

                # visualize target heatmap
                target_heatmap = bev_targets[0, 0].cpu().detach().numpy()
                plt.matshow(target_heatmap, origin="lower")
                plt.savefig(f"{output_root}/target_heatmap_{class_.value}.png")

                # visualize predicted heatmap
                predicted_heatmap = predictions[0, 0].cpu().detach().sigmoid().numpy()
                plt.matshow(predicted_heatmap, origin="lower")
                plt.savefig(f"{output_root}/predicted_heatmap_{class_.value}.png")

    # training finished, make inference on all classes
    class_detections = {}
    for class_ in class_bev_targets.keys():
        with torch.no_grad():
            class_models[class_].eval()
            class_detections[class_] = class_models[class_].inference(bev_lidar[0].to(device))

    # visualize detections and ground truth
    lidar = bev_lidar[0].sum(0).nonzero().detach().cpu()[:, [1, 0]]
    visualize_detections(lidar, class_detections, class_labels[0])
    plt.savefig(f"{output_root}/detections.png")
    plt.close("all")


def train(
    data_root: str,
    output_root: str,
    seed: int = 42,
    batch_size: int = 2,
    num_workers: int = 8,
    num_epochs: int = 5,
    log_frequency: int = 100,
    learning_rate: float = 1e-4,
    checkpoint_path: Optional[str] = None,
) -> None:
    """Train detector on the Pandaset dataset.

    Args:
        data_root: The root directory of the Pandaset dataset.
        output_root: The root directory to output visualizations and checkpoints.
        seed: A fixed random seed for reproducibility.
        batch_size: The batch size per training iteration.
        num_workers: The number of dataloader workers.
        num_epochs: The number of epochs to run training over.
        log_frequency: The number of training iterations between logs/visualizations.
        learning_rate: The learning rate for training the detection model.
        checkpoint_path: Optionally, whether to initialize the model from a checkpoint.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.random.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.random.manual_seed(seed)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    os.makedirs(output_root, exist_ok=True)

    # setup model
    model_config = DetectionModelConfig()
    model = DetectionModel(model_config)
    if checkpoint_path is not None:
        model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
    model = model.to(device)

    # setup data
    dataset = PandasetDataset(data_root, model_config)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=custom_collate,
    )

    # setup loss function and optimizer
    loss_fn = DetectionLossFunction(model_config.loss)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # start training
    for epoch in range(num_epochs):
        for idx, (bev_lidar, bev_targets, labels) in tqdm(enumerate(dataloader)):
            model.train()
            predictions = model(bev_lidar.to(device))
            loss, loss_metadata = loss_fn(predictions, bev_targets.to(device))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # inference on the training example, and save vis results
            if (idx + 1) % log_frequency == 0:
                print(
                    f"Epoch {epoch} [{idx}/{len(dataloader)}]: "
                    f"Loss - {loss.item():.4f} "
                    f"Heatmap Loss - {loss_metadata.heatmap_loss.item():.4f} "
                    f"Offset Loss - {loss_metadata.offset_loss.item():.4f} "
                    f"Size Loss - {loss_metadata.size_loss.item():.4f} "
                    f"Heading Loss - {loss_metadata.heading_loss.item():.4f} "
                )

                # visualize target heatmap
                target_heatmap = bev_targets[0, 0].cpu().detach().numpy()
                plt.matshow(target_heatmap, origin="lower")
                plt.savefig(f"{output_root}/target_heatmap.png")

                # visualize predicted heatmap
                predicted_heatmap = predictions[0, 0].cpu().detach().sigmoid().numpy()
                plt.matshow(predicted_heatmap, origin="lower")
                plt.savefig(f"{output_root}/predicted_heatmap.png")

                # visualize detections and ground truth
                with torch.no_grad():
                    model.eval()
                    detections = model.inference(bev_lidar[0].to(device))
                lidar = bev_lidar[0].sum(0).nonzero().detach().cpu()[:, [1, 0]]
                visualize_detections(lidar, detections, labels[0])
                plt.savefig(f"{output_root}/detections.png")
                plt.close("all")

        torch.save(model.state_dict(), f"{output_root}/{epoch:03d}.pth")


@torch.no_grad()
def test(
    data_root: str,
    output_root: str,
    seed: int = 42,
    num_workers: int = 8,
    checkpoint_path: Optional[str] = None,
) -> None:
    """Visualize the outputs of the detector on Pandaset.

    Args:
        data_root: The root directory of the Pandaset dataset.
        output_root: The root directory to output visualizations and checkpoints.
        seed: A fixed random seed for reproducibility.
        num_workers: The number of dataloader workers.
        checkpoint_path: Optionally, whether to initialize the model from a checkpoint.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.random.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.random.manual_seed(seed)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    os.makedirs(output_root, exist_ok=True)

    # Split checkpoint path for all classes
    class_checkpoint_paths = {}
    if checkpoint_path is not None:
        for path in checkpoint_path.split("\\"):
            class_ = path[4: path.index('.pth')]
            class_checkpoint_paths[class_] = path

    # Setup model for all classes
    model_config = DetectionModelConfig()
    class_models = init_multiclass_models(model_config, device, checkpoint_path)

    # setup data
    dataset = PandasetDataset(data_root, model_config, test=True)
    dataloader = torch.utils.data.DataLoader(
        dataset, num_workers=num_workers, collate_fn=custom_collate
    )

    for idx, (bev_lidar, _, class_labels) in tqdm(enumerate(dataloader)):
        class_detections = {}

        # For each class, use the corresponding class model to detect
        for class_ in DETECTING_CLASSES:
            model = class_models[class_]
            model.eval()
            detections = model.inference(bev_lidar[0].to(device))
            class_detections[class_] = detections

        lidar = bev_lidar[0].sum(0).nonzero().detach().cpu()[:, [1, 0]]
        visualize_detections(lidar, class_detections, class_labels[0])
        plt.savefig(f"{output_root}/{idx:03d}.png")
        plt.close("all")


@torch.no_grad()
def evaluate(
    data_root: str,
    output_root: str,
    seed: int = 42,
    num_workers: int = 8,
    checkpoint_path: Optional[str] = None,
) -> None:
    """Evaluate the detector on Pandaset and save its metrics.

    Args:
        data_root: The root directory of the Pandaset dataset.
        output_root: The root directory to output visualizations and checkpoints.
        seed: A fixed random seed for reproducibility.
        num_workers: The number of dataloader workers.
        checkpoint_path: Optionally, whether to initialize the model from a checkpoint.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.random.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.random.manual_seed(seed)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    os.makedirs(output_root, exist_ok=True)

    # setup model
    model_config = DetectionModelConfig()
    model = DetectionModel(model_config)
    if checkpoint_path is not None:
        model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
    model = model.to(device)

    # setup data
    dataset = PandasetDataset(data_root, model_config, test=True)
    dataloader = torch.utils.data.DataLoader(
        dataset, num_workers=num_workers, collate_fn=custom_collate
    )

    evaluator = Evaluator(ap_thresholds=[2.0, 4.0, 8.0, 16.0])
    for _, (bev_lidar, _, labels) in tqdm(enumerate(dataloader)):
        model.eval()
        detections = model.inference(bev_lidar[0].to(device))
        evaluator.append(detections.to(torch.device("cpu")), labels[0])

    result = evaluator.evaluate()
    result_df = result.as_dataframe()
    with open(f"{output_root}/result.csv", "w") as f:
        f.write(result_df.to_csv())

    result.visualize()
    plt.savefig(f"{output_root}/results.png")
    plt.close("all")

    print(result_df)


@torch.no_grad()
def init_multiclass_models(
    model_config: DetectionModelConfig, 
    device, 
    checkpoint_path: Optional[str] = None
) -> Dict[LabelClass, DetectionModel]:
    # Split checkpoint path for all classes
    class_checkpoint_paths = {}
    if checkpoint_path is not None:
        for path in checkpoint_path.split("\\"):
            class_ = path[4: path.index('.pth')]
            class_checkpoint_paths[class_] = path

    # Setup model for all classes
    class_models = {}
    for class_ in DETECTING_CLASSES:
        model = DetectionModel(model_config)
        if checkpoint_path is not None and class_ in class_checkpoint_paths:
            model.load_state_dict(torch.load(class_checkpoint_paths[class_], map_location="cpu"))
        model = model.to(device)
        class_models[class_] = model
    
    return class_models


if __name__ == "__main__":
    import fire

    fire.Fire()
