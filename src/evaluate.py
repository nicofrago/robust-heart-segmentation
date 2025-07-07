import torch
import torch.nn.functional as F
from tqdm import tqdm
from src.attacks import gen_perturbed_image
from src.utils import get_data_min_max
import numpy as np
from torch.utils.data import DataLoader

def evaluate_model(
        model: torch.nn.Module,
        test_loader: DataLoader,
        loss_function: torch.nn.Module,
        num_classes: int,
        adversarial: str = None,
        epsilon: float = 0.003,
        adv_iterations: int = 5,
)-> tuple:
    """
    Evaluate the model performance on the test dataset using Dice Score.
    
    Args:
        model: The neural network model to evaluate.
        test_loader: The data loader for the test dataset.
        loss_function: The loss function to calculate the loss.
        num_classes: The number of classes in the dataset.
        adversarial: The adversarial attack method (default is None).
        epsilon: The epsilon value for adversarial perturbation (default is 0.003).
        adv_iterations: The number of iterations for adversarial attack (default is 5).
        
    Returns:
        Tuple of test Dice Score and test loss.
    """
    # Test Evaluation with Dice Score
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    test_loss = 0.0
    dice_score_total = 0.0

    minv, maxv = get_data_min_max(test_loader.dataset)

    for images, labels in tqdm(test_loader):
        images = images.to(device)
        labels = labels.to(device)

        if adversarial is not None:
            images = gen_perturbed_image(
                 images, labels, loss_function, model, minv, maxv, 
                 method=adversarial, epsilon=epsilon, num_classes=num_classes, 
                 iterations=adv_iterations
            )

        with torch.no_grad():
                outputs = model(images)

                if isinstance(outputs, (tuple, list)):
                    outputs = outputs[0]
                else:
                    outputs = outputs

                loss = loss_function(outputs, labels)
                test_loss += loss.item() * images.size(0)

                # Compute Dice score
                probs = F.softmax(outputs, dim=1)
                targets_one_hot = F.one_hot(
                     labels, num_classes=num_classes
                ).permute(0, 3, 1, 2).float()

                dims = (0, 2, 3)
                intersection = torch.sum(probs * targets_one_hot, dims)
                cardinality = torch.sum(probs + targets_one_hot, dims)
                dice = (2. * intersection + 1e-7) / (cardinality + 1e-7)
                dice_score_total += dice.mean().item() * images.size(0)

    test_loss /= len(test_loader.dataset)
    test_dice = dice_score_total / len(test_loader.dataset)
    print(f'Test Loss: {test_loss:.4f} | Test Dice: {test_dice:.4f}')
    return test_dice, test_loss