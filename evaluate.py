import torch
import torch.nn.functional as F
from tqdm import tqdm
from attacks import gen_perturbed_image
from utils import get_data_min_max
import numpy as np

def evaluate_model(
        model,
        test_loader,
        loss_function,
        num_classes,
        adversarial=None,
        epsilon=0.003,
        adv_iterations=5,
    ):
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
            images = gen_perturbed_image(images, labels, loss_function, model, minv, maxv, method=adversarial, epsilon=epsilon, num_classes=num_classes, iterations=adv_iterations)

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
                targets_one_hot = F.one_hot(labels, num_classes=num_classes).permute(0, 3, 1, 2).float()

                dims = (0, 2, 3)
                intersection = torch.sum(probs * targets_one_hot, dims)
                cardinality = torch.sum(probs + targets_one_hot, dims)
                dice = (2. * intersection + 1e-7) / (cardinality + 1e-7)
                dice_score_total += dice.mean().item() * images.size(0)

    test_loss /= len(test_loader.dataset)
    test_dice = dice_score_total / len(test_loader.dataset)
    print(f'Test Loss: {test_loss:.4f} | Test Dice: {test_dice:.4f}')
    return test_dice, test_loss