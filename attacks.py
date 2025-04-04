import torch
import torch.nn.functional as F

def gen_perturbed_image(
        im,
        lb,
        loss_function,
        model,
        minv,
        maxv,
        method='FSGM',
        epsilon=0.003,
        num_classes=8,
        iterations=3
):
    if method == 'FSGM':
        return FSGM(im, lb, loss_function, model, minv, maxv, epsilon=epsilon)
    elif method == 'PGD':
        return PGD(im, lb, loss_function, model, minv, maxv, epsilon=epsilon, iterations=iterations)
    elif method == 'DAG':
        return DAG(im, lb, num_classes, model, minv, maxv, epsilon=epsilon, iterations=iterations)
    else:
        raise ValueError(f"Unknown method: {method}. Use 'FSGM' or 'PGD' or 'DAG'.")
    

def FSGM(
        im,
        lb,
        loss_function,
        model,
        minv,
        maxv,
        epsilon=0.003
):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    minv = torch.tensor(minv).to(device)
    maxv = torch.tensor(maxv).to(device)

    with torch.enable_grad():
        im_g = im.clone().detach().requires_grad_()
        model.zero_grad()
        output = model(im_g)

        if isinstance(output, (tuple, list)):
            output = output[0]
        else:
            output = output

        loss = loss_function(output, lb)
        loss.backward()

        perturbation = epsilon * torch.sign(im_g.grad)
        perturbed_image = im + perturbation

        # Ensure the perturbed image is within the valid range
        perturbed_image = torch.maximum(torch.minimum(perturbed_image, im + epsilon), im - epsilon)
        perturbed_image = torch.clamp(perturbed_image, minv, maxv)

    return perturbed_image

def PGD(
    im,
    lb,
    loss_function,
    model,
    minv,
    maxv,
    epsilon=0.003,
    iterations=3
):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    minv = torch.tensor(minv).to(device)
    maxv = torch.tensor(maxv).to(device)

    alpha = epsilon / iterations

    # initial perturbation
    perturbed_image = im.clone().detach() + torch.empty_like(im).uniform_(-epsilon, epsilon)
    perturbed_image = torch.maximum(torch.minimum(perturbed_image, im + epsilon), im - epsilon) # same than a clamp
    perturbed_image = torch.clamp(perturbed_image, minv, maxv)
    perturbed_image.requires_grad_()

    for i in range(iterations):
        model.zero_grad()

        output = model(perturbed_image)
        loss = loss_function(output, lb)
        loss.backward()

        with torch.no_grad():
            perturbation = alpha * torch.sign(perturbed_image.grad)
            perturbed_image+= perturbation
            perturbed_image = torch.maximum(torch.minimum(perturbed_image, im + epsilon), im - epsilon) # same than a clamp
            perturbed_image = torch.clamp(perturbed_image, minv, maxv)

        perturbed_image = perturbed_image.detach().requires_grad_()

    return perturbed_image

def DAG(
    im,
    lb,
    num_classes,
    model,
    minv,
    maxv,
    epsilon=0.1,
    iterations=5
):
    # Compute min and max values for the dataset
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    minv = torch.tensor(minv).to(device)
    maxv = torch.tensor(maxv).to(device)

    # Generate random target different from ground truth
    target = torch.randint(0, num_classes, lb.shape, device=lb.device)
    mask_same = target == lb
    target[mask_same] = (target[mask_same] + 1) % num_classes

    # Hyperparameters
    # alpha = epsilon / iterations
    alpha = 1

    # Setup input image
    adv_x = im.clone().detach().requires_grad_()

    for i in range(iterations):
        model.zero_grad()

        # Forward pass
        logits = model(adv_x)  # shape [B, C, H, W]
        if isinstance(logits, (tuple, list)):
            logits = logits[0]
        else:
            logits = logits
        pred = torch.argmax(logits, dim=1)  # shape [B, H, W]

        # Mask where prediction is NOT equal to the target
        mask = pred != target  # shape [B, H, W]
        if mask.sum() == 0:
            break  # early stop: already fooled

        # Get logits for target and predicted classes
        logits_perm = logits.permute(0, 2, 3, 1)  # shape [B, H, W, C]
        logits_target = logits_perm[mask, target[mask]]  # shape [N]
        logits_pred   = logits_perm[mask, pred[mask]]    # shape [N]

        # Compute DAG loss
        # loss = (logits_target - logits_pred).sum()
        loss = (logits_target - logits_pred).sum() / (mask.sum() + 1e-8) # normalize loss

        loss.backward()

        with torch.no_grad():
            grad = adv_x.grad
            grad = grad / (grad.norm(p=2) + 1e-8)
            adv_x = adv_x + alpha * grad

            # Project into epsilon ball
            adv_x = torch.clamp(adv_x, im - epsilon, im + epsilon)
            # Clamp to valid pixel range
            adv_x = torch.clamp(adv_x, minv, maxv)

        adv_x = adv_x.detach().requires_grad_()

    return adv_x