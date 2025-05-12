import torch
from torch import Tensor

from sentence_transformers import SentenceTransformer
from sentence_transformers.losses import MultipleNegativesRankingLoss

def tensor_transform(x: Tensor) -> Tensor:
    c=1
    x_min = x.min(dim=1, keepdim=True).values
    x_shifted = x - x_min
    
    eps = 1e-6
    x_max = x_shifted.max(dim=1, keepdim=True).values
    scale = torch.where(x_max == 0, eps * torch.ones_like(x_max), x_max)
    x_scaled = x_shifted / scale
    return x_scaled * c  # Scale to [0, c] without L1 normalization




def js_div(a: Tensor, b: Tensor) -> Tensor:
    """
    Compute Jensen-Shannon divergence between probability distributions.

    Args:
        a (Tensor): Batch of distributions shape [B, D]
        b (Tensor): Batch of distributions shape [C, D]

    Returns:
        Tensor: JS divergence scores shape [B, C]
    """

    # for scaling and normalization of
    a = tensor_transform(a)
    b = tensor_transform(b)
    print('TRANSFORMED A', a)
    print('TRANSFORMED B ', b)
    # Expand for broadcast and pairwise calculation between each query and doc
    a = a.unsqueeze(1)  # [B, 1, D]
    b = b.unsqueeze(0)  # [1, C, D]
         # epsilon and clamp to avoid zeros in log calculations
    eps = 1e-6
    a = torch.clamp(a, min=eps, max=1.0-eps)
    b = torch.clamp(b, min=eps, max=1.0-eps)
    print('A: ', a)
    print('B: ', b)
    m = 0.5 * (a + b) + eps
    print('M: ', m)
    # log2 to keep values within range [0, 1]
    kl_am = torch.sum(a * torch.log2(a / m), dim=-1)
    kl_bm = torch.sum(b * torch.log2(b / m), dim=-1)
    print('KL FOR AM AND KL FOR BM', kl_am, kl_bm)
    # Compute JS divergence and convert to similarity to match scale of standard similarities
    print('JS DIVERGENCE', 0.5 * (kl_am + kl_bm))
    js_similarity = 1.0 - 0.5 * (kl_am + kl_bm)
    print('JS Similarity',js_similarity)
    return js_similarity

def sanity_check_js_div(js_div):
    """Tests JS divergence implementation with controlled inputs."""
    # Test 1: Identical vectors (similarity = 1)
    queries = torch.tensor([[0.0, 5.0]], dtype=torch.float32).long()
    positives = queries.clone()

    # Test 2: Orthogonal vectors (similarity ~0)
    # CHECK THIS
    docs = torch.tensor([[5.0, 0.0]], dtype=torch.float32).long()

    # Test 3: Zero vectors (should use uniform fallback)
    zeros = torch.zeros_like(queries)

    # Test 4: Opposite vector
    opposite = torch.tensor([[0.0, -5.0]], dtype=torch.float32).long()

    # Forward pass checks
    for name, a, b in [("Identical", queries, positives),
                      ("Orthogonal", queries, docs),
                      ("Zeros", zeros, zeros),
                       ("Opposite", queries, opposite)]:
        print(f"\n=== Test: {name} ===")

        # Check tensor_transform
        a_transformed = tensor_transform(a)
        b_transformed = tensor_transform(b)
        print("a_transformed sum:", a_transformed.sum(dim=1))
        print("b_transformed sum:", b_transformed.sum(dim=1))

        # Check JS similarity
        scores = js_div(a, b)
        print("JS scores shape:", scores.shape)
        print("JS scores:", scores)

        # Check for NaNs/Infs
        if torch.isnan(scores).any() or torch.isinf(scores).any():
            print("NaNs/Infs detected in JS scores!")
        else:
            print("JS scores are finite")

    # Loss and gradient check
    dummy_model = SentenceTransformer('distilbert-base-uncased')
    loss_fn = MultipleNegativesRankingLoss(dummy_model)

    # Create fake batch
    features = [
        {"input_ids": torch.ones(2, 3).long(), "attention_mask": torch.ones(2, 3).long()},
        {"input_ids": torch.ones(2, 3).long(), "attention_mask": torch.ones(2, 3).long()}
    ]

    print("\n=== Training Loss Check ===")
    try:
        loss = loss_fn(features, None)
        loss.backward()
        print("Loss:", loss.item())
        print("Gradients:")
        for name, param in dummy_model.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.norm().item()
                print(f"{name}: {grad_norm}")
                if torch.isnan(param.grad).any():
                    print(f"NaN gradients in {name}!")
    except Exception as e:
        print(f"Crash during loss computation: {str(e)}")

# Initialize model and loss
sanity_check_js_div(js_div)
