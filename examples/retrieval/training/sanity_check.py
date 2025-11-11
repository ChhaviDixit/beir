# Sanity check for tensor_transform
import torch

from torch import Tensor

def tensor_transform(x: Tensor) -> Tensor:
    
    # linearly tranforms the input vector to range [0, c], followed by L1 normalization for "probability distribution"

    # c is upper limit of [0, c], can be treated as a hyperparameter
    c=1
    x_min = x.min(dim=1, keepdim=True).values
    x_shifted = x - x_min

    x_max = x_shifted.max(dim=1, keepdim = True).values
    eps = 1e-6
    # to avoid division by 0
    scale = torch.where(x_max==0, eps * torch.ones_like(x_max), x_max)
    # x_scaled = x_shifted/x_max
    x_scaled = x_shifted/scale
    x_scaled_to_c = x_scaled*c

    x_sum = x_scaled_to_c.sum(dim=1, keepdim=True)
    # uniform distribution if sum is 0
    x_sum_normalized = torch.where(x_sum == 0, torch.ones_like(x_scaled) / x_scaled_to_c.size(1), x_scaled_to_c/x_sum)
    return x_sum_normalized



def js_div(a: Tensor, b: Tensor) -> Tensor:
    a = tensor_transform(torch.as_tensor(a))
    b = tensor_transform(torch.as_tensor(b))
    
    a = a.unsqueeze(1)
    eps=1e-5
    a = torch.clamp(a, min=eps, max=1.0-eps)
    
    batch_size_a = a.size(0)
    batch_size_b = b.size(0)
    
    # Chunk size - adjusted based on available memory
    chunk_size = 256  
    
    js_similarity = torch.zeros((batch_size_a, batch_size_b), device=a.device)
    
    for i in range(0, batch_size_b, chunk_size):
        b_chunk = b[i:i + chunk_size]
        b_chunk = b_chunk.unsqueeze(0)
        b_chunk = torch.clamp(b_chunk, min=eps, max=1.0-eps)
        
        m = 0.5 * (a + b_chunk) + eps
        
        kl_am = torch.sum(a * torch.log2(a / m), dim=-1)
        kl_bm = torch.sum(b_chunk * torch.log2(b_chunk / m), dim=-1)
        
        js_similarity[:, i:i + chunk_size] = 1.0 - 0.5 * (kl_am + kl_bm)
        
    return js_similarity

# Create a sample tensor
x = torch.tensor([[1.0, 2.0, 3.0], [4.0, 0.0, 5.0], [0.0, 0.0, 0.0]])
transformed_x = tensor_transform(x)
print("Original tensor:")
print(x)
print("\nTransformed tensor:")
print(transformed_x)

# Check if rows sum to 1 (approximately due to floating point precision)
print("\nSum of each row in transformed tensor:")
print(transformed_x.sum(dim=1))


# Sanity check for js_div
# Create two sample tensors
a = torch.tensor([[1.0, 2.0, 3.0], [4.0, 0.0, 5.0]])
b = torch.tensor([[1.0, 2.0, 3.0], [6.0, 7.0, 8.0]])

js_similarity_matrix = js_div(a, b)
print("\nJS Similarity matrix:")
print(js_similarity_matrix)

# Check expected values (e.g., similarity of a row with itself should be close to 1)
print("\nSimilarity of first row of 'a' with first row of 'b' (should be close to 1):")
print(js_similarity_matrix[0, 0])