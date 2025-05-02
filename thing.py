import timeit
import torch
import torch.nn.functional as F

def pad_scipy_reflect_vectorized(input_tensor: torch.Tensor, padding: tuple[int, int, int, int]) -> torch.Tensor:
    """
    Manually implements padding equivalent to SciPy's 'reflect' mode using
    vectorized PyTorch operations (slicing, flip, cat).

    SciPy 'reflect': Extends by reflecting about the center of the edge pixels.

    Args:
        input_tensor (torch.Tensor): Input tensor (N, C, H, W).
        padding (tuple[int, int, int, int]): Padding tuple (pad_left, pad_right, pad_top, pad_bottom).

    Returns:
        torch.Tensor: Padded tensor.
    """
    N, C, H, W = input_tensor.shape
    pad_left, pad_right, pad_top, pad_bottom = padding

    # --- Vertical padding ---
    if pad_top > 0:
        top_slice = input_tensor[..., 0:pad_top, :]
        top_pad = top_slice.flip(dims=[-2])
    else: top_pad = torch.empty((N, C, 0, W), dtype=input_tensor.dtype, device=input_tensor.device) # Handle 0 padding

    if pad_bottom > 0:
        bottom_slice = input_tensor[..., H-pad_bottom:H, :]
        bottom_pad = bottom_slice.flip(dims=[-2])
    else: bottom_pad = torch.empty((N, C, 0, W), dtype=input_tensor.dtype, device=input_tensor.device)

    vert_padded = torch.cat([top_pad, input_tensor, bottom_pad], dim=-2) # dim=-2 is Height

    # --- Horizontal padding (using the vertically padded tensor) ---
    # Recalculate height for slicing after vertical padding
    H_padded = vert_padded.shape[-2]

    if pad_left > 0:
        left_slice = vert_padded[..., :, 0:pad_left]
        left_pad = left_slice.flip(dims=[-1])
    else: left_pad = torch.empty((N, C, H_padded, 0), dtype=input_tensor.dtype, device=input_tensor.device)

    if pad_right > 0:
        right_slice = vert_padded[..., :, W-pad_right:W]
        right_pad = right_slice.flip(dims=[-1])
    else: right_pad = torch.empty((N, C, H_padded, 0), dtype=input_tensor.dtype, device=input_tensor.device)

    output_padded = torch.cat([left_pad, vert_padded, right_pad], dim=-1) # dim=-1 is Width

    return output_padded

def pad_scipy_reflect_optimized(input_tensor: torch.Tensor, padding: tuple[int, int, int, int]) -> torch.Tensor:
    """
    Optimized implementation of padding equivalent to SciPy's 'reflect' mode.
    Allocates the final tensor once and fills regions by slicing/flipping the input.

    SciPy 'reflect': Extends by reflecting about the edge pixel boundary.
                     e.g., [1,2,3,4] pad=2 -> [2,1, | 1,2,3,4 | 4,3]

    Args:
        input_tensor (torch.Tensor): Input tensor (N, C, H, W).
        padding (tuple[int, int, int, int]): Padding tuple (pad_left, pad_right, pad_top, pad_bottom).

    Returns:
        torch.Tensor: Padded tensor.
    """
    N, C, H, W = input_tensor.shape
    pad_left, pad_right, pad_top, pad_bottom = padding

    # Calculate output dimensions
    out_H = H + pad_top + pad_bottom
    out_W = W + pad_left + pad_right

    # Allocate output tensor
    output = torch.empty((N, C, out_H, out_W), dtype=input_tensor.dtype, device=input_tensor.device)

    # 1. Fill the center region with the original tensor
    output[..., pad_top:pad_top + H, pad_left:pad_left + W] = input_tensor

    # --- Fill Edges (excluding corners) ---

    # 2. Top edge
    if pad_top > 0:
        # Slice: First pad_top rows
        # Flip vertically
        top_slice = input_tensor[..., 0:pad_top, :]
        output[..., 0:pad_top, pad_left:pad_left + W] = top_slice.flip(dims=[-2])

    # 3. Bottom edge
    if pad_bottom > 0:
        # Slice: Last pad_bottom rows
        # Flip vertically
        bottom_slice = input_tensor[..., H - pad_bottom:H, :]
        output[..., pad_top + H:pad_top + H + pad_bottom, pad_left:pad_left + W] = bottom_slice.flip(dims=[-2])

    # 4. Left edge
    if pad_left > 0:
        # Slice: First pad_left columns
        # Flip horizontally
        left_slice = input_tensor[..., :, 0:pad_left]
        output[..., pad_top:pad_top + H, 0:pad_left] = left_slice.flip(dims=[-1])

    # 5. Right edge
    if pad_right > 0:
        # Slice: Last pad_right columns
        # Flip horizontally
        right_slice = input_tensor[..., :, W - pad_right:W]
        output[..., pad_top:pad_top + H, pad_left + W:pad_left + W + pad_right] = right_slice.flip(dims=[-1])


    # --- Fill Corners (require double reflection) ---

    # 6. Top-Left corner
    if pad_top > 0 and pad_left > 0:
        # Slice: Top-left corner of input
        # Flip both vertically and horizontally
        corner_slice = input_tensor[..., 0:pad_top, 0:pad_left]
        output[..., 0:pad_top, 0:pad_left] = corner_slice.flip(dims=[-2, -1])

    # 7. Top-Right corner
    if pad_top > 0 and pad_right > 0:
        # Slice: Top-right corner of input
        # Flip both vertically and horizontally
        corner_slice = input_tensor[..., 0:pad_top, W - pad_right:W]
        output[..., 0:pad_top, pad_left + W:pad_left + W + pad_right] = corner_slice.flip(dims=[-2, -1])

    # 8. Bottom-Left corner
    if pad_bottom > 0 and pad_left > 0:
        # Slice: Bottom-left corner of input
        # Flip both vertically and horizontally
        corner_slice = input_tensor[..., H - pad_bottom:H, 0:pad_left]
        output[..., pad_top + H:pad_top + H + pad_bottom, 0:pad_left] = corner_slice.flip(dims=[-2, -1])

    # 9. Bottom-Right corner
    if pad_bottom > 0 and pad_right > 0:
        # Slice: Bottom-right corner of input
        # Flip both vertically and horizontally
        corner_slice = input_tensor[..., H - pad_bottom:H, W - pad_right:W]
        output[..., pad_top + H:pad_top + H + pad_bottom, pad_left + W:pad_left + W + pad_right] = corner_slice.flip(dims=[-2, -1])

    return output
def pad_scipy_reflect_optimized_nonzero(input_tensor: torch.Tensor, padding: tuple[int, int, int, int]) -> torch.Tensor:
    """
    Optimized implementation of padding equivalent to SciPy's 'reflect' mode.
    ASSUMES ALL PADDING VALUES (pad_left, pad_right, pad_top, pad_bottom) ARE > 0.
    Allocates the final tensor once and fills regions by slicing/flipping the input.

    Args:
        input_tensor (torch.Tensor): Input tensor (N, C, H, W).
        padding (tuple[int, int, int, int]): Padding tuple (pad_left, pad_right, pad_top, pad_bottom).
                                            ALL VALUES MUST BE > 0.

    Returns:
        torch.Tensor: Padded tensor.

    Raises:
        IndexError or other tensor errors if any padding value is 0.
    """
    N, C, H, W = input_tensor.shape
    pad_left, pad_right, pad_top, pad_bottom = padding

    # Optional: Add an assertion if you want to enforce the non-zero constraint explicitly at runtime
    # assert pad_left > 0 and pad_right > 0 and pad_top > 0 and pad_bottom > 0, \
    #        "All padding values must be > 0 for this specialized function"

    # Calculate output dimensions
    out_H = H + pad_top + pad_bottom
    out_W = W + pad_left + pad_right

    # Allocate output tensor
    output = torch.empty((N, C, out_H, out_W), dtype=input_tensor.dtype, device=input_tensor.device)

    # 1. Fill the center region
    output[..., pad_top:pad_top + H, pad_left:pad_left + W] = input_tensor

    # --- Fill Edges (No checks needed) ---
    # 2. Top edge
    top_slice = input_tensor[..., 0:pad_top, :]
    output[..., 0:pad_top, pad_left:pad_left + W] = top_slice.flip(dims=[-2])

    # 3. Bottom edge
    bottom_slice = input_tensor[..., H - pad_bottom:H, :]
    output[..., pad_top + H:pad_top + H + pad_bottom, pad_left:pad_left + W] = bottom_slice.flip(dims=[-2])

    # 4. Left edge
    left_slice = input_tensor[..., :, 0:pad_left]
    output[..., pad_top:pad_top + H, 0:pad_left] = left_slice.flip(dims=[-1])

    # 5. Right edge
    right_slice = input_tensor[..., :, W - pad_right:W]
    output[..., pad_top:pad_top + H, pad_left + W:pad_left + W + pad_right] = right_slice.flip(dims=[-1])

    # --- Fill Corners (No checks needed) ---
    # 6. Top-Left corner
    corner_tl = input_tensor[..., 0:pad_top, 0:pad_left]
    output[..., 0:pad_top, 0:pad_left] = corner_tl.flip(dims=[-2, -1])

    # 7. Top-Right corner
    corner_tr = input_tensor[..., 0:pad_top, W - pad_right:W]
    output[..., 0:pad_top, pad_left + W:pad_left + W + pad_right] = corner_tr.flip(dims=[-2, -1])

    # 8. Bottom-Left corner
    corner_bl = input_tensor[..., H - pad_bottom:H, 0:pad_left]
    output[..., pad_top + H:pad_top + H + pad_bottom, 0:pad_left] = corner_bl.flip(dims=[-2, -1])

    # 9. Bottom-Right corner
    corner_br = input_tensor[..., H - pad_bottom:H, W - pad_right:W]
    output[..., pad_top + H:pad_top + H + pad_bottom, pad_left + W:pad_left + W + pad_right] = corner_br.flip(dims=[-2, -1])

    return output


# --- Performance Comparison ---
print("\n--- Benchmarking (Non-Zero Padding Assumption) ---")
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")
# Use a larger tensor for meaningful benchmark
input_large = torch.randn(16, 3, 256, 256, device=device)
# CRITICAL: Use padding > 0 for the non-zero function test
padding_large_nonzero = (3, 3, 3, 3)
# Also test with mixed padding for the original function
padding_large_mixed = (0, 3, 0, 3)


n_runs = 200 # Increase runs slightly

# Ensure correctness first
output_ref = pad_scipy_reflect_optimized(input_large, padding_large_nonzero)
output_nz = pad_scipy_reflect_optimized_nonzero(input_large, padding_large_nonzero)
print(f"Outputs match for non-zero padding: {torch.equal(output_ref, output_nz)}")

t_original_nz_case = timeit.timeit(lambda: pad_scipy_reflect_optimized(input_large, padding_large_nonzero), number=n_runs)
t_optimized_nz = timeit.timeit(lambda: pad_scipy_reflect_optimized_nonzero(input_large, padding_large_nonzero), number=n_runs)

# Also time original with mixed padding for context
t_original_mixed_case = timeit.timeit(lambda: pad_scipy_reflect_optimized(input_large, padding_large_mixed), number=n_runs)


print(f"\nOriginal function time (padding={padding_large_nonzero}):   {t_original_nz_case / n_runs:.8f} seconds per run")
print(f"Non-Zero function time (padding={padding_large_nonzero}): {t_optimized_nz / n_runs:.8f} seconds per run")
if t_original_nz_case > 0:
  speedup = t_original_nz_case / t_optimized_nz
  print(f"Speedup: {speedup:.3f}x")
  if speedup < 1.05: # If speedup is less than 5%
      print("--> Speedup is likely minimal/within noise margin.")

print(f"\nOriginal function time (padding={padding_large_mixed}): {t_original_mixed_case / n_runs:.8f} seconds per run (for comparison)")


# Example of failure with zero padding
try:
    padding_with_zero = (0, 1, 1, 1)
    print("\nTesting non-zero function with zero padding (expecting error)...")
    pad_scipy_reflect_optimized_nonzero(input_large, padding_with_zero)
except Exception as e:
    print(f"--> Caught expected error: {type(e).__name__}: {e}")