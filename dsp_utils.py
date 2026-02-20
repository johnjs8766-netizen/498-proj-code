import torch
import torch.nn.functional as F
import numpy as np

# --- 1. LPC Analysis (Pure PyTorch, Differentiable) ---
def lpc_torch(waveform, order):
    """
    Computes LPC coefficients using Levinson-Durbin recursion.
    Robust to silence and float32 instability.
    """
    b, t = waveform.shape
    device = waveform.device
    
    # Pre-emphasis (optional but recommended for stability)
    waveform = torch.cat([waveform[:, 0:1], waveform[:, 1:] - 0.97 * waveform[:, :-1]], dim=1)
    
    # 1. Autocorrelation via FFT
    n_fft = 2 ** (t * 2 - 1).bit_length()
    X = torch.fft.rfft(waveform, n=n_fft, dim=-1)
    R = torch.fft.irfft(torch.abs(X) ** 2, n=n_fft, dim=-1)
    
    r = R[:, :order + 1]
    
    # 2. Levinson-Durbin
    # Add epsilon to diagonal (energy) to prevent div/0 on silence
    r[:, 0] = r[:, 0] * 1.0001 + 1e-6
    
    E = r[:, 0]
    a = torch.zeros(b, order + 1, device=device)
    a[:, 0] = 1.0
    
    a_prev = a.clone()
    
    for k in range(1, order + 1):
        # Calculate reflection coef
        acc = (a_prev[:, :k] * torch.flip(r[:, 1:k+1], [1])).sum(dim=1)
        k_coeff = -acc / (E + 1e-8)
        
        # Clamp reflection coefs to ensure stability (|k| < 1)
        k_coeff = torch.clamp(k_coeff, -0.999, 0.999)
        
        # Update weights
        a[:, k] = k_coeff
        a[:, 1:k] = a_prev[:, 1:k] + k_coeff.unsqueeze(1) * torch.flip(a_prev[:, 1:k], [1])
        
        # Update Error
        E = E * (1 - k_coeff**2)
        a_prev = a.clone()

    return a

# --- 2. LSF Utils (Analysis only) ---
def lpc_to_lsf(lpc_coeffs):
    """
    Converts LPC to LSF. Non-differentiable (uses numpy roots).
    """
    batch_size = lpc_coeffs.shape[0]
    device = lpc_coeffs.device
    lsfs = []
    lpc_np = lpc_coeffs.detach().cpu().numpy()
    
    for i in range(batch_size):
        # Add small noise to avoid singular roots on perfect silence
        poly = lpc_np[i] + 1e-9 * np.random.randn(len(lpc_np[i]))
        roots = np.roots(poly)
        angles = np.angle(roots)
        # Filter for positive frequencies 0 < w < pi
        angles = np.sort(angles[angles > 0])
        
        # Pad/Trim to ensure fixed size
        target = lpc_coeffs.shape[1] - 1
        if len(angles) < target:
             angles = np.pad(angles, (0, target - len(angles)), 'edge')
        elif len(angles) > target:
             angles = angles[:target]
        lsfs.append(angles)
        
    return torch.tensor(np.array(lsfs), dtype=torch.float32, device=device)

# --- 3. Differentiable Synthesis (LSF -> LPC) ---
def lsf_to_lpc_diff(lsf: torch.Tensor) -> torch.Tensor:
    """
    Stable double-precision LSF -> LPC conversion.

    Args:
        lsf: [B, p] (p must be even)

    Returns:
        a_coeffs: [B, p+1]
    """

    B, p = lsf.shape
    assert p % 2 == 0, "LPC order must be even"

    device = lsf.device
    lsf = lsf.double()

    # Convert to cosines
    cos_lsf = torch.cos(lsf)

    # Split
    cos_odd = cos_lsf[:, 0::2]
    cos_even = cos_lsf[:, 1::2]

    def build_poly(cos_vals):
        """
        Build ∏ (1 - 2c z^-1 + z^-2)
        Returns [B, p/2*2 + 1]
        """
        B, n = cos_vals.shape
        poly = torch.ones(B, 1, dtype=torch.float64, device=device)

        for i in range(n):
            c = cos_vals[:, i:i+1]  # [B,1]

            # Quadratic section
            section = torch.cat([
                torch.ones_like(c),
                -2*c,
                torch.ones_like(c)
            ], dim=1)  # [B,3]

            # Standard convolution (manual, stable)
            L = poly.shape[1]
            new_poly = torch.zeros(B, L+2, dtype=torch.float64, device=device)

            for k in range(3):
                new_poly[:, k:k+L] += section[:, k:k+1] * poly

            poly = new_poly

        return poly

    P = build_poly(cos_odd)
    Q = build_poly(cos_even)

    # Multiply P by (1 - z^-1)
    P = F.pad(P, (0,1)) - F.pad(P, (1,0))

    # Multiply Q by (1 + z^-1)
    Q = F.pad(Q, (0,1)) + F.pad(Q, (1,0))

    # Combine
    A = 0.5 * (P + Q)

    # A should now be length p+1
    A = A[:, :p+1]

    # Force a0 = 1
    A[:, 0] = 1.0

    return A.float()

# --- 4. MicPro Transform ---
def micpro_transform(lsf, params):
    # Same as before, but with stricter clamping
    w = lsf / np.pi
    xi1, xi2, xi3 = params[:, 0:1], params[:, 1:2], params[:, 2:3]
    
    w_t1 = w * xi1 
    w_t2 = w_t1 + (xi2 - 1.0) * (torch.sin(2 * np.pi * w_t1) / lsf.shape[1])
    
    w_next = torch.roll(w_t2, -1, dims=1)
    w_next[:, -1] = 1.0 
    delta = w_next - w_t2
    # Ensure delta doesn't flip negative
    delta = torch.clamp(delta, min=1e-4)
    
    delta_new = delta + (xi3 - 1.0) * (0.1 - delta)
    w_t3 = torch.cumsum(delta_new, dim=1)
    
    # Strict clamping to avoid unstable filters
    w_out = torch.clamp(w_t3, 0.02, 0.98) 
    w_out, _ = torch.sort(w_out, dim=1)
    
    return w_out * np.pi