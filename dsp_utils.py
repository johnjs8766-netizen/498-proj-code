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
def lsf_to_lpc_diff(lsf):
    """
    True Differentiable LSF to LPC conversion.
    Expands the polynomials P(z) and Q(z) from roots.
    """
    batch_size, order = lsf.shape
    device = lsf.device
    
    # 1. Convert LSF (radians) to LSP (cosines)
    qs = torch.cos(lsf)
    
    # 2. Separate into P (odd) and Q (even) lines
    # qs are sorted w1, w2, w3...
    q_odd = qs[:, 0::2]  # w1, w3, w5...
    q_even = qs[:, 1::2] # w2, w4, w6...
    
    # 3. Construct Polynomials P(z) and Q(z)
    # P(z) = (1 - z^-1) * Prod(1 - 2*q_odd*z^-1 + z^-2)
    # Q(z) = (1 + z^-1) * Prod(1 - 2*q_even*z^-1 + z^-2)
    
    # Helper to expand product of quadratic terms
    def expand_roots(roots):
        # roots shape: [Batch, N_sections]
        n_sect = roots.shape[1]
        
        # Start with [1.0]
        # We process effectively by expanding (1 - 2r z^-1 + z^-2)
        # Using a specialized loop for gradients
        
        # Initialize polynomial [Batch, 1] -> 1.0
        poly = torch.ones(batch_size, 1, device=device)
        
        for i in range(n_sect):
            r = roots[:, i:i+1] # current root cosine
            
            # Current polynomial coefficients
            # Multiply poly by (1, -2r, 1)
            # Equivalent to:
            # new[n] = old[n] - 2r*old[n-1] + old[n-2]
            
            # Pad poly for shifting
            p_0 = torch.cat([poly, torch.zeros(batch_size, 2, device=device)], dim=1)
            p_1 = torch.cat([torch.zeros(batch_size, 1, device=device), poly, torch.zeros(batch_size, 1, device=device)], dim=1)
            p_2 = torch.cat([torch.zeros(batch_size, 2, device=device), poly], dim=1)
            
            poly = p_0 - (2 * r * p_1) + p_2
            
        return poly

    p_poly = expand_roots(q_odd)
    q_poly = expand_roots(q_even)
    
    # 4. Apply boundary conditions
    # P(z) *= (1 - z^-1)
    p_final = torch.cat([p_poly, torch.zeros(batch_size, 1, device=device)], dim=1) - \
              torch.cat([torch.zeros(batch_size, 1, device=device), p_poly], dim=1)
              
    # Q(z) *= (1 + z^-1)
    q_final = torch.cat([q_poly, torch.zeros(batch_size, 1, device=device)], dim=1) + \
              torch.cat([torch.zeros(batch_size, 1, device=device), q_poly], dim=1)
              
    # 5. A(z) = 0.5 * (P(z) + Q(z))
    a_coeffs = 0.5 * (p_final + q_final)
    
    # Ignore the very last coefficient which is usually 0 due to order matching
    return a_coeffs[:, :-1] 

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