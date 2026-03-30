import os
import torch

from models import HybridWatermarker, PublicHybridWatermarker


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    num_devices = 6
    generator_ckpt = "checkpoints/stageC_generator_epoch_8.pth"

    os.makedirs("exported_public_model", exist_ok=True)
    os.makedirs("user_keys", exist_ok=True)

    # -----------------------------------
    # Load trained full generator
    # -----------------------------------
    full_generator = HybridWatermarker(
        n_fft=512,
        hop_length=160,
        win_length=400,
        num_devices=num_devices,
        alpha_stft=0.05,
        alpha_wave=0.02,
        code_time_steps=16
    ).to(device)

    full_generator.load_state_dict(torch.load(generator_ckpt, map_location=device))
    full_generator.eval()

    # -----------------------------------
    # Build public deployment generator
    # -----------------------------------
    public_generator = PublicHybridWatermarker(
        n_fft=512,
        hop_length=160,
        win_length=400,
        alpha_stft=0.05,
        alpha_wave=0.02,
        code_time_steps=16,
        cond_dim=32
    ).to(device)

    # copy shared weights
    public_generator.mask_net.load_state_dict(full_generator.mask_net.state_dict())
    public_generator.wave_branch.in_proj.load_state_dict(full_generator.wave_branch.in_proj.state_dict())
    public_generator.wave_branch.cond_proj.load_state_dict(full_generator.wave_branch.cond_proj.state_dict())
    public_generator.wave_branch.res_stack.load_state_dict(full_generator.wave_branch.res_stack.state_dict())
    public_generator.wave_branch.out.load_state_dict(full_generator.wave_branch.out.state_dict())

    # save public model
    torch.save(
        public_generator.state_dict(),
        "exported_public_model/public_generator.pth"
    )

    # -----------------------------------
    # Export one secret file per user
    # -----------------------------------
    with torch.no_grad():
        tf_codes = full_generator.codebook.codes.detach().cpu()                 # (U,F,Tc)
        wave_keys = full_generator.wave_branch.cond.embedding.weight.detach().cpu()  # (U,cond_dim)

        for u in range(num_devices):
            secret = {
                "user_id": u,
                "tf_code": tf_codes[u],      # (F,Tc)
                "wave_key": wave_keys[u],    # (cond_dim,)
            }
            torch.save(secret, f"user_keys/user_{u}_secret.pt")

    print("Export finished.")
    print("Saved public model to: exported_public_model/public_generator.pth")
    print("Saved user secrets to: user_keys/user_<id>_secret.pt")


if __name__ == "__main__":
    main()
