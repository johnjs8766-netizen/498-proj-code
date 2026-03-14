# ------------------------------
# Note:
# you need to run this script in a linux environment
# litert_torch does not support windows or Mac yet
# ------------------------------
import torch
from models import PluginSEANetGenerator
from torch.utils.data import DataLoader
from datasets import WatermarkDataset
import litert_torch
from litert_torch.quantize.quant_config import QuantConfig
from litert_torch.generative.quantize import quant_attrs, quant_recipe

# ------------------------------
# Settings
# ------------------------------
device = "cpu"
key_dim = 128
num_devices = 6
alpha = 2.0
audio_len = 16000 * 3
batch_size = 1

pt_checkpoint = "checkpoints/generator_epoch_16.pth"
tflite_path = "embedded_models/generator_int8.tflite"

# #https://wiki.seeedstudio.com/XIAO-BLE-Sense-TFLite-Getting-Started/
# #https://openelab.io/blogs/learn/tensorflow-lite-on-esp32?srsltid=AfmBOopGvm-_gLD4mVV7rxlUGS6S7ubEvEyGfU7mNeAHYISa1AsoQi31

# ------------------------------
# Load PyTorch generator
# ------------------------------
generator = PluginSEANetGenerator(key_dim=key_dim).to(device)
generator.load_state_dict(torch.load(pt_checkpoint, map_location=torch.device('cpu')))
generator.eval()

# Load key vault
key_vault = torch.zeros(num_devices, key_dim).to(device)
for u in range(num_devices):
    key_vault[u] = torch.load(f"user_keys/user_{u}_plugin_key.pt", map_location=torch.device('cpu')).to(device)

# -----------------------------
# PyTorch → TFLite
# -----------------------------

print("Converting PyTorch -> TFLite")

test_dataset = WatermarkDataset("dataset_libritts/dataset/metadata.csv", target_sr=16000, duration=3.0, mode='test')
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Example input
dummy_audio = torch.randn(1, 1, 48000)      # 3 sec @ 16kHz
dummy_key = torch.randn(1, 128)
print(dummy_audio.dtype)
print(dummy_key.dtype)
def representative_dataset_gen():
    for raw_audio, _ in test_loader:
        batch_size = raw_audio.shape[0]
        dummy_keys = torch.randn(batch_size, 128)  # same shape as keys
        yield (raw_audio, dummy_keys)

print(list(quant_attrs.Mode))
print(list(quant_attrs.Dtype))
print(list(quant_attrs.Algorithm))
print(list(quant_attrs.Granularity))
print(quant_recipe.supported_schemes.get_supported_layer_schemes())

layer_recipe = quant_recipe.LayerQuantRecipe(
    activation_dtype=quant_recipe.quant_attrs.Dtype.FP32,       # must be FP32
    weight_dtype=quant_recipe.quant_attrs.Dtype.INT8,           # weights quantized
    mode=quant_recipe.quant_attrs.Mode.WEIGHT_ONLY,          # supported mode
    algorithm=quant_recipe.quant_attrs.Algorithm.MIN_MAX,       # supported algorithm
    granularity=quant_recipe.quant_attrs.Granularity.CHANNELWISE
)
layer_recipe.verify()  # sanity check

gen_recipe = quant_recipe.GenerativeQuantRecipe(
    default=layer_recipe
)
gen_recipe.verify()

# 2️⃣ Wrap it in QuantConfig
quant_cfg = QuantConfig(generative_recipe=gen_recipe)

# 3️⃣ Convert the model
edge_model = litert_torch.convert(
    module=generator,
    sample_args=(dummy_audio, dummy_key),  # your example tensors
    quant_config=quant_cfg,
)

# Export the model
edge_model.export(tflite_path)

print("TFLite model saved")

