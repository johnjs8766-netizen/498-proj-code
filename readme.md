The watermarking pipeline consists of an encoder and a decoder.

Each device is assigned a unique secret key vector with dimensions of [1, 128]. This key functions as a plugin; to inject a device-specific watermark, the user must combine the encoder with their assigned secret key.

The decoder is maintained exclusively by the verifier. During the training phase, the verifier trains a single encoder-decoder pair. Once trained, the verifier distributes the encoder alongside a unique secret key vector to each user, meaning the user does not need access to the decoder.

The keys are stored in ./user_keys folder (assume there are 6 authorized devices), and the checkpoints of encoder and decoder are stored in ./checkpoint folder.

To train the model run:
python3 train.py