<div align="center">

# z-image-burn
<img src="./assets/slop.png" width="512px"/>

<sub>Generated using this project</sub>

---

<br/>
</div>

An implementation of [Z-Image](https://github.com/Tongyi-MAI/Z-Image) in rust using
[Burn](https://github.com/tracel-ai/burn).

## Usage

First download the model: https://huggingface.co/andreashgk/z-image-burn \
And the autoencoder: https://huggingface.co/Comfy-Org/z_image_turbo/blob/main/split_files/vae/ae.safetensors

Safetensors models that work for z-image in ComfyUI can also be loaded but will load slower.

As the text encoder is not yet implemented (it will come!!), you will need to run a python script to generate an embedding of a prompt first:
```py
python scripts/embed_prompt.py 'An edit of the popular meme referred to as place, place japan where on the top left is a drawing of a guy looking uninterested at the text "slop" to the right of him. Below him in the bottom left side of the image is a drawing of the same person but happy, looking at the text "slop, rust" to the right of him, in the bottom right of the image.'
```

The model can be run using the CLI:
```
cargo run -p z-image-cli -F <backend> -- --prompt-file prompt.safetensors
```
Be sure to fill in the backend with an actual backend: `tch`, `cuda`, `rocm`, `cpu`, `vulkan`.
The `tch` backend appears to be by far the fastest at the moment.

Use `--help` for more options.

## Acknowledgements

This implementation is based on the one found in the main Z-Image repo and ComfyUI.
The autoencoder implementation is based on the flux vae reference implementation.
