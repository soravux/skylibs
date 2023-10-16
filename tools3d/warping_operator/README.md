# Warping operator

The warping operator of [Gardner et al., 2017](https://dl.acm.org/doi/10.1145/3130800.3130891) is implemented in the function `tools3d.warping_operator.warpEnvironmentMap`. This allows to simulate a translation through an HDR environment map with unknown geometry by approximating it with a sphere. It can be used to relight 3D models from a different position than the camera position in the original panorama.

## Documentation

The function `warpEnvironmentMap` is documented in the source code [here](__init__.py).

## Example usage

The script `example_warp_operator.py` shows a fun example usage, where the camera moves from the far back to the far front of the environment map. First, have a panorama image ready, e.g. `pano.exr`. Then, run the script with:

```bash
python tools3d/warping_operator/example_warp_operator.py --environment 'pano.exr'
```
