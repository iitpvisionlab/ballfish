## Ballfish

Image augmentation library


### Installation
`pip install ballfish`

### Documentation

https://ballfish.readthedocs.io/

### Example


```python
from ballfish import create_augmentation, Datum
from random import Random

augmentation = create_augmentation(
    [
        # First, augment the quadrangle (quad argument)
        # with geometric transformations
        {
            "name": "rotate",
            "angle_deg": {"name": "uniform", "a": 0, "b": 360},
        },
        # Second, apply the projective transformation to source argument
        # After this step, no more quad transformations make sense
        {"name": "rasterize"},
        # Third, apply raster transformations
        {
            "name": "noise",
            "std": {"name": "truncnorm", "a": 0, "b": 1 / 10},
        },
    ]
)

random = Random(13)
# The source argument must be a `torch.tensor` in (N, C, H, W) format

# The quad argument should be provided in a clockwise order,
# starting from the top-left corner, in (x, y) format.
result = augmentation(Datum(
    source=source, quad=quad, width=64, height=64
), random)

# Transformed image and quads
print(result.quads, result.image)
```
