#!/usr/bin/env python3
from __future__ import annotations
import io
from pathlib import Path
from random import Random
import base64
import torch
from torch import Tensor
from ballfish.transformation import create_distribution
from ballfish.transformation import Datum, Quad
from ballfish import create_augmentation, Args, Datum, Quad
from torchvision.transforms.v2.functional import pil_to_tensor, to_pil_image


def dummy_datum(
    source: Tensor | None = None,
    quad: Quad | None = None,
    width: int = 16,
    height: int = 16,
    image: Tensor | None = None,
) -> Datum:
    if source is None:
        source = torch.zeros((1, 1, 1, 1))
    if quad is None:
        h, w = map(float, source.shape[-2:])
        quad = ((0.0, 0.0), (w, 0.0), (w, h), (0.0, h))
    return Datum(
        source=source, quad=quad, width=width, height=height, image=image
    )


def style(
    stroke: str = "#dd0000",
    fill: str = "none",
    stroke_width: float = 0.1,
    stroke_opacity: float | None = None,
):
    params = ["stroke-width:" + str(stroke_width), "stroke-miterlimit:4"]
    if stroke:
        params.append("stroke:" + stroke)
    if fill:
        params.append("fill:" + fill)
    if stroke_opacity:
        params.append("stroke-opacity:" + str(stroke_opacity))
    return ";".join(params)


class SVG:
    def __init__(self, width: float, height: float) -> None:
        self._width, self._height = width, height
        self._lines: list[str] = []
        self.set_shift()

    def _svg(self):
        yield (
            '<svg version="1.1" viewBox="0 0 {self._width} {self._height}" '
            'xmlns="http://www.w3.org/2000/svg" '
            'xmlns:xlink="http://www.w3.org/1999/xlink">'.format(self=self)
        )
        for line in self._lines:
            yield line
        yield "</svg>"

    def set_shift(self, x: float = 0.0, y: float = 0.0):
        self._x, self._y = x, y
        return self

    def add_quad(self, quad: Quad, style: str | None = None):
        quad = tuple((x + self._x, y + self._y) for x, y in quad)
        d = [f"M{quad[0][0]}", str(quad[0][1])]
        for p in quad[1:]:
            d.extend((f"L{p[0]}", str(p[1])))
        d.append("Z")
        self._lines.append(
            '<path d="{d}"{extra} />'.format(
                d=" ".join(d), extra=f' style="{style}"' if style else ""
            )
        )
        return self

    def add_line(self, x1: float, y1: float, x2: float, y2: float, style: str):
        self._lines.append(
            '<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" style="{style}" />'.format(
                x1=x1 + self._x,
                y1=y1 + self._y,
                x2=x2 + self._x,
                y2=y2 + self._y,
                style=style,
            )
        )
        return self

    def add_text(
        self,
        x: float,
        y: float,
        text: str,
        style: str = "font-size:0.26666683px;fill:#000000;fill-opacity:0.8",
    ):
        self._lines.append(
            '<text x="{x}" y="{y}" style="{style}">{text}</text>'.format(
                x=x + self._x, y=y + self._y, text=text, style=style
            )
        )

    def add_rect(
        self, x: float, y: float, width: float, height: float, style: str
    ):
        assert height, height
        assert width, width
        self._lines.append(
            f'<rect x="{x + self._x}" y="{y + self._y}" width="{width}" '
            f'height="{height}" style="{style}" />'
        )

    def add_image(self, x: float, y: float, image: Tensor, scale: int = 1):
        assert image.shape[0] == 1, image.shape[0]
        image = image.squeeze(0)
        image_u8 = (image.clip(0, 1) * 255 + 0.5).to(torch.uint8)
        image_pil = to_pil_image(image_u8)
        height, width = image_u8.shape[-2:]

        with io.BytesIO() as output:
            image_pil.save(
                output, format="PNG", optimize=True, compress_level=9
            )
            raw = output.getvalue()
        url = "data:image/png;base64," + base64.b64encode(raw).decode("ascii")
        self._lines.append(
            f'<image xlink:href="{url}" height="{height // scale}" '
            f'width="{width // scale}" x="{x + self._x}" y="{y + self._y}" '
            f'style="image-rendering:optimizeSpeed"/>'
        )

    def save(self, path: Path) -> None:
        print("gen", path)
        content = "\n".join(self._svg())
        path.write_text(content)


def gen_quad_transforms(path: Path, force: bool) -> None:
    augmentations = (
        (
            "projective_shift",
            create_augmentation(
                [
                    {
                        "name": "projective_shift",
                        "x": {"name": "truncnorm", "a": -3.1, "b": 3.1},
                    }
                ]
            ),
        ),
        (
            "rotate",
            create_augmentation(
                [
                    {
                        "name": "rotate",
                        "angle_deg": {"name": "uniform", "a": 0, "b": 360},
                    }
                ]
            ),
        ),
        (
            "paddings_addition",
            create_augmentation(
                [
                    {
                        "name": "paddings_addition",
                        "top": {"name": "uniform", "a": 0, "b": 0.25},
                        "right": {"name": "uniform", "a": 0, "b": 0.25},
                        "bottom": {"name": "uniform", "a": 0, "b": 0.25},
                        "left": {"name": "uniform", "a": 0, "b": 0.25},
                    }
                ]
            ),
        ),
        (
            "projective_paddings_addition",
            create_augmentation(
                [
                    {
                        "name": "projective_paddings_addition",
                        "top": {"name": "uniform", "a": 0, "b": 0.25},
                        "right": {"name": "uniform", "a": 0, "b": 0.25},
                        "bottom": {"name": "uniform", "a": 0, "b": 0.25},
                        "left": {"name": "uniform", "a": 0, "b": 0.25},
                    }
                ]
            ),
        ),
        (
            "projective1pt",
            create_augmentation(
                [
                    {
                        "name": "projective1pt",
                        "x": {"name": "truncnorm", "a": -0.25, "b": 0.25},
                        "y": {"name": "uniform", "a": -0.25, "b": 0.25},
                    }
                ]
            ),
        ),
        (
            "projective4pt",
            create_augmentation(
                [
                    {
                        "name": "projective4pt",
                        "x": {"name": "truncnorm", "a": -0.25, "b": 0.25},
                        "y": {"name": "uniform", "a": -0.25, "b": 0.25},
                    }
                ]
            ),
        ),
        (
            "scale",
            create_augmentation(
                [
                    {
                        "name": "scale",
                        "factor": {"name": "truncnorm", "a": 0.7, "b": 1.3},
                    }
                ]
            ),
        ),
    )

    quadrangles = (
        (
            (0.0, 0.0),
            (0.5, 0.0),
            (0.5, 1.0),
            (0.0, 1.0),
        ),
        (
            (0.1, 0.0),
            (0.5, 0.0),
            (0.5, 1.0),
            (0.0, 0.9),
        ),
    )

    random = Random(13)
    for name, augmentation in augmentations:
        out_path_svg = path / (name + ".svg")
        if out_path_svg.exists() and not force:
            continue
        svg = SVG(26, 6.0)
        n = 1 if name == "flip" else 10
        for quad_idx, quad in enumerate(quadrangles):
            for i in range(n):
                description = dummy_datum(quad=quad, width=16, height=16)
                new_description = augmentation(description, random)
                svg.set_shift(i * 2.5 + 1.0, quad_idx * 3.0 + 1.0)
                svg.add_quad(quad, style(stroke="#000000"))
                svg.add_quad(new_description.quads[0], style())
        svg.save(out_path_svg)


def gen_distributions(path: Path, force: bool) -> None:
    distributions = (
        (
            create_distribution({"name": "uniform", "a": -0.75, "b": 0.75}),
            "uniform_075_075",
        ),
        (
            create_distribution({"name": "uniform", "a": 0, "b": 0.5}),
            "uniform_000_050",
        ),
        (
            create_distribution(
                {"name": "truncnorm", "mu": 0, "sigma": 0.75, "delta": 1.0}
            ),
            "truncnorm_000_075_100",
        ),
        (
            create_distribution(
                {"name": "truncnorm", "mu": 0.4, "sigma": 0.3, "delta": 1.0}
            ),
            "truncnorm_040_030_100",
        ),
        (
            create_distribution(
                {
                    "name": "truncnorm",
                    "mu": 0,
                    "sigma": 0.5,
                    "a": 0.0,
                    "b": 1.0,
                }
            ),
            "truncnorm_000_050_000_100",
        ),
        (
            create_distribution({"name": "constant", "value": 0.25}),
            "constant_025",
        ),
        (
            create_distribution({"name": "randrange", "start": -1, "stop": 2}),
            "randrange_-1_2",
        ),
        (
            create_distribution({"name": "choice", "values": [-0.1, 0.1, 1]}),
            "choice",
        ),
        (
            create_distribution(
                {"name": "choice", "values": [(-0.1, 30), (0.1, 60), (1, 10)]}
            ),
            "choice_with_probability",
        ),
    )
    for distribution, name in distributions:
        out_path_svg = path / (name + ".svg")
        if out_path_svg.exists() and not force:
            continue
        random = Random(13)
        svg = SVG(3.0, 3.0)
        svg.set_shift(1.5, 2.5)
        samples = torch.tensor(
            [distribution(random) for i in range(100000)], dtype=torch.float64
        )
        hist, bin_edges = torch.histogram(samples, bins=60)

        width = area = 0.0
        for hist_idx, hist_val in enumerate(hist):
            left, right = bin_edges[hist_idx : hist_idx + 2]
            bin_length = right - left
            assert bin_length > 0.0, bin_length
            if hist_val:
                width += bin_length
            area += bin_length * hist_val

        if width < 0.2:  # for discrete distributions
            area = torch.sum(hist) / (torch.count_nonzero(hist))

        for hist_idx, hist_val in enumerate(hist):
            left, right = bin_edges[hist_idx : hist_idx + 2]
            height = hist_val / area
            if height:
                svg.add_rect(
                    width=right - left,
                    height=height,
                    x=left,
                    y=-height,
                    style="fill:#dd0000;stroke:none;",
                )

        ls = style(stroke="#000000", stroke_width=0.025, stroke_opacity=0.8)
        svg.add_line(-1.5, 0, 1.5, 0, ls)
        svg.add_line(-1.0, 0.05, -1.0, -0.05, ls)
        svg.add_line(1.0, 0.05, 1.0, -0.05, ls)
        svg.add_line(0, -1.5, 0, 0.05, ls)
        svg.add_line(-0.05, -1.0, 0.05, -1.0, ls)
        svg.add_text(-0.07, 0.25, "0")
        svg.add_text(1.0 - 0.07, 0.25, "1")
        svg.add_text(-1 - 0.09, 0.25, "-1")
        svg.add_text(0.05, -1, "1")
        svg.save(out_path_svg)


def gen_image_transforms(path: Path, force: bool) -> None:
    augmentations = (
        (
            "pow",
            create_augmentation(
                [
                    {
                        "name": "pow",
                        "pow": {"name": "truncnorm", "a": 0.6, "b": 3.0},
                    }
                ]
            ),
        ),
        (
            "log_e",
            create_augmentation(
                [
                    {
                        "name": "log",
                        "base": "e",
                    }
                ]
            ),
        ),
        (
            "multiply",
            create_augmentation(
                [
                    {
                        "name": "multiply",
                        "factor": {"name": "truncnorm", "a": 1 / 3, "b": 3.0},
                    }
                ]
            ),
        ),
        (
            "add",
            create_augmentation(
                [
                    {
                        "name": "add",
                        "value": {
                            "name": "truncnorm",
                            "a": -1 / 3,
                            "b": 1 / 3,
                        },
                    }
                ]
            ),
        ),
        (
            "noise_homoscedastic",
            create_augmentation(
                [
                    {
                        "name": "noise",
                        "std": {"name": "truncnorm", "a": 0, "b": 1 / 10},
                    }
                ]
            ),
        ),
        (
            "noise_heteroscedastic",
            create_augmentation(
                [
                    {
                        "name": "noise",
                        "std": {"name": "truncnorm", "a": 0, "b": 1 / 10},
                        "type": "heteroscedastic",
                    }
                ]
            ),
        ),
        (
            "sharpness",
            create_augmentation(
                [
                    {
                        "name": "sharpness",
                        "factor": {"name": "truncnorm", "a": 1 / 2, "b": 12},
                    }
                ]
            ),
        ),
        (
            "flip_horizontal",
            create_augmentation(
                [
                    {
                        "name": "flip",
                        "direction": "horizontal",
                    },
                    {"name": "rasterize"},
                ]
            ),
        ),
        (
            "flip_primary_diagonal",
            create_augmentation(
                [
                    {
                        "name": "flip",
                        "direction": "primary_diagonal",
                    },
                    {"name": "rasterize"},
                ]
            ),
        ),
        (
            "flip_secondary_diagonal",
            create_augmentation(
                [
                    {
                        "name": "flip",
                        "direction": "secondary_diagonal",
                    },
                    {"name": "rasterize"},
                ]
            ),
        ),
        (
            "flip_vertical",
            create_augmentation(
                [
                    {
                        "name": "flip",
                        "direction": "vertical",
                    },
                    {"name": "rasterize"},
                ]
            ),
        ),
        (
            "shading",
            create_augmentation(
                [
                    {
                        "name": "shading",
                        "value": {"name": "truncnorm", "a": -0.5, "b": 0.5},
                    },
                ]
            ),
        ),
        (
            "grayscale",
            create_augmentation(
                [
                    {"name": "grayscale"},
                ]
            ),
        ),
        (
            "resize",
            create_augmentation(
                [
                    {"name": "resize", "width": 100, "height": 60},
                ]
            ),
        ),
    )
    from PIL import Image
    from torchvision.transforms import Resize

    image_path = Path(__file__).parent / "example.png"
    img_rgb = Resize((64, 64))(
        pil_to_tensor(Image.open(image_path).convert("RGB"))
    ).unsqueeze(0) / torch.full((1,), 255.0)
    assert img_rgb.ndim == 4
    img_gray = torch.mean(img_rgb, dim=1).unsqueeze(0)

    for name, augmentation in augmentations:
        out_path_svg = path / (name + ".svg")
        if out_path_svg.exists() and not force:
            continue
        random = Random(13)
        svg = SVG(210, 16 * 2 + 4 + 2)
        n = 1 if name.startswith(("log_", "flip_", "grayscale", "re")) else 10
        for img_idx, img in enumerate((img_gray, img_rgb)):
            svg.set_shift(1.0, 1.0 + img_idx * 20)
            svg.add_image(0, 0, img, scale=4)
            svg.add_text(
                19,
                11,
                "⇒",
                style="font-size:8px;fill:#dd0000;fill-opacity:1.0",
            )
            for i in range(n):
                if name.startswith("flip_"):
                    datum = dummy_datum(
                        source=img.clone(), width=64, height=64
                    )
                else:
                    if name == "sharpness":
                        from torchvision.transforms.v2 import GaussianBlur

                        img_to_use = GaussianBlur(3)(img)
                    else:
                        img_to_use = img.clone()
                    datum = dummy_datum(image=img_to_use)
                try:
                    out = augmentation(datum, random)
                except Exception as e:
                    svg.add_text(
                        29,
                        11,
                        e.__class__.__name__,
                        style="font-size:8px;fill:#000000;fill-opacity:1.0",
                    )
                    print(f"<!-- {e}, {type(e)} -->")
                else:
                    svg.add_image(29 + 18 * i, 0, out.image, scale=4)
        svg.save(out_path_svg)


def generate_images(force: bool = False):
    out_path = Path(__file__).parent / "_static" / "transformations"
    out_path.mkdir(parents=True, exist_ok=True)
    gen_quad_transforms(out_path, force)
    gen_distributions(out_path, force)
    gen_image_transforms(out_path, force)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser("generate svgs for documentation")
    parser.add_argument("--force", action="store_true")
    generate_images(**vars(parser.parse_args()))
