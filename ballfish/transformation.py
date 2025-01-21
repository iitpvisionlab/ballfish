from __future__ import annotations
from typing import (
    Callable,
    TypeAlias,
    TypedDict,
    Literal,
    Type,
    TYPE_CHECKING,
)
from math import radians, hypot, sin, cos
from random import Random
import operator
import torch
from torch import Tensor
from .distribution import create_distribution, DistributionParams, Distribution
from .projective import Quad, projection_transform_point, calc_projection
from .projective_transform import projective_transform, Mode, PaddingMode
from collections.abc import Sequence

if TYPE_CHECKING:
    from typing import NotRequired

PerEnum: TypeAlias = Literal["channel", "batch", "tensor"]

all_transformation_classes: dict[str, Type[Transformation]] = {}


class Datum:
    """
    Input and output class for augmentation
    """

    quads: list[Quad]

    def __init__(
        self,
        source: Tensor | None = None,
        quad: Quad | None = None,
        width: int | None = None,
        height: int | None = None,
        image: Tensor | None = None,
    ) -> None:
        assert source is None or source.ndim == 4, source.ndim
        #: Main input that :class:`Rasterize` takes as input.
        #: Expected to be in (N, C, H, W) format.  This input unfortunately
        #: must be in `torch.float{32,64}` formats because torch's
        #: `torch.nn.functional.grid_sample` doesn't work with other types.
        self.source = source
        if source is not None:
            if width is None or height is None:
                assert width is None and height is None
                height, width = source.shape[-2:]
            if quad is None:
                quad = (0, 0), (width, 0), (width, height), (0, height)
            #: rois
            self.quads = [quad] * source.shape[0]
        #: Width for :class:`Rasterize` and for calculating
        #: projective transformations
        self.width = width
        #: Height
        self.height = height
        #: the output image, must be `None` when :class:`Rasterize` is used
        self.image = image


class Transformation:
    name: str

    def __init_subclass__(cls) -> None:
        if cls.name != "base":
            assert cls.name not in all_transformation_classes
            all_transformation_classes[cls.name] = cls
        super().__init_subclass__()

    def __call__(self, datum: Datum, random: Random) -> Datum:
        raise NotImplementedError


class GeometricTransform(Transformation):
    name = "base"

    def new_quad(self, quad: Quad, datum: Datum, random: Random) -> Quad:
        raise NotImplementedError

    def __call__(self, datum: Datum, random: Random) -> Datum:
        datum.quads = [
            self.new_quad(quad, datum, random) for quad in datum.quads
        ]
        return datum


class ArgDict(TypedDict):
    probability: NotRequired[float]


class Projective1pt(GeometricTransform):
    """
    Shifts one point of the quadrangle in random direction.

    .. image:: _static/transformations/projective1pt.svg

    .. code-block:: JSON

       {
           "name": "projective1pt",
           "x": {"name": "truncnorm", "a": -0.25, "b": 0.25},
           "y": {"name": "uniform", "a": -0.25, "b": 0.25}
       }
    """

    name = "projective1pt"

    class Args(ArgDict):
        name: Literal["projective1pt"]
        x: DistributionParams
        y: DistributionParams

    def __init__(
        self,
        x: DistributionParams,
        y: DistributionParams,
    ) -> None:
        """
        :param x: `create_distribution` arguments, dict
        :param y: `create_distribution` arguments, dict
        """
        self._x = create_distribution(x)
        self._y = create_distribution(y)

    def new_quad(self, quad: Quad, datum: Datum, random: Random) -> Quad:
        (x1, y1), (x2, y2) = quad[0], quad[2]  # guess the scale
        size = max(abs(x2 - x1), abs(y2 - y1))
        shift_x = self._x(random) * size
        shift_y = self._y(random) * size
        out_quad = list(quad)
        random_point_idx = random.randint(0, 3)
        point = out_quad[random_point_idx]
        out_quad[random_point_idx] = (point[0] + shift_x, point[1] + shift_y)

        return tuple(out_quad)


class Projective4pt(GeometricTransform):
    """
    Shifts four point of the quadrangle in random direction.

    .. image:: _static/transformations/projective4pt.svg

    .. code-block:: JSON

       {
           "name": "projective4pt",
           "x": {"name": "truncnorm", "a": -0.25, "b": 0.25},
           "y": {"name": "uniform", "a": -0.25, "b": 0.25}
       }
    """

    name = "projective4pt"

    class Args(ArgDict):
        name: Literal["projective4pt"]
        x: DistributionParams
        y: DistributionParams

    def __init__(
        self,
        x: DistributionParams,
        y: DistributionParams,
    ) -> None:
        """
        :param x: `create_distribution` arguments, dict
        :param y: `create_distribution` arguments, dict
        """
        self._x = create_distribution(x)
        self._y = create_distribution(y)

    def new_quad(self, quad: Quad, datum: Datum, random: Random) -> Quad:
        (x1, y1), (x2, y2) = quad[0], quad[2]  # guess the scale
        size = max(abs(x2 - x1), abs(y2 - y1))
        return tuple(
            [
                (
                    point[0] + self._x(random) * size,
                    point[1] + self._y(random) * size,
                )
                for point in quad
            ]
        )


class Flip(GeometricTransform):
    """
    Flips the quadrangle vertically or horizontally.
    Only changes points order, that is, visually the quadrangle doesn't
    change, but its visualization does.

    For diagonal names see: https://en.wikipedia.org/wiki/Main_diagonal
    """

    name = "flip"

    class Args(ArgDict):
        name: Literal["flip"]
        direction: Literal[
            "horizontal", "vertical", "primary_diagonal", "secondary_diagonal"
        ]

    def __init__(
        self,
        direction: Literal[
            "horizontal", "vertical", "primary_diagonal", "secondary_diagonal"
        ] = "horizontal",
    ) -> None:
        self._direction = getattr(self, direction)

    @staticmethod
    def horizontal(q: Quad) -> Quad:
        """
        .. image:: _static/transformations/flip_horizontal.svg

        .. code-block:: JSON

            {"name": "flip", "direction": "horizontal"}
        """
        return (q[1], q[0], q[3], q[2])

    @staticmethod
    def vertical(q: Quad) -> Quad:
        """
        .. image:: _static/transformations/flip_vertical.svg

        .. code-block:: JSON

           {"name": "flip", "direction": "vertical"}
        """
        return (q[2], q[3], q[0], q[1])

    @staticmethod
    def primary_diagonal(q: Quad) -> Quad:
        """
        .. image:: _static/transformations/flip_primary_diagonal.svg

        .. code-block:: JSON

           {"name": "flip", "direction": "primary_diagonal"}
        """
        return (q[0], q[3], q[2], q[1])

    @staticmethod
    def secondary_diagonal(q: Quad) -> Quad:
        """
        .. image:: _static/transformations/flip_secondary_diagonal.svg

        .. code-block:: JSON

           {"name": "flip", "direction": "secondary_diagonal"}
        """
        return (q[2], q[1], q[0], q[3])

    def new_quad(self, quad: Quad, datum: Datum, random: Random) -> Quad:
        return self._direction(quad)


class PaddingsAddition(GeometricTransform):
    """
    Adds random padding to the quadrangle sides.

    .. image:: _static/transformations/paddings_addition.svg

    .. code-block:: JSON

       {
           "name": "paddings_addition",
           "top": {"name": "uniform", "a": 0, "b": 0.25},
           "right": {"name": "uniform", "a": 0, "b": 0.25},
           "bottom": {"name": "uniform", "a": 0, "b": 0.25},
           "left": {"name": "uniform", "a": 0, "b": 0.25}
       }
    """

    name = "paddings_addition"

    class Args(ArgDict):
        name: Literal["paddings_addition"]
        top: DistributionParams
        right: DistributionParams
        bottom: DistributionParams
        left: DistributionParams

    def __init__(
        self,
        top: DistributionParams = 0.0,
        right: DistributionParams = 0.0,
        bottom: DistributionParams = 0.0,
        left: DistributionParams = 0.0,
    ) -> None:
        self._top = create_distribution(top)
        self._right = create_distribution(right)
        self._bottom = create_distribution(bottom)
        self._left = create_distribution(left)

    def new_quad(self, quad: Quad, datum: Datum, random: Random) -> Quad:
        (x1, y1), (x2, y2) = quad[0], quad[2]
        width, height = abs(x2 - x1), abs(y2 - y1)

        shift_top = self._top(random) * height
        shift_right = self._right(random) * width
        shift_bottom = self._bottom(random) * height
        shift_left = self._left(random) * width

        return (
            (quad[0][0] + shift_left, quad[0][1] + shift_top),
            (quad[1][0] + shift_right, quad[1][1] + shift_top),
            (quad[2][0] + shift_right, quad[2][1] + shift_bottom),
            (quad[3][0] + shift_left, quad[3][1] + shift_bottom),
        )


class ProjectivePaddingsAddition(PaddingsAddition):
    """
    Same as `PaddingsAddition`, but addition respects original projective
    transformation.

    .. image:: _static/transformations/projective_paddings_addition.svg

    .. code-block:: JSON

       {
           "name": "projective_paddings_addition",
           "top": {"name": "uniform", "a": 0, "b": 0.25},
           "right": {"name": "uniform", "a": 0, "b": 0.25},
           "bottom": {"name": "uniform", "a": 0, "b": 0.25},
           "left": {"name": "uniform", "a": 0, "b": 0.25}
       }
    """

    name = "projective_paddings_addition"

    class Args(ArgDict):
        name: Literal["projective_paddings_addition"]
        top: DistributionParams
        right: DistributionParams
        bottom: DistributionParams
        left: DistributionParams

    def __init__(
        self,
        top: DistributionParams = 0.0,
        right: DistributionParams = 0.0,
        bottom: DistributionParams = 0.0,
        left: DistributionParams = 0.0,
    ) -> None:
        self._top = create_distribution(top)
        self._right = create_distribution(right)
        self._bottom = create_distribution(bottom)
        self._left = create_distribution(left)

    def new_quad(self, quad: Quad, datum: Datum, random: Random) -> Quad:
        width, height = datum.width, datum.height

        shift_top = self._top(random) * height
        shift_right = self._right(random) * width
        shift_bottom = self._bottom(random) * height
        shift_left = self._left(random) * width

        rect = ((0.0, 0.0), (width, 0.0), (width, height), (0.0, height))
        m = calc_projection(rect, quad)

        return (
            projection_transform_point((shift_left, shift_top), m),
            projection_transform_point((width + shift_right, shift_top), m),
            projection_transform_point(
                (width + shift_right, height + shift_bottom), m
            ),
            projection_transform_point((shift_left, height + shift_bottom), m),
        )


class Rotate(GeometricTransform):
    """
    Rotates the quadrangle around its center.

    .. image:: _static/transformations/rotate.svg

    .. code-block:: JSON

       {
           "name": "rotate",
           "angle_deg": {"name": "uniform", "a": 0, "b": 360}
       }
    """

    name = "rotate"

    class Args(ArgDict):
        name: Literal["rotate"]
        angle_deg: DistributionParams

    def __init__(self, angle_deg: DistributionParams) -> None:
        """
        :param angle_deg: `create_distribution` arguments, dict
        """
        self._angle_deg = create_distribution(angle_deg)

    def new_quad(self, quad: Quad, datum: Datum, random: Random) -> Quad:
        angle = radians(self._angle_deg(random))
        return self._rotate_center(quad, angle)

    @staticmethod
    def _get_center(points: Quad):
        sx = sy = sL = 0
        for i, (x1, y1) in enumerate(points):
            x0, y0 = points[i - 1]
            L = hypot(x1 - x0, y1 - y0)
            sx += (x0 + x1) * 0.5 * L
            sy += (y0 + y1) * 0.5 * L
            sL += L
        return sx / sL, sy / sL

    @classmethod
    def _rotate_center(cls, quad: Quad, angle: float) -> Quad:
        ox, oy = cls._get_center(quad)
        cos_angle = cos(angle)
        sin_angle = sin(angle)
        return tuple(
            [
                (
                    ox + cos_angle * (x - ox) - sin_angle * (y - oy),
                    oy + sin_angle * (x - ox) + cos_angle * (y - oy),
                )
                for x, y in quad
            ]
        )


class ProjectiveShift(GeometricTransform):
    """
    Projectively shifts the quadrangle.

    .. image:: _static/transformations/projective_shift.svg

    .. code-block:: JSON

       {
           "name": "projective_shift",
           "x": {"name": "truncnorm", "a": -3.1, "b": 3.1}
       }
    """

    name = "projective_shift"

    class Args(ArgDict):
        name: Literal["projective_shift"]
        x: NotRequired[DistributionParams]
        y: NotRequired[DistributionParams]

    def __init__(
        self,
        x: DistributionParams = 0.0,
        y: DistributionParams = 0.0,
    ) -> None:
        """
        :param x: `create_distribution` arguments, dict
        :param y: `create_distribution` arguments, dict
        """
        self._x = create_distribution(x)
        self._y = create_distribution(y)

    def new_quad(self, quad: Quad, datum: Datum, random: Random) -> Quad:
        width, height = datum.width, datum.height
        x_shift = self._x(random) * width
        y_shift = self._y(random) * height
        rect = ((0.0, 0.0), (width, 0.0), (width, height), (0.0, height))
        m = calc_projection(rect, quad)
        ret = (
            projection_transform_point((x_shift, y_shift), m),
            projection_transform_point((width + x_shift, y_shift), m),
            projection_transform_point((width + x_shift, height + y_shift), m),
            projection_transform_point((x_shift, height + y_shift), m),
        )
        return ret


class Scale(GeometricTransform):
    """
    Scales the quadrangle to the factor specified in the `distribution`.

    .. image:: _static/transformations/scale.svg

    .. code-block:: JSON

       {
           "name": "scale",
           "factor": {"name": "truncnorm", "a": 0.7, "b": 1.3}
       }
    """

    name = "scale"

    class Args(ArgDict):
        name: Literal["scale"]
        factor: DistributionParams

    def __init__(self, factor: DistributionParams) -> None:
        """
        :param factor: `create_distribution` arguments, dict
        """
        self._factor = create_distribution(factor)

    def new_quad(self, quad: Quad, datum: Datum, random: Random) -> Quad:
        centre_x = sum([quad[i][0] for i in range(len(quad))]) / len(quad)
        centre_y = sum([quad[i][1] for i in range(len(quad))]) / len(quad)
        scale = self._factor(random)

        out_quad = list(quad)
        for i in range(len(out_quad)):
            out_quad[i] = (
                centre_x + scale * (out_quad[i][0] - centre_x),
                centre_y + scale * (out_quad[i][1] - centre_y),
            )

        return tuple(out_quad)


class Rasterize(Transformation):
    """
    Rasterizes the image from quadrangle using projective transform and
    the size specified in :class:`Datum`.

    .. code-block:: JSON

       {"name": "rasterize"}
    """

    name = "rasterize"
    _mode: Mode
    _padding_mode: PaddingMode

    class Args(TypedDict):
        name: Literal["rasterize"]
        mode: NotRequired[Mode]
        padding_mode: NotRequired[PaddingMode]

    def __init__(
        self, mode: Mode = "bilinear", padding_mode: PaddingMode = "zeros"
    ) -> None:
        self._mode = mode
        self._padding_mode = padding_mode

    def __call__(self, datum: Datum, _random: Random) -> Datum:
        rect = self.rect(datum)
        mats_py = [calc_projection(rect, quad) for quad in datum.quads]

        src = datum.source
        assert src is not None, "Source must be specified"
        assert datum.height is not None and datum.width is not None
        if len(mats_py) == 1:
            mat = torch.tensor(*mats_py, dtype=torch.float32)
            datum.image = projective_transform(
                src,
                mat,
                (datum.height, datum.width),
                mode=self._mode,
                padding_mode=self._padding_mode,
            )
        else:
            datum.image = torch.empty(
                size=(len(mats_py), src.shape[1], datum.height, datum.width),
                dtype=src.dtype,
                layout=src.layout,
                device=src.device,
            )
            for mat_py, dst in enumerate(mats_py, datum.image):
                mat = torch.tensor(mat_py, dtype=torch.float32)
                dst[:] = projective_transform(
                    src, mat, (datum.height, datum.width)
                )

        return datum

    def rect(self, datum: Datum) -> Quad:
        return (
            (0.0, 0.0),
            (datum.width, 0.0),
            (datum.width, datum.height),
            (0.0, datum.height),
        )


NoiseType: TypeAlias = Literal["heteroscedastic", "homoscedastic"]


class Noise(Transformation):
    """
    Adds normal noise to the image `numpy.random.RandomState.normal`.

    .. image:: _static/transformations/noise_homoscedastic.svg

    .. code-block:: JSON

       {
           "name": "noise",
           "std": {"name": "truncnorm", "a": 0, "b": 0.1}
       }

    .. image:: _static/transformations/noise_heteroscedastic.svg

    .. code-block:: JSON

       {
           "name": "noise",
           "std": {"name": "truncnorm", "a": 0, "b": 0.1},
           "type": "heteroscedastic"
       }
    """

    name = "noise"

    class Args(ArgDict):
        name: Literal["noise"]
        std: DistributionParams
        mean: NotRequired[DistributionParams]
        type: NotRequired[NoiseType]

    def __init__(
        self,
        std: DistributionParams,
        mean: DistributionParams | None = None,
        type: NoiseType = "homoscedastic",
    ) -> None:
        if mean is None:
            mean = 0.0 if type == "homoscedastic" else 1.0
        self._mean = create_distribution(mean)
        self._std = create_distribution(std)
        self._op = operator.iadd if type == "homoscedastic" else operator.imul

    def __call__(self, datum: Datum, random: Random):
        assert datum.image is not None, "missing datum.image"
        mean, std = self._mean(random), self._std(random)
        noise = torch.randn_like(
            datum.image
        )  # Generating on GPU is fastest with `torch.randn_like(...)`
        if std != 1.0:
            noise *= std
        if mean != 0.0:
            noise += mean
        self._op(datum.image, noise)
        return datum


DistributionsParams: TypeAlias = (
    DistributionParams | Sequence[DistributionParams]
)


class OperationTransform(Transformation):
    """
    Allow operation applying to "channel", "batch" or "tensor"
    """

    name = "base"
    _apply: Callable[[Tensor, Random], None]

    def __init__(self, per: PerEnum) -> None:
        self._apply = getattr(self, f"_apply_{per}")

    @staticmethod
    def _create_distributions(
        value: DistributionsParams, per: PerEnum = "tensor"
    ) -> Distribution | Callable[[Random], Tensor]:
        """
        Convenient way to user to specify custom distribution for each channel
        """
        if isinstance(value, Sequence):
            if per == "channel":
                raise ValueError(
                    "Can't use per channel operation when "
                    "multiple channels specified"
                )
            distributions = [create_distribution(d) for d in value]

            def channels_distribution(
                random: Random,
                distributions: list[Distribution] = distributions,
            ) -> Tensor:
                return torch.tensor([[[d(random)]] for d in distributions])

            return channels_distribution
        else:
            return create_distribution(value)

    def _op(self, image: Tensor, random: Random) -> None:
        raise NotImplementedError

    def _apply_tensor(self, image: Tensor, random: Random) -> None:
        self._op(image, random)

    def _apply_batch(self, image: Tensor, random: Random) -> None:
        for batch in image:
            self._op(batch, random)

    def _apply_channel(self, image: Tensor, random: Random) -> None:
        for batch in image:
            for channel in batch:
                self._op(channel, random)

    def __call__(self, datum: Datum, random: Random) -> Datum:
        assert datum.image is not None, "missing datum.image"
        self._apply(datum.image, random)
        return datum


class Add(OperationTransform):
    """
    Add the `value` to `Datum.image`

    .. image:: _static/transformations/add.svg

    .. code-block:: JSON

       {
           "name": "add",
           "value": {
               "name": "truncnorm",
               "a": -0.333,
               "b": 0.333,
           }
       }
    """

    name = "add"

    class Args(ArgDict):
        name: Literal["add"]
        value: DistributionsParams
        per: NotRequired[PerEnum]

    def __init__(
        self, value: DistributionsParams, per: PerEnum = "tensor"
    ) -> None:
        super().__init__(per)
        self._value = self._create_distributions(value, per)

    def _op(self, image: Tensor, random: Random) -> None:
        image += self._value(random)


class Multiply(OperationTransform):
    """
    Multiply `Datum.image` by the `factor`

    .. image:: _static/transformations/multiply.svg

    .. code-block:: JSON

       {
           "name": "multiply",
           "factor": {"name": "truncnorm", "a": 0.333, "b": 3.0}
       }
    """

    name = "multiply"

    class Args(ArgDict):
        name: Literal["multiply"]
        factor: DistributionsParams
        per: NotRequired[PerEnum]

    def __init__(
        self, factor: DistributionsParams, per: PerEnum = "tensor"
    ) -> None:
        super().__init__(per)
        self._factor = self._create_distributions(factor, per)

    def _op(self, image: Tensor, random: Random) -> None:
        image *= self._factor(random)


class Divide(OperationTransform):
    """
    Divide `Datum.image` by the `value`.

    .. code-block:: JSON

       {"name": "divide", "value": 255}
    """

    name = "divide"

    class Args(ArgDict):
        name: Literal["divide"]
        value: DistributionsParams
        per: NotRequired[PerEnum]

    def __init__(
        self, value: DistributionsParams, per: PerEnum = "tensor"
    ) -> None:
        super().__init__(per)
        self._value = self._create_distributions(value, per)

    def _op(self, image: Tensor, random: Random):
        image /= self._value(random)


class Pow(OperationTransform):
    """
    Raise `Datum.image` to the power of `pow`

    .. image:: _static/transformations/pow.svg

    .. code-block:: JSON

       {
           "name": "pow",
           "pow": {"name": "truncnorm", "a": 0.6, "b": 3.0}
       }
    """

    name = "pow"

    class Args(ArgDict):
        name: Literal["pow"]
        pow: DistributionsParams
        per: NotRequired[PerEnum]

    def __init__(
        self, pow: DistributionsParams, per: PerEnum = "tensor"
    ) -> None:
        super().__init__(per)
        self._pow = self._create_distributions(pow, per)

    def _op(self, image: Tensor, random: Random) -> None:
        pow = self._pow(random)
        torch.pow(image, pow, out=image)


class Log(Transformation):
    """
    Calculates one of the three logarithms for `Datum.image`

    .. figure:: _static/transformations/log_e.svg

       :math:`\\ln`

       .. code-block:: JSON

          {"name": "log", "base": "e"}
    """

    name = "log"

    class Args(ArgDict):
        name: Literal["log"]
        base: NotRequired[Literal["2", "e", "10"]]

    def __init__(self, base: Literal["2", "e", "10"] = "e") -> None:
        self._log_func = {
            "2": torch.log2,
            "e": torch.log,
            "10": torch.log10,
        }[base]

    def __call__(self, datum: Datum, random: Random) -> Datum:
        assert datum.image is not None, "missing datum.image"

        datum.image += 1.0
        self._log_func(datum.image, out=datum.image)
        return datum


class Clip(Transformation):
    """
    Clip `Datum.image` value to `min` and `max`

    .. code-block:: JSON

        {"name": "clip", "min": 0.0, "max": 1.0}
    """

    name = "clip"

    class Args(ArgDict):
        name: Literal["clip"]
        min: NotRequired[float]
        max: NotRequired[float]

    def __init__(self, min: float, max: float) -> None:
        self._min, self._max = min, max

    def __call__(self, datum: Datum, random: Random) -> Datum:
        assert datum.image is not None, "missing datum.image"

        torch.clip(datum.image, min=self._min, max=self._max, out=datum.image)
        return datum


class Grayscale(Transformation):
    """
    Average of all channels.
    Set `num_output_channels` to make number ou output channels not one.

    .. image:: _static/transformations/grayscale.svg

    .. code-block:: JSON

       {"name": "grayscale"}
    """

    name = "grayscale"

    class Args(ArgDict):
        name: Literal["grayscale"]
        num_output_channels: NotRequired[int]

    def __init__(self, num_output_channels: int = 1) -> None:
        self._channels = num_output_channels

    def __call__(self, datum: Datum, random: Random) -> Datum:
        assert datum.image is not None, "missing datum.image"

        from torchvision.transforms.v2.functional import rgb_to_grayscale

        datum.image = rgb_to_grayscale(datum.image, self._channels)
        return datum


class Sharpness(Transformation):
    """
    Makes image sharper.

    .. image:: _static/transformations/sharpness.svg

    .. code-block:: JSON

       {
           "name": "sharpness",
           "factor": {"name": "truncnorm", "a": 0.5, "b": 12}
       }
    """

    name = "sharpness"

    kernel = (
        (1 / 15, 1 / 15, 1 / 15),
        (1 / 15, 5 / 15, 1 / 15),
        (1 / 15, 1 / 15, 1 / 15),
    )

    class Args(ArgDict):
        name: Literal["sharpness"]
        factor: DistributionParams

    def __init__(self, factor: DistributionParams) -> None:
        self._factor = create_distribution(factor)

    def __call__(self, datum: Datum, random: Random) -> Datum:
        assert datum.image is not None, "missing datum.image"

        from torchvision.transforms.v2.functional import adjust_sharpness

        factor = self._factor(random)
        datum.image = adjust_sharpness(datum.image, factor)

        return datum

    @staticmethod
    def _blend(a: Tensor, b: Tensor, factor: float) -> Tensor:
        if factor == 0.0:
            return a
        if factor == 1.0:
            return b
        return a + (b - a) * factor


class Shading(Transformation):
    """
    Makes a random band darker.

    .. figure:: _static/transformations/shading.svg

    .. code-block:: JSON

       {
           "name": "shading",
           "value": {"name": "truncnorm", "a": -0.5, "b": 0.5}
       }
    """

    name = "shading"

    class Args(ArgDict):
        name: Literal["shading"]
        value: DistributionParams

    def __init__(self, value: DistributionParams) -> None:
        self._value = create_distribution(value)

    def get_mask(self, width: int, height: int, random: Random) -> Tensor:
        y = torch.arange(0, height).unsqueeze(1)
        x = torch.arange(0, width).unsqueeze(0)
        x1, y1 = random.randint(0, width - 1), random.randint(0, height - 1)
        x2, y2 = random.randint(0, width - 1), random.randint(0, height - 1)
        return x * (y2 - y1) - y * (x2 - x1) > x1 * y2 - x2 * y1

    def __call__(self, datum: Datum, random: Random) -> Datum:
        assert datum.image is not None, "No image to apply Shading to"
        height, width = datum.image.shape[-2:]
        for batch in datum.image:
            mask = self.get_mask(width, height, random)
            masked_pixels_count = torch.count_nonzero(mask)
            if masked_pixels_count == 0:
                continue
            batch[:, mask] += self._value(random) * batch.max()
        return datum


_INTERPOLATION = Literal["nearest", "nearest_exact", "bilinear", "bicubic"]


class Resize(Transformation):
    """
    Changes the size of a tensor. Usually this function is not needed
    because resizing is done in `rasterize` step.

    .. figure:: _static/transformations/resize.svg

    .. code-block:: JSON

       {"name": "resize", "width": 100, "height": 60}
    """

    name = "resize"

    class Args(ArgDict):
        name: Literal["resize"]
        width: int
        height: int
        antialias: NotRequired[bool]
        interpolation: NotRequired[_INTERPOLATION]

    def __init__(
        self,
        width: int,
        height: int,
        antialias: bool = True,
        interpolation: _INTERPOLATION = "bilinear",
    ) -> None:
        self._width = width
        self._height = height

        from torchvision.transforms.v2 import Resize, InterpolationMode

        interpolation_value = getattr(InterpolationMode, interpolation.upper())
        self._resize = Resize(
            [height, width],
            antialias=antialias,
            interpolation=interpolation_value,
        )

    def __call__(self, datum: Datum, random: Random) -> Datum:
        assert datum.image is not None, "missing datum.image"
        datum.image = self._resize(datum.image)
        return datum


Args: TypeAlias = (
    Projective1pt.Args
    | Projective4pt.Args
    | Flip.Args
    | PaddingsAddition.Args
    | ProjectivePaddingsAddition.Args
    | Rotate.Args
    | ProjectiveShift.Args
    | Scale.Args
    | Rasterize.Args
    | Sharpness.Args
    | Pow.Args
    | Log.Args
    | Multiply.Args
    | Divide.Args
    | Add.Args
    | Noise.Args
    | Clip.Args
    | Shading.Args
    | Grayscale.Args
    | Resize.Args
)


def create(kwargs: Args) -> Transformation:
    name: str = kwargs["name"]
    kwargs = kwargs.copy()
    del kwargs["name"]
    if name not in all_transformation_classes:
        raise Exception(
            f"Unknown transformation name `{name}`, "
            f"available names are: {sorted(all_transformation_classes)}"
        )
    cls = all_transformation_classes[name]

    try:
        return cls(**kwargs)
    except Exception as e:
        raise ValueError(
            f"Exception in {cls} ({name}) for arguments {kwargs}"
        ) from e
