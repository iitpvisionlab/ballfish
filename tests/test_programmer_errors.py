from __future__ import annotations
from typing import ForwardRef
from unittest import TestCase
from ballfish import Args
from ballfish import transformation, Transformation


def _gather_transformations():
    for cls_name, item in vars(transformation).items():
        if (
            isinstance(item, type)
            and issubclass(item, Transformation)
            and item is not Transformation
            and item.name != "base"
        ):
            yield cls_name, item


class ProgrammerErrors(TestCase):
    def test_no_duplicated_names(self):
        all_operations = set(
            arg.__annotations__["name"]
            ._evaluate(None, None, recursive_guard=frozenset())
            .__args__[0]
            for arg in Args.__args__
        )
        self.assertEqual(
            len(all_operations),
            len(Args.__args__),
            "Programmer error. Name duplication detected",
        )

    def test_name_is_correct(self):
        for cls_name, item in _gather_transformations():
            expected_name = "".join(
                word.capitalize() for word in item.name.split("_")
            )
            self.assertEqual(cls_name, expected_name)
            self.assertEqual(
                item.Args.__annotations__["name"]
                ._evaluate(None, None, recursive_guard=frozenset())
                .__args__[0],
                item.name,
            )

    def test_args_match_init(self):
        for cls_name, item in _gather_transformations():
            a = set(item.Args.__annotations__) - {"name", "probability"}
            if a:
                b = set(item.__init__.__annotations__)
                b.remove("return")
                self.assertEqual(a, b, f"invalid arguments in `{cls_name}`")
            else:
                self.assertIs(item.__init__, object.__init__)

    def test_operation_transform(self):
        from ballfish.transformation import OperationTransform

        for cls_name, item in _gather_transformations():
            if not issubclass(item, OperationTransform):
                continue
            annotations = item.Args.__annotations__
            self.assertIn("per", annotations, cls_name)
            self.assertEqual(
                annotations["per"].__forward_arg__, "NotRequired[PerEnum]"
            )

            self.assertEqual(
                item.__init__.__annotations__["per"],
                "PerEnum",
                f"`{cls_name}`",
            )
            if "value" in annotations:
                self.assertEqual(
                    annotations["value"].__forward_arg__,
                    "DistributionsParams",
                    f"`{cls_name}`",
                )
                self.assertEqual(
                    item.__init__.__annotations__["value"],
                    "DistributionsParams",
                    f"`{cls_name}`",
                )
            elif "factor" in annotations:
                self.assertEqual(
                    annotations["factor"].__forward_arg__,
                    "DistributionsParams",
                    f"`{cls_name}`",
                )
                self.assertEqual(
                    item.__init__.__annotations__["factor"],
                    "DistributionsParams",
                    f"`{cls_name}`",
                )
            elif "pow" in annotations:
                self.assertEqual(
                    annotations["pow"].__forward_arg__,
                    "DistributionsParams",
                    f"`{cls_name}`",
                )
                self.assertEqual(
                    item.__init__.__annotations__["pow"],
                    "DistributionsParams",
                    f"`{cls_name}`",
                )
            else:
                assert False, f"`{cls_name}`"
