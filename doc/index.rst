Welcome to Ballfish's documentation!
====================================

.. toctree::
   :maxdepth: 2
   :caption: Contents:

.. automodule:: ballfish
    :members:
    :exclude-members: create_augmentation

    .. autofunction:: create_augmentation(operations: Sequence[Args]) -> Callable[[Datum, Random], Datum]

.. automodule:: ballfish.distribution
    :members:
    :no-undoc-members:
    :exclude-members: ChoiceParams, ConstantParams, RandrangeParams, TruncnormParams, UniformParams

.. automodule:: ballfish.transformation
    :members:
    :no-undoc-members:
    :exclude-members: ArgDict, ConstantDict, Args


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
