# pipeline wraps transformers pipeline with extensions in DocQuery
# we're simply re-exporting it here.
import sys

from transformers.utils import _LazyModule

from .version import VERSION


_import_structure = {
    "transformers_patch": ["pipeline"],
}

sys.modules[__name__] = _LazyModule(
    __name__,
    globals()["__file__"],
    _import_structure,
    module_spec=__spec__,
    extra_objects={"__version__": VERSION},
)
