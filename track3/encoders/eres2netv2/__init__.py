# Vendored from 3D-Speaker (https://github.com/modelscope/3D-Speaker), Apache License 2.0.
# Only the import statements at the top of ERes2NetV2.py were changed to be
# package-relative, and the __main__ profiling block was removed. The model code
# itself is unmodified.

from .ERes2NetV2 import ERes2NetV2, BasicBlockERes2NetV2, BasicBlockERes2NetV2AFF

__all__ = ["ERes2NetV2", "BasicBlockERes2NetV2", "BasicBlockERes2NetV2AFF"]
