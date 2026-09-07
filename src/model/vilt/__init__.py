"""
================================================================================
ViLT 子包（第三方代码）
================================================================================
本目录下的代码是从 HuggingFace `transformers` 库复制的 ViLT
（Vision-and-Language Transformer）模型实现，非论文作者原创。
项目将其内置于仓库是为了：

    1. 避免依赖特定版本的 transformers（不同版本接口可能变化）；
    2. 方便直接调用 `ViltModel`（作为 ANGA 的冻结骨干）与
       `ViltImageProcessor`（图像预处理）。

论文作者原创的核心模型在 `src/model/ANGA.py` 与 `src/model/modules.py`。

各文件作用：
    - configuration_vilt.py          : ViLT 模型配置类 ViltConfig
    - modeling_vilt.py               : ViLT 的 PyTorch 实现（ViltModel 等）
    - image_processing_vilt.py       : 图像预处理器 ViltImageProcessor
    - processing_vilt.py             : 图文联合处理器 ViltProcessor
    - feature_extraction_vilt.py     : 已弃用的 ViltFeatureExtractor
    - convert_vilt_original_to_pytorch.py : 官方权重转换脚本

注：文件末尾的显式 import 语句被注释，实际导入走的是 transformers 的
懒加载机制（_LazyModule）；若懒加载不可用，可手动取消下面的注释。
================================================================================
"""
# Copyright 2022 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from typing import TYPE_CHECKING

from transformers.utils import OptionalDependencyNotAvailable, _LazyModule, is_torch_available, is_vision_available


_import_structure = {"configuration_vilt": ["ViltConfig"]}

try:
    if not is_vision_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    _import_structure["feature_extraction_vilt"] = ["ViltFeatureExtractor"]
    _import_structure["image_processing_vilt"] = ["ViltImageProcessor"]
    _import_structure["processing_vilt"] = ["ViltProcessor"]

try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    _import_structure["modeling_vilt"] = [
        "ViltForImageAndTextRetrieval",
        "ViltForImagesAndTextClassification",
        "ViltForTokenClassification",
        "ViltForMaskedLM",
        "ViltForQuestionAnswering",
        "ViltLayer",
        "ViltModel",
        "ViltPreTrainedModel",
    ]


if TYPE_CHECKING:
    from .configuration_vilt import ViltConfig

    try:
        if not is_vision_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .feature_extraction_vilt import ViltFeatureExtractor
        from .image_processing_vilt import ViltImageProcessor
        from .processing_vilt import ViltProcessor

    try:
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .modeling_vilt import (
            ViltForImageAndTextRetrieval,
            ViltForImagesAndTextClassification,
            ViltForMaskedLM,
            ViltForQuestionAnswering,
            ViltForTokenClassification,
            ViltLayer,
            ViltModel,
            ViltPreTrainedModel,
        )


else:
    import sys

    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure)
# from .modeling_vilt import ViltModel
# from .feature_extraction_vilt import ViltImageProcessor