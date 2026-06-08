"""[向后兼容 shim] 保留原顶层模块入口。

历史版本中 ``webui.py`` 通过 ``import ner_model as zwk`` 访问 NER 相关全部功能。
为了避免破坏现有调用点，本文件保留为薄壳，全部从 :mod:`ner` 子包重新导出。

新代码请直接 ``from ner import ...``。

训练入口已迁移到 :mod:`ner.train`，可用::

    python -m ner.train
"""

from ner import (  # noqa: F401  re-export 兼容
    Bert_Model,
    Entity_Extend,
    Nerdataset,
    build_tag2idx,
    find_entities,
    get_data,
    get_ner_result,
    merge,
    rule_find,
    tfidf_alignment,
)

# 历史模块级常量，保留以兼容直接访问 ``ner_model.cache_model`` 的潜在用法
cache_model = 'best_roberta_rnn_model_ent_aug'

__all__ = [
    "Bert_Model",
    "Entity_Extend",
    "Nerdataset",
    "build_tag2idx",
    "find_entities",
    "get_data",
    "get_ner_result",
    "merge",
    "rule_find",
    "tfidf_alignment",
    "cache_model",
]
