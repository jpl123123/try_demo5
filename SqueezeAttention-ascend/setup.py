from setuptools import setup, find_packages

setup(
    name="SqueezeAttention-ascend",
    version="0.1.0",
    description=(
        "SqueezeAttention monkeypatch adapter for vLLM-Ascend v0.23.0: converts "
        "the SqueezeAttention 2D KV-budget mechanism (layer-wise KMeans budgets x "
        "streaming token eviction) to vLLM-Ascend block-cache compaction without "
        "modifying vllm-ascend source."
    ),
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0",
        "numpy>=1.24",
    ],
    extras_require={
        "sklearn": ["scikit-learn>=1.0"],
    },
    entry_points={
        "vllm.general_plugins": [
            "squeezeattention_ascend = squeezeattention_ascend.plugin:register_squeezeattention_backend",
        ],
    },
)
