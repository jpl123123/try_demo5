from setuptools import setup, find_packages

setup(
    name="kvpress-ascend",
    version="0.1.0",
    description=(
        "kvpress monkeypatch adapter for vLLM-Ascend v0.23.0: converts the kvpress "
        "KV-cache compression mechanism to vLLM-Ascend block-cache compaction "
        "without modifying vllm-ascend source."
    ),
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0",
        "numpy>=1.24",
    ],
    extras_require={
        "kvpress": ["kvpress>=0.1"],
    },
    entry_points={
        "vllm.general_plugins": [
            "kvpress_ascend = kvpress_ascend.plugin:register_kvpress_backend",
        ],
    },
)
