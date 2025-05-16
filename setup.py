from setuptools import setup, find_packages
import os

# 项目名称
PROJECT_NAME = "ReasonCraft"
# 项目版本
VERSION = "0.1.0"
# 项目描述
DESCRIPTION = "A tool to bridge natural language with Prolog using LLMs (ZhipuAI)."

# 从 README.md 读取长描述
# 这使得 PyPI 页面可以显示你的 README 内容
try:
    with open(os.path.join(os.path.dirname(__file__), 'README.md'), encoding='utf-8') as f:
        LONG_DESCRIPTION = f.read()
except FileNotFoundError:
    LONG_DESCRIPTION = DESCRIPTION

setup(
    name=PROJECT_NAME,
    version=VERSION,
    author="Your Name / Organization",  # 替换为你的名字或组织
    author_email="your.email@example.com",  # 替换为你的邮箱
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/ReasonCraft",  # 替换为你的项目URL
    
    # find_packages() 会自动找到 reasoncraft 包目录
    # 'reasoncraft' 是包含 __init__.py 和 main.py 的目录名
    packages=find_packages(where="."), # 如果你的包在 src/reasoncraft，则用 find_packages(where="src") 和 package_dir={'': 'src'}

    # 依赖列表
    install_requires=[
        "pyswip>=0.2.10", # pyswip 的一个合理版本
        "zhipuai>=2.0.0",  # 请检查 ZhipuAI SDK 的最新推荐版本
        # 're' 是标准库，不需要列出
        # 'os' 是标准库，不需要列出
    ],
    
    # Python 版本要求
    python_requires='>=3.7', # 基于代码中使用的 f-strings 和 ZhipuAI SDK 的可能要求

    # 定义命令行入口点
    # 这允许用户在安装后直接从命令行运行 'reasoncraft' 命令
    # 'reasoncraft' 是命令名
    # 'reasoncraft.main:main' 指向 reasoncraft 包中 main.py 文件里的 main 函数
    entry_points={
        'console_scripts': [
            f'{PROJECT_NAME.lower()}=reasoncraft.main:main',
        ],
    },

    # 附加元数据
    classifiers=[
        "Development Status :: 3 - Alpha",  # 开发阶段: 3 - Alpha, 4 - Beta, 5 - Production/Stable
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Text Processing :: Linguistic",
        "License :: OSI Approved :: MIT License",  # 替换为你选择的许可证
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Operating System :: OS Independent",
    ],
    
    # 如果你想把 knowledge_base.pl 作为包的一部分分发（例如一个默认的KB）
    # 你需要把它放在 reasoncraft 包内，比如 reasoncraft/knowledge_base_default.pl
    # 然后在 main.py 中用 importlib.resources 来定位它
    package_data={
        'reasoncraft': ['knowledge_base.pl'],
    },
    # include_package_data=True, # 如果使用 MANIFEST.in
)