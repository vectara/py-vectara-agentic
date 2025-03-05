from setuptools import setup, find_packages
import os
import re

def read_requirements():
    with open("requirements.txt") as req:
        return req.read().splitlines()

def read_version():
    version_file = os.path.join("vectara_agentic", "_version.py")
    with open(version_file, "r") as vf:
        content = vf.read()
    return re.search(r'__version__\s*=\s*["\']([^"\']+)["\']', content).group(1)


setup(
    name="vectara_agentic",
    version=read_version(),
    author="Ofer Mendelevitch",
    author_email="ofer@vectara.com",
    description="A Python package for creating AI Assistants and AI Agents with Vectara",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/vectara/py-vectara-agentic",
    packages=find_packages(),
    install_requires=read_requirements(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Development Status :: 4 - Beta",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    keywords=["LLM", "NLP", "RAG", "Agentic-RAG", "AI assistant", "AI Agent", "Vectara"],
    project_urls={
        "Documentation": "https://vectara.github.io/py-vectara-agentic/",
    },
    python_requires=">=3.10",
)
