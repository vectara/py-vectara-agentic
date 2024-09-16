from setuptools import setup, find_packages


def read_requirements():
    with open("requirements.txt") as req:
        return req.read().splitlines()


setup(
    name="vectara_agentic",
    version="0.1.8",
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
    keywords = ["LLM", "NLP", "RAG", "Agentic-RAG"],
    project_urls={
        "Documentation": "https://vectara.github.io/vectara-agentic-docs/",
    },
    python_requires=">=3.10",
)
