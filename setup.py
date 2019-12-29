import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="bert_qa",
    version="0.0.2",
    author="Artit Wangperawong",
    author_email="artitw@gmail.com",
    description="Question-Answering system using state-of-the-art pre-trained language models.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/artitw/BERT_QA",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
    keywords='nlp text squad data science machine learning',
    install_requires=[
        "sentencepiece",
        'tensorflow>=2.0.0',
        'tensorflow-gpu>=2.0.0',
        'tensorflow-hub>=0.6.0',
    ],
)