import io
import setuptools

setuptools.setup(
    name="paraphrase_reranker",
    version="0.0.5",
    author="Ilya Koziev",
    author_email="inkoziev@gmail.com",
    description="Paraphrase detection and reranking",
    url="https://github.com/Koziev/paraphrase_reranker",
    packages=setuptools.find_packages(),
    package_data={'paraphrase_reranker': ['*.py', '*.cfg', '*.pt']},
    include_package_data=True,
)
