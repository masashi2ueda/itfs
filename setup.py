from setuptools import setup, find_packages

# requirements.txt を読み込む関数
def parse_requirements(file):
    with open(file, "r") as f:
        return [line.strip() for line in f if line.strip() and not line.startswith("#")]

# requirements.txt の内容をリストとして取得
install_requires = parse_requirements("requirements.txt")

setup(
    name='itfs',
    version='0.0.1',
    description='',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    author='Masashi Ueda',
    author_email='',
    url='https://github.com/masashi2ueda/itfs',
    packages=find_packages(),
    install_requires=install_requires,
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
    ],
    python_requires='>=3.7'
)