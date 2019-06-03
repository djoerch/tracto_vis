import setuptools
import glob

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="mrathon",
    version="0.0.1",
    author="Daniel Jorgens",
    author_email="danjorg@kth.se",
    description="Visualisation of streamlines with a sliding window and gif creation.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://bitbucket.org/djoerch/mrathon",
    packages=setuptools.find_packages(),
    install_requires=[
        "dipy>=0.15.0",
        "fury>=0.2.0",
        "imageio==2.5.0",
        "imageio-ffmpeg==0.3.0",
        "nibabel>=2.4.0",
        "Pillow==6.0.0",
        "tqdm>=4.31.1",
        "vtk>=8.1.2",
        "numpy>=1.16.3"
    ],
    scripts=glob.glob('scripts/*'),
)
