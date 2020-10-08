from setuptools import setup, find_packages

setup(name='markups-irove',
      version='0.0.1',
      packages=find_packages(include=['markups', 'markups.*']),
      install_requires=[
          'opencv-python'
      ]
)
