# Spot the Difference!

## Build guide

Install `numpy`, `Flask` and `opencv`
```shell
$ conda install numpy
$ conda install opencv
$ pip3 install Flask
```

Install `pymeanshift` from source
```shell
$ cd lib/pymeanshift
$ python3 setup.py install
$ cd ../..
```

## Deployment
```shell
$ flask run
```