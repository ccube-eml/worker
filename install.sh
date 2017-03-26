#!/bin/sh

# Installs the ubuntu dependencies.
apt-get update
apt-get install -y \
	python3 \
	python3-dev \
	python3-pip \
	python3-setuptools \
	python3-numpy \
	python3-scipy \
	build-essential \
	libatlas-dev \
	libatlas3gf-base

# Downloads the latest cCube code.
mkdir /ccube
curl -sSL https://github.com/ccube-eml/worker/archive/master.tar.gz | tar -xz --strip 1 -C /ccube

# Installs the Python requirements.
RUN pip3 install -r /ccube/requirements.txt
