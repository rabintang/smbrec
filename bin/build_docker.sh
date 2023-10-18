#!/bin/bash

PWD=$(cd `dirname $0`; pwd)
name=pangu/smbrec
version=$(grep "VERSION = " ${PWD}/../setup.py | awk '{print $3}' | tr -d "'")

cd ${PWD}/../
sudo docker build -t ${name}:v${version} .
