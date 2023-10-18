#!/bin/bash

PWD=$(cd `dirname $0`; pwd)
version=$(grep "VERSION = " ${PWD}/../setup.py | awk '{print $3}' | tr -d "'")

echo $version
cd ${PWD}/../
git tag v${version}
git push origin v${version}
