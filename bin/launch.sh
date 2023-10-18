#
# Copyright (c) 2021, The SmbRec Authors.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

#!/bin/bash

date=$(date +"%Y%m%d" -d "-1 days")
config=""
param=""
git=""
branch="master"

usage() {
    echo -e "Usage:\n  $0 -c 'pipelien.json' [-g 'https://oauth2:access_token@xxxx-workspace.git'] [-b 'branch'] [-p '--data_config.train_data.days 14'] [-d 20201220]" 1>&2
    exit 1
}

while getopts ":c:g:b:p:d:h" OPT; do
    case $OPT in
        c) config="$OPTARG" ;;
        g) git="$OPTARG" ;;
        b) branch="$OPTARG" ;;
        p) param="$OPTARG" ;;
        d) date="$OPTARG" ;;
        h) usage ;;
        ?) echo "Unknown config $OPTARG"
           usage ;;
    esac
done
shift $(($OPTIND - 1))

if [ "$git" ]; then
    mkdir -p /var/workspace
    cd /var/workspace
    export PYTHONPATH=$PYTHONPATH:/var/workspace
    git clone -b ${BRANCH} --single-branch --depth 1 $git
    if [ $? -ne 0 ]; then
        echo "git clone fail. $git"
        exit 1
    fi
    src=`ls -t | head -n 1`
    dest=`echo $src | sed 's/-/_/g'`
    mv $src $dest
fi

if [[ ! "$config" || ! -f $config ]]; then
    echo "Cannot find config file, ${config}"
    exit 1
fi

echo -e "Pipeline Config: ${config}\ndate: ${date}\ngit: ${git}\nParams: ${param}"

export CLASSPATH=${CLASSPATH}:$(${HADOOP_HOME}/bin/hadoop classpath --glob)

# NOTE(rabin) tf.dataset开启shuffle会导致内存泄漏，因此调整其内存分配策略。具体见：https://github.com/tensorflow/tensorflow/issues/44176
#MALLOC_MMAP_THRESHOLD_=0 python /usr/local/bin/sbrlaunch.py ${config} ${param} --date ${date}
python /usr/local/bin/sbrlaunch.py ${config} ${param} --date ${date}
if [ $? -ne 0 ]; then
    echo "run fail. config: ${config}, git: ${git}, date: ${date}"
    exit 1
fi

