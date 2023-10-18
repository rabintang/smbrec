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

config=""
param=""

usage() {
    echo -e "Usage:\n  $0 -c 'pipelien.json' [-p '--data_config.train_data.days 14']" 1>&2
    exit 1
}

while getopts ":c:p:h" OPT; do
    case $OPT in
        c) config="$OPTARG" ;;
        p) param="$OPTARG" ;;
        h) usage ;;
        ?) echo "Unknown config $OPTARG"
           usage ;;
    esac
done
shift $(($OPTIND - 1))

if [[ ! "$config" || ! -f $config ]]; then
    echo "Cannot find config file, ${config}"
    exit 1
fi

echo -e "Pipeline Config: ${config}\nParams: ${param}"

# NOTE(rabin) tf.dataset开启shuffle会导致内存泄漏，因此调整其内存分配策略。具体见：https://github.com/tensorflow/tensorflow/issues/44176
python launcher/launch.py ${config} ${param}
if [ $? -ne 0 ]; then
    echo "run fail. config: ${config}"
    exit 1
fi
