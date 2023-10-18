#  Copyright (c) 2021, The SmbRec Authors.  All rights reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

import hashlib
import tarfile
import os
import shutil
from smbrec.utils import hdfs_utils


def listdir(path):
    """ 获取给定目录的文件列表

    Args:
        path: 目标目录

    Returns:
        目录下的文件列表
    """
    def _list_files(path):
        if os.path.isfile(path):
            data_files = [path]
        else:
            data_files = []
            names = os.listdir(path)
            names = sorted(names, key=lambda x: int(x)) if names[0].isdigit() else names
            for name in names:
                data_files.extend(_list_files(os.path.join(path, name)))
        return data_files
        
    # path is a file list
    if isinstance(path, list):
        data_files = [data_file for data_file in path
                      if os.path.isfile(data_file) or _is_hdfs(data_file)]
        return data_files

    # path is a hdfs path
    # https://github.com/tensorflow/examples/blob/master/community/en/docs/deploy/hadoop.md
    if _is_hdfs(path):
        data_files = hdfs_utils.listdir(path)
        return data_files

    # path is a local path
    if not os.path.exists(path):
        raise ValueError('path do not exist, ', path)
    data_files = _list_files(path)

    return data_files


def create_dir(path):
    """ 创建目录

    Args:
        path: 目录路径

    Returns:
        None
    """
    if _is_hdfs(path):
        return hdfs_utils.create_dir(path)
    return os.makedirs(path)


def exists(path):
    """ 判断路径是否存在

    Args:
        path: 待判断的路径

    Returns:
        True or False
    """
    if _is_hdfs(path):
        return hdfs_utils.exists(path)
    return os.path.exists(path)


def copy(source_path, dest_path, overwrite=True):
    if _is_hdfs(source_path):
        hdfs_utils.download(source_path, dest_path, overwrite=overwrite)
    elif _is_hdfs(dest_path):
        hdfs_utils.upload(source_path, dest_path, overwrite=overwrite)
    else:
        shutil.copytree(source_path, dest_path)


def readall(path):
    """ 读取给定文件的内容

    Args:
        path: 文件路径

    Returns:
        文件的内容
    """
    if _is_hdfs(path):
        return hdfs_utils.readall(path)
    with open(path, 'r') as inf:
        return inf.read()


def compress(src_path, dest_dir):
    """ 打包原路径到目标路径，并以md5命名tar包

    Args:
        src_path: 源数据路径
        dest_dir: 目标目录

    Returns:
        Tar包的路径
    """
    # tar compress
    tar_path = os.path.join(dest_dir, "tmp.tar.gz")
    with tarfile.open(tar_path, "w:gz") as tar:
        tar.add(src_path, arcname=os.path.basename(src_path))

    # calculate md5
    with open(tar_path, 'rb') as fp:
        data = fp.read()
        file_md5 = hashlib.md5(data).hexdigest()

    # rename tar file
    md5_path = os.path.join(dest_dir, file_md5 + ".tar.gz")
    os.rename(tar_path, md5_path)
    return md5_path


def _is_hdfs(path):
    return path.startswith("hdfs://")
