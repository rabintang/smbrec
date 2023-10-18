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

import os
from hdfs import InsecureClient, HdfsError
from urllib.parse import urlparse


def hdfs_client():
    """ 获取hdfs client

    Returns:
        HDFS client
    """
    assert "HADOOP_NAMENODE" in os.environ
    namenodes = os.environ["HADOOP_NAMENODE"].split(',')
    client = None
    for address in namenodes:
        client = InsecureClient(url=address, root='/')
        try:
            client.status("/")
            break
        except HdfsError as e:
            continue
    return client


def listdir(path):
    """ 获取给定目录的文件列表

    Args:
        path: 待获取文件列表的目录地址

    Returns:
        目录下的文件地址集合
    """
    client = hdfs_client()

    def _list_files(client, path):
        items = client.list(path, status=True)
        paths = []
        for item in items:
            if item[0] == '_SUCCESS':
                continue
            filepath = os.path.join(path, item[0])
            if item[1]['type'] == 'FILE':
                paths.append(filepath)
            else:
                paths.extend(_list_files(client, filepath))
        return paths

    filepath, url_parsed = _get_filepath(path)
    url_parsed = url_parsed._replace(fragment='')
    paths = _list_files(client, filepath)
    return [url_parsed._replace(path=path).geturl() for path in paths]


def exists(path):
    client = hdfs_client()
    filepath, _ = _get_filepath(path)
    try:
        client.status(filepath)
        return True
    except HdfsError as e:
        return False


def download(hdfs_path, local_path, overwrite=True):
    """ 从hdfs下载数据到本地

    Args:
        hdfs_path: hdfs路径
        local_path: 本地路径
        overwrite: 是否覆盖
    """
    client = hdfs_client()
    filepath, _ = _get_filepath(hdfs_path)
    client.download(filepath, local_path, overwrite=overwrite, n_threads=5)


def upload(local_path, hdfs_path, overwrite=False):
    """ 上传本地数据到hdfs

    Args:
        local_path: 本地数据路径
        hdfs_path: hdfs路径
        overwrite: 是否覆盖
    """
    client = hdfs_client()
    filepath, _ = _get_filepath(hdfs_path)
    client.upload(filepath, local_path, overwrite=overwrite, n_threads=5)


def create_dir(hdfs_path):
    """ 创建目录

    Args:
        hdfs_path: 目录路径

    Returns:
        None
    """
    client = hdfs_client()
    filepath, _ = _get_filepath(hdfs_path)
    client.makedirs(filepath)


def readall(path):
    """ 读取文件内容并返回

    Args:
        path: 文件路径

    Returns:
        文件内容
    """
    client = hdfs_client()
    filepath, _ = _get_filepath(path)
    with client.read(filepath) as reader:
        return reader.read()


def _get_filepath(path):
    """ 获取hdfs的绝对路径

    Args:
        path: 相对路径

    Returns:
        (hdfs绝对路径, ParseResultBytes对象)
    """
    hdfs_parsed = urlparse(path)
    filepath = hdfs_parsed.path + '#' + hdfs_parsed.fragment if hdfs_parsed.fragment else hdfs_parsed.path
    return filepath, hdfs_parsed

