#-*- coding:utf-8 -*-
#!/usr/bin/python

import sys
import tensorflow as tf


def format_output(example, columns):
    if columns is None:
        print(example)
    else:
        values = {}
        for column in columns:
            feature = example.features.feature[column]
            values.setdefault(column, [])
            kind = feature.WhichOneof('kind')
            if kind is None:
                continue
            for value in getattr(getattr(feature, kind), "value"):
                values[column].append(str(value))
        output = " ".join([key+":"+",".join(value) for key,value in values.items()])
        print(output)


def print_parsed_example(input_path, columns=None, num=0, querys=[]):
    """ 打印符合要求的tfrecords

    Params:
        input_path: 待打印的tfreocrds文件路径
        columns: 打印的feature columns的column names
        num: 打印的tfrecords记录条数，0表示不限制条数
        querys: 打印的查询的条件，满足list中的任意一个条件即可。每个query是一个dict，
                key是待匹配的column name，value是匹配的column value，只有当tfrecord
                满足query的所有key value时，才算匹配成功

    Returns:
        打印满足条件的tfrecords
    """
    record_iterator = tf.compat.v1.io.tf_record_iterator(path=input_path)
    index = 0
    for string_record in record_iterator:
        if num and index >= num:
            return
        example = tf.train.Example()
        example.ParseFromString(string_record)
        if not querys:
            format_output(example, columns)
            index += 1
        else:
            for query in querys:
                for key, value in query.items():
                    feature = example.features.feature[key]
                    kind = feature.WhichOneof('kind')
                    if kind is None:
                        break
                    values = getattr(getattr(feature, kind), "value")
                    if len(values) == 0 or values[0] != value:
                        #print(values[0])
                        break
                else:
                    # example满足某条query条件则退出
                    format_output(example, columns)
                    index += 1
                    break


if __name__ == "__main__":
    #columns = ['authorId', '1000000_index', '1000000_value', '3000000_index', '3000000_value']
    querys = [{'user_id':530523851}]
    columns = None
    #querys = []

    num = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    print_parsed_example(sys.argv[1], columns=columns, num=num, querys=querys)
