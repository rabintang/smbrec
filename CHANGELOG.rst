^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Changelog for SmbRec
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
0.1.0 (2021-07)
------------------
* 支持多任务学习，以及多任务学习配置项；
* Model提供 prepare_* 相关方法，用于通过配置实例化对应实例；
* 提供 launcher/launch.py，用于从配置直接完成从 数据解析、加载，到模型训练、评估、导出 等个流程；
* 提供 FakeData 用于生成模拟数据；
* 支持模型的增量训练；

0.1.1 (2021-10)
------------------
* 支持Docker调用；
* 提供predict.py；
* 实现 VocabEncoder、SequenceSlice、AttentionPooling 等Layer；
* 实现 NamedSparseTensorSpec 以支持SparseTensor的导出；
* 修复一些发现的bugs；

0.1.3 (2023-1)
------------------
* 新增支持模型md5校验以及tar.gz打包
* 支持tensorflow1.14
* 增加对data_config.[train_data|eval_data].cache_path的支持，优先从cache_path读数据