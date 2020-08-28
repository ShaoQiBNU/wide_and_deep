wide & deep
============

# 一. 背景

> wide and deep模型是谷歌在2016年发布的一类用于分类和回归的模型，并应用到了 Google Play 的应用推荐中。wide and deep 模型的核心思想是结合线性模型的记忆能力（memorization）和 DNN 模型的泛化能力（generalization），在训练过程中同时优化 2 个模型的参数，从而达到整体模型的预测能力最优。

# 二. wide & deep model

> wide and deep模型结构设计如图所示：

![image](https://github.com/ShaoQiBNU/wide_and_deep/blob/master/images/1.png)

## (一) wide

>wide 端如图1左边所示，是一种特殊的神经网络，它的输入和输出直接相连，属于广义线性模型。输入特征可以是连续特征，也可以是稀疏的离散特征，离散特征之间进行交叉后可以构成更高维的离散特征。线性模型训练中通过 L1 正则化，能够很快收敛到有效的特征组合中。

#### cross-product transformation

> cross-product transformation定义如下：

![image](https://github.com/ShaoQiBNU/wide_and_deep/blob/master/images/2.png)

## (二) deep

> deep 端对应的是 DNN 模型，每个特征对应一个低维的实数向量，称之为特征的 embedding。DNN 模型通过反向传播调整隐藏层的权重，并且更新特征的 embedding。

## (三) wide & deep 

> 整个模型的输出是线性模型输出与DNN模型输出的叠加，模型训练采用的是联合训练（joint training），训练误差会同时反馈到线性模型和 DNN 模型中进行参数更新。相比于 ensemble learning 中单个模型进行独立训练，模型的融合仅在最终预测阶段进行。

> joint training 中模型的融合是在训练阶段进行的，单个模型的权重更新会受到 wide 端和 deep 端对模型训练误差的共同影响。因此在模型的特征设计阶段，wide 端模型和 deep 端模型只需要分别专注于擅长的方面，wide 端模型通过离散特征的交叉组合进行 memorization，deep 端模型通过特征的 embedding 进行 generalization，这样单个模型的大小和复杂度也能得到控制，而整体模型的性能仍能得到提高。

> **wide端采用FTRL和L1正则化来优化，deep端采用AdaGrad算法来优化，wide & deep Model的后向传播采用mini-batch stochastic optimization。**

![image](https://github.com/ShaoQiBNU/wide_and_deep/blob/master/images/3.png)

#### Memorization

> 之前大规模稀疏输入的处理是：通过线性模型 + 特征交叉。所带来的Memorization以及记忆能力非常有效和可解释。但是Generalization（泛化能力）需要更多的人工特征工程。

#### Generalization： 
> 相比之下，DNN几乎不需要特征工程。通过对低纬度的dense embedding进行组合可以学习到更深层次的隐藏特征。但是，缺点是有点over-generalize（过度泛化）。推荐系统中表现为：会给用户推荐不是那么相关的物品，尤其是user-item矩阵比较稀疏并且是high-rank（高秩矩阵）

#### 两者区别

> Memorization趋向于更加保守，推荐用户之前有过行为的items。相比之下，generalization更加趋向于提高推荐系统的多样性（diversity）。

#### Wide & Deep
> Wide & Deep包括两部分：线性模型 + DNN部分。结合上面两者的优点，平衡memorization和generalization，服务于推荐系统。相比于wide-only和deep-only的模型，wide & deep提升显著。

# 三. 实例

> Google官方给出了一个实例——预测收入是否超过5万美元，二分类问题。 数据集概览如下：

![image](https://github.com/ShaoQiBNU/wide_and_deep/blob/master/images/3.jpg)

## (一) 数据下载

> 运行官方代码，下载数据，代码如下：

```python
# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Download and clean the Census Income Dataset."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import sys

from six.moves import urllib
import tensorflow as tf

DATA_URL = 'https://archive.ics.uci.edu/ml/machine-learning-databases/adult'
TRAINING_FILE = 'adult.data'
TRAINING_URL = '%s/%s' % (DATA_URL, TRAINING_FILE)
EVAL_FILE = 'adult.test'
EVAL_URL = '%s/%s' % (DATA_URL, EVAL_FILE)

parser = argparse.ArgumentParser()

parser.add_argument(
    '--data_dir', type=str, default='./',
    help='Directory to download census data')


def _download_and_clean_file(filename, url):
  """Downloads data from url, and makes changes to match the CSV format."""
  temp_file, _ = urllib.request.urlretrieve(url)
  with tf.gfile.Open(temp_file, 'r') as temp_eval_file:
    with tf.gfile.Open(filename, 'w') as eval_file:
      for line in temp_eval_file:
        line = line.strip()
        line = line.replace(', ', ',')
        if not line or ',' not in line:
          continue
        if line[-1] == '.':
          line = line[:-1]
        line += '\n'
        eval_file.write(line)
  tf.gfile.Remove(temp_file)


def main(_):
  if not tf.gfile.Exists(FLAGS.data_dir):
    tf.gfile.MkDir(FLAGS.data_dir)

  training_file_path = os.path.join(FLAGS.data_dir, TRAINING_FILE)
  _download_and_clean_file(training_file_path, TRAINING_URL)

  eval_file_path = os.path.join(FLAGS.data_dir, EVAL_FILE)
  _download_and_clean_file(eval_file_path, EVAL_URL)


if __name__ == '__main__':
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(argv=[sys.argv[0]] + unparsed)
```

## (二) wide & deep

> 运用tensorflow的tf.estimator.DNNLinearCombinedClassifier函数对收入进行预测，代码如下：

```python
import tensorflow as tf

# 1. 最基本的特征：
# Continuous columns. Wide和Deep组件都会用到。
age = tf.feature_column.numeric_column('age')
education_num = tf.feature_column.numeric_column('education_num')
capital_gain = tf.feature_column.numeric_column('capital_gain')
capital_loss = tf.feature_column.numeric_column('capital_loss')
hours_per_week = tf.feature_column.numeric_column('hours_per_week')


# 离散特征
education = tf.feature_column.categorical_column_with_vocabulary_list(
    'education', ['Bachelors', 'HS-grad', '11th', 'Masters', '9th', 'Some-college',
        'Assoc-acdm', 'Assoc-voc', '7th-8th', 'Doctorate', 'Prof-school',
        '5th-6th', '10th', '1st-4th', 'Preschool', '12th'])

marital_status = tf.feature_column.categorical_column_with_vocabulary_list(
    'marital_status', ['Married-civ-spouse', 'Divorced', 'Married-spouse-absent',
        'Never-married', 'Separated', 'Married-AF-spouse', 'Widowed'])

relationship = tf.feature_column.categorical_column_with_vocabulary_list(
    'relationship', ['Husband', 'Not-in-family', 'Wife', 'Own-child', 'Unmarried',
        'Other-relative'])

workclass = tf.feature_column.categorical_column_with_vocabulary_list(
    'workclass', ['Self-emp-not-inc', 'Private', 'State-gov', 'Federal-gov',
        'Local-gov', '?', 'Self-emp-inc', 'Without-pay', 'Never-worked'])

# hash buckets
occupation = tf.feature_column.categorical_column_with_hash_bucket(
    'occupation', hash_bucket_size=1000
)

# Transformations
age_buckets = tf.feature_column.bucketized_column(
    age, boundaries=[18, 25, 30, 35, 40, 45, 50, 55, 60, 65]
)


# 2. The Wide Model: Linear Model with CrossedFeatureColumns
"""
The wide model is a linear model with a wide set of *sparse and crossed feature* columns
Wide部分用了一个规范化后的连续特征age_buckets，其他的连续特征没有使用
"""
base_columns = [
    # 全是离散特征
    education, marital_status, relationship, workclass, occupation, age_buckets,
]

crossed_columns = [
    tf.feature_column.crossed_column(
        ['education', 'occupation'], hash_bucket_size=1000),
    tf.feature_column.crossed_column(
        [age_buckets, 'education', 'occupation'], hash_bucket_size=1000
    )]


# 3. The Deep Model: Neural Network with Embeddings
"""
1. Sparse Features -> Embedding vector -> 串联(Embedding vector, 连续特征) -> 输入到Hidden Layer
2. Embedding Values随机初始化
3. 另外一种处理离散特征的方法是：one-hot or multi-hot representation. 但是仅仅适用于维度较低的，embedding是更加通用的做法
4. embedding_column(embedding);indicator_column(multi-hot);
"""
deep_columns = [
    age,
    education_num,
    capital_gain,
    capital_loss,
    hours_per_week,
    tf.feature_column.indicator_column(workclass),
    tf.feature_column.indicator_column(education),
    tf.feature_column.indicator_column(marital_status),
    tf.feature_column.indicator_column(relationship),

    # To show an example of embedding
    tf.feature_column.embedding_column(occupation, dimension=8)
]

model_dir = './model/wide_deep'

# 4. Combine Wide & Deep
model = tf.estimator.DNNLinearCombinedClassifier(
    model_dir=model_dir,
    linear_feature_columns=base_columns + crossed_columns,
    dnn_feature_columns=deep_columns,
    dnn_hidden_units=[100, 50]
)

# 5. Train & Evaluate
_CSV_COLUMNS = [
    'age', 'workclass', 'fnlwgt', 'education', 'education_num',
    'marital_status', 'occupation', 'relationship', 'race', 'gender',
    'capital_gain', 'capital_loss', 'hours_per_week', 'native_country',
    'income_bracket'
]
_CSV_COLUMN_DEFAULTS = [[0], [''], [0], [''], [0], [''], [''], [''], [''], [''],
                        [0], [0], [0], [''], ['']]
_NUM_EXAMPLES = {
    'train': 32561,
    'validation': 16281,
}

def input_fn(data_file, num_epochs, shuffle, batch_size):
    """为Estimator创建一个input function"""
    assert tf.gfile.Exists(data_file), "{0} not found.".format(data_file)

    def parse_csv(line):
        print("Parsing", data_file)

        # tf.decode_csv会把csv文件转换成 a list of Tensor,一列一个
        # record_defaults用于指明每一列的缺失值用什么填充
        columns = tf.decode_csv(line, record_defaults=_CSV_COLUMN_DEFAULTS)

        features = dict(zip(_CSV_COLUMNS, columns))
        labels = features.pop('income_bracket')

        # tf.equal(x, y) 返回一个bool类型Tensor， 表示x == y, element-wise
        return features, tf.equal(labels, '>50K')

    dataset = tf.data.TextLineDataset(data_file).map(parse_csv, num_parallel_calls=5)

    if shuffle:
        dataset = dataset.shuffle(buffer_size=_NUM_EXAMPLES['train'] + _NUM_EXAMPLES['validation'])

    dataset = dataset.repeat(num_epochs)
    dataset = dataset.batch(batch_size)

    iterator = dataset.make_one_shot_iterator()
    batch_features, batch_labels = iterator.get_next()

    return batch_features, batch_labels

# Train + Eval
train_epochs = 30
epochs_per_eval = 2
batch_size = 40
train_file = 'adult.data'
test_file = 'adult.test'

for n in range(train_epochs // epochs_per_eval):
    model.train(input_fn=lambda: input_fn(train_file, epochs_per_eval, True, batch_size))
    results = model.evaluate(input_fn=lambda: input_fn(test_file, 1, False, batch_size))

    # Display Eval results
    print("Results at epoch {0}".format((n+1) * epochs_per_eval))
    print('-'*30)

    for key in sorted(results):
        print("{0:20}: {1:.4f}".format(key, results[key]))
```

> 其中关于特征的处理见：https://www.jianshu.com/p/fceb64c790f3 和 https://blog.csdn.net/sxf1061926959/article/details/78440220?readlog

## (三) R语言版本

https://tensorflow.rstudio.com/tfestimators/articles/examples/wide_and_deep.html

https://github.com/rstudio/tfestimators/blob/master/vignettes/examples/wide_and_deep.R



参考：

https://blog.csdn.net/u010352603/article/details/80590129

https://ai.googleblog.com/2016/06/wide-deep-learning-better-together-with.html

https://github.com/tensorflow/models/tree/master/official/r1/wide_deep
