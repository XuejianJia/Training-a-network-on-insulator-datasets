# 在自己的数据集上使用slim进行神经网络训练
在进行以下操作之前，请确保您已经将slim安装到了您的设备上。安装详情见[main TF-Slim page](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/contrib/slim).


----------
## Table of contents

<a href="#Convert">将自己的数据集转换成TF-Record格式</a><br>
<a href="#Per_train">导入预训练模型</a><br>
<a href="#Fine_tuning">在自己的数据集上Fine tuning模型</a><br>

## 转换数据集格式
<a id='Covert'></a>

这里以绝缘子数据集为例，讲解了怎样将自己的图片数据集转换为TensorFlow能接受的TF-Record格式

首先打开[insulators.py](https://github.com/XuejianJia/Training-a-network-on-insulator-datasets/blob/master/slim/datasets/insulators.py),按照自己的数据集情况修改`_FILE_PATTERN`、`SPLITS_TO_SIZES`、`_NUM_CLASSES`这3个参数：</a><br>
`_FILE_PATTERN` 表示生成的TF-Record文件的文件名前缀；</a><br>
`SPLITS_TO_SIZES` 表示数据集的分割情况，即训练集包含多少个样本，验证集包含多少个样本；</a><br>
`_NUM_CLASSES` 表示数据集的类别数。</a><br>
另外，`_ITEMS_TO_DESCRIPTIONS` 是对生成文件的描述，可根据实际情况修改。</a><br>
代码其他部分保持不变。

然后打开[convert_insulators.py](https://github.com/XuejianJia/Training-a-network-on-insulator-datasets/blob/master/slim/datasets/convert_insulators.py),按照自己的数据集情况修改`_NUM_VALIDATION`、`_NUM_SHARDS`参数：</a><br>
`_NUM_VALIDATION` 表示分割出来的验证集样本数量，要和上一步的数量一致！</a><br>
`_NUM_SHARDS` 表示生成的TF-Record文件需要分割成几块，如果数据集较大，可选择生成多个TF-Record文件，例如，此处绝缘子样本数为237，样本较少，则将这个参数设置为1，即只生成一个TF-Record文件。</a><br>
在`_get_filenames_and_classes` 函数中，修改 `insulator_root = os.path.join(dataset_dir, 'insulator_photos')` 中的参数，`'insulator_photos'` 为保存数据集样本的文件夹名字。该文件夹下包含以类名为文件名的子文件夹，在各个子文件夹下包含png或jpg格式的图片。注意：如果修改了 `insulator_root` 的参数名，则函数内相同的参数名也要修改。</a><br>
在 `_get_dataset_filename` 函数中，修改 `'insulators_%s_%05d-of-%05d.tfrecord'` 中的前缀为自己的数据集名称，和第一步中的前缀对应。</a><br>

#### 将修改后的`insulator.py`和`convert_insulators.py`放在/slim/datasets/路径下！

在/slim/路径下运行以下语句即可将数据集转换为TF-Record格式<br>

    python
    from datasets import convert_insulators
    convert_insulators.run('数据集的保存路径（insulator_photos）的母路径')
    

## 导入预训练模型
<a id='Per_train'></a>
详见[insulator_walkthrough.ipynb](https://github.com/XuejianJia/Training-a-network-on-insulator-datasets/blob/master/slim/insulator_walkthrough.ipynb).

####首先定位在/slim/路径下，导入各种依赖包。

    from __future__ import absolute_import
    from __future__ import division
    from __future__ import print_function
    
    import matplotlib
    %matplotlib inline
    import matplotlib.pyplot as plt
    import math
    import numpy as np
    import tensorflow as tf
    import time
    
    from datasets import dataset_utils
    
    # Main slim library
    from tensorflow.contrib import slim

#### 下载Inception V1 预训练模型

    url = "http://download.tensorflow.org/models/inception_v1_2016_08_28.tar.gz"
    checkpoints_dir = '/tmp/checkpoints'
    
    if not tf.gfile.Exists(checkpoints_dir):
    	tf.gfile.MakeDirs(checkpoints_dir)
    
    dataset_utils.download_and_uncompress_tarball(url, checkpoints_dir)

如果需要导入其他模型的预训练参数，修改url的值即可。

#### 将预训练模型导入到网络初始化参数,并从网上下载一张图片来验证参数是否导入正确。<br>

    import os
    
    try:
    	import urllib2 as urllib
    except ImportError:
    	import urllib.request as urllib
    
    from datasets import imagenet
    from nets import inception
    from preprocessing import inception_preprocessing
    
    image_size = inception.inception_v1.default_image_size
    
    with tf.Graph().as_default():
    	url = 'https://upload.wikimedia.org/wikipedia/commons/7/70/EnglishCockerSpaniel_simon.jpg'
    	image_string = urllib.urlopen(url).read()
    	image = tf.image.decode_jpeg(image_string, channels=3)
    	processed_image = inception_preprocessing.preprocess_image(image, image_size, image_size, is_training=False)
    	processed_images  = tf.expand_dims(processed_image, 0)
    
    	# Create the model, use the default arg scope to configure the batch norm parameters.
    	with slim.arg_scope(inception.inception_v1_arg_scope()):
    		logits, _ = inception.inception_v1(processed_images, num_classes=1001, is_training=False)
    	probabilities = tf.nn.softmax(logits)
    
    	init_fn = slim.assign_from_checkpoint_fn(os.path.join(checkpoints_dir, 'inception_v1.ckpt'),slim.get_model_variables('InceptionV1'))
    
    	with tf.Session() as sess:
    		init_fn(sess)
    		np_image, probabilities = sess.run([image, probabilities])
    		probabilities = probabilities[0, 0:]
    		sorted_inds = [i[0] for i in sorted(enumerate(-probabilities), key=lambda x:x[1])]
    
    	plt.figure()
    	plt.imshow(np_image.astype(np.uint8))
    	plt.axis('off')
    	plt.show()
    
    	names = imagenet.create_readable_names_for_imagenet_labels()
    	for i in range(5):
    		index = sorted_inds[i]
    		print('Probability %0.2f%% => [%s]' % (probabilities[index] * 100, names[index]))

我们需要将每张图片都转换成模型需要的大小，但是单纯的从checkpoint文件中，我们无法获取这个参数，所以，这里应用TensorFlow提供的`preprocessing`文件夹中的`inception_preprocessing.py`来进行图像预处理，当应用其他网络模型时，则选择该文件夹下的相应预处理函数。<br>
当应用其他模型结构时，对上面程序需要修改的地方有：<br>

    image_size = inception.inception_v1.default_image_size

    processed_image = inception_preprocessing.preprocess_image(image, image_size, image_size, is_training=False)

	with slim.arg_scope(inception.inception_v1_arg_scope()):
        logits, _ = inception.inception_v1(processed_images, num_classes=1001, is_training=False)

	init_fn = slim.assign_from_checkpoint_fn(
        os.path.join(checkpoints_dir, 'inception_v1.ckpt'),
        slim.get_model_variables('InceptionV1'))

## Fine tuning
<a id='Fine_tuning'></a>

定义一个数据读取和预处理函数

    def load_batch(dataset, batch_size=32, height=299, width=299, is_training=False):
    """Loads a single batch of data.
    
    Args:
      dataset: The dataset to load.
      batch_size: The number of images in the batch.
      height: The size of each image after preprocessing.
      width: The size of each image after preprocessing.
      is_training: Whether or not we're currently training or evaluating.
    
    Returns:
      images: A Tensor of size [batch_size, height, width, 3], image samples that have been preprocessed.
      images_raw: A Tensor of size [batch_size, height, width, 3], image samples that can be used for visualization.
      labels: A Tensor of size [batch_size], whose values range between 0 and dataset.num_classes.
    """
    	data_provider = slim.dataset_data_provider.DatasetDataProvider(dataset, common_queue_capacity=32,
        common_queue_min=8)
    	image_raw, label = data_provider.get(['image', 'label'])
    
    	# Preprocess image for usage by Inception.
    	image = inception_preprocessing.preprocess_image(image_raw, height, width, is_training=is_training)
    
    	# Preprocess the image for display purposes.
    	image_raw = tf.expand_dims(image_raw, 0)
    	image_raw = tf.image.resize_images(image_raw, [height, width])
    	image_raw = tf.squeeze(image_raw)

    	# Batch it up.
    	images, images_raw, labels = tf.train.batch(
          	[image, image_raw, label],
          	batch_size=batch_size,
          	num_threads=1,
          	capacity=2 * batch_size)
    
    	return images, images_raw, labels

当应用其他模型结构时，需要修改以下代码：

    image = inception_preprocessing.preprocess_image(image_raw, height, width, is_training=is_training)


#### 导入预训练模型并fine tuning（导入过程和<a href="#Per_train">导入预训练模型</a>一样）

    insulators_data_dir = '/workspace/JiaXuejian'

	def get_init_fn():
    """Returns a function run by the chief worker to warm-start the training."""
    	checkpoint_exclude_scopes=["InceptionV1/Logits", "InceptionV1/AuxLogits"]
    
    	exclusions = [scope.strip() for scope in checkpoint_exclude_scopes]

    	variables_to_restore = []
    	for var in slim.get_model_variables():
        	for exclusion in exclusions:
            	if var.op.name.startswith(exclusion):
                	break
        	else:
            	variables_to_restore.append(var)

    	return slim.assign_from_checkpoint_fn(os.path.join(checkpoints_dir, 'inception_v1.ckpt'),variables_to_restore)


	train_dir = '/tmp/inception_finetuned1/'

	with tf.Graph().as_default():
    	tf.logging.set_verbosity(tf.logging.INFO)
    
    	dataset = insulators.get_split('train', insulators_data_dir)
    	images, _, labels = load_batch(dataset, height=image_size, width=image_size)
    
    	# Create the model, use the default arg scope to configure the batch norm parameters.
    	with slim.arg_scope(inception.inception_v1_arg_scope()):
        	logits, _ = inception.inception_v1(images, num_classes=dataset.num_classes, is_training=True)
        
    	# Specify the loss function:
    	one_hot_labels = slim.one_hot_encoding(labels, dataset.num_classes)
    	slim.losses.softmax_cross_entropy(logits, one_hot_labels)
    	total_loss = slim.losses.get_total_loss()

    	# Create some summaries to visualize the training process:
    	tf.summary.scalar('losses/Total Loss', total_loss)
  
    	# Specify the optimizer and create the train op:
    	optimizer = tf.train.AdamOptimizer(learning_rate=0.01)
    	train_op = slim.learning.create_train_op(total_loss, optimizer)
    
    	# Run the training:
    	final_loss = slim.learning.train(
        	train_op,
        	logdir=train_dir,
        	init_fn=get_init_fn(),
        	number_of_steps=1000)
        
  
	print('Finished training. Last batch loss %f' % final_loss)

当应用其他模型时，需要修改以下代码：


    image_size = inception.inception_v1.default_image_size
	checkpoint_exclude_scopes=["InceptionV1/Logits", "InceptionV1/AuxLogits"]
	slim.assign_from_checkpoint_fn(
      os.path.join(checkpoints_dir, 'inception_v1.ckpt'),
      variables_to_restore)
	with slim.arg_scope(inception.inception_v1_arg_scope()):
        logits, _ = inception.inception_v1(images, num_classes=dataset.num_classes, is_training=True)

#### 取一些样本应用在Fine tuning后的模型上

    insulators_data_dir = '/workspace/JiaXuejian'
	image_size = inception.inception_v1.default_image_size
	batch_size = 3
	
	with tf.Graph().as_default():
    	tf.logging.set_verbosity(tf.logging.INFO)
    
    	dataset = insulators.get_split('train', insulators_data_dir)
    	images, images_raw, labels = load_batch(dataset, height=image_size, width=image_size)
    
    	# Create the model, use the default arg scope to configure the batch norm parameters.
    	with slim.arg_scope(inception.inception_v1_arg_scope()):
        	logits, _ = inception.inception_v1(images, num_classes=dataset.num_classes, is_training=True)

    	probabilities = tf.nn.softmax(logits)
    
    	checkpoint_path = tf.train.latest_checkpoint(train_dir)
    	init_fn = slim.assign_from_checkpoint_fn(
      		checkpoint_path,
      		slim.get_variables_to_restore())
    
    	with tf.Session() as sess:
        	with slim.queues.QueueRunners(sess):
            	sess.run(tf.initialize_local_variables())
            	init_fn(sess)
            	np_probabilities, np_images_raw, np_labels = sess.run([probabilities, images_raw, labels])
    
            	for i in range(batch_size): 
                	image = np_images_raw[i, :, :, :]
                	true_label = np_labels[i]
                	predicted_label = np.argmax(np_probabilities[i, :])
                	predicted_name = dataset.labels_to_names[predicted_label]
                	true_name = dataset.labels_to_names[true_label]
                
                	plt.figure()
                	plt.imshow(image.astype(np.uint8))
                	plt.title('Ground Truth: [%s], Prediction [%s]' % (true_name, predicted_name))
                	plt.axis('off')
                	plt.show()

当应用其他模型结构时，需要修改以下代码：

	image_size = inception.inception_v1.default_image_size
	with slim.arg_scope(inception.inception_v1_arg_scope()):
        logits, _ = inception.inception_v1(images, num_classes=dataset.num_classes, is_training=True)