import tensorflow as tf

tensorflow_version = tf.__version__
gpu_available = tf.config.list_physical_devices('GPU')

a = tf.constant([1.0, 2.0], name="a")
b = tf.constant([1.0, 2.0], name="b")
result = tf.add(a, b, name="add")
print(result)

print("tensorflow version:", tensorflow_version, "\tGPU available:", gpu_available)

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

features = tf.constant([12,23,10,17])
labels = tf.constant([0, 1, 1, 0])
dataset = tf.data.Dataset.from_tensor_slices((features, labels))
print(dataset)
for element in dataset:
    print(element)
