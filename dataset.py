from tensorflow.keras.datasets import mnist
import tensorflow as tf
from abc import abstractmethod, ABC

class MNISTDataset(ABC):
    def __init__(self, batch_size, overfit_batch=False):
        x, y = self.load_raw_data()
        dset = tf.data.Dataset.from_tensor_slices((x, y))
        if overfit_batch:
            dset = dset.take(batch_size)
        dset = dset.map(self.preprocess_example)
        if not overfit_batch:
            dset = dset.map(self.augment_example)
        dset = dset.shuffle(10000)
        dset = dset.batch(batch_size)
        self.dset = dset 
        self.iterator = self.dset.make_initializable_iterator()
        self.next_batch = self.iterator.get_next()

    def init(self, sess):
        sess.run(self.iterator.initializer)

    @abstractmethod
    def load_raw_data(self):
        pass

    def preprocess_example(self, *example):
        # Convert to float
        new_image = tf.image.convert_image_dtype(tf.reshape(example[0], [28, 28, 1]),
                                                 tf.float32)
        # Scale to [0, 1) 
        new_image = new_image / 255.
        new_label = tf.cast(tf.one_hot(example[1], 10), tf.float32)
        return new_image, new_label

    def augment_example(self, *example):
        new_image = example[0]
        rotation = 20
        crop = 0.1

        transforms = []
        with tf.name_scope('augmentation'):
            shp = tf.shape(new_image)
            height, width = shp[0], shp[1]
            width = tf.cast(width, tf.float32)
            height = tf.cast(height, tf.float32)

            identity = tf.constant([1, 0, 0, 0, 1, 0, 0, 0], dtype=tf.float32)

            if rotation > 0:
                angle_rad = rotation * 3.141592653589793 / 180.0
                angles = tf.random_uniform([], -angle_rad, angle_rad)
                f = tf.contrib.image.angles_to_projective_transforms(angles,
                                                                     height, width)
                new_image = tf.contrib.image.transform(new_image, f)

            if crop < 1:
                crop_value = tf.random_uniform([], crop, 1.0)
                crop_size = tf.floor(28 * crop_value)
                cropped = tf.random_crop(new_image, [crop_size, crop_size, 1])
                new_image = tf.image.resize_images(tf.expand_dims(cropped, 0), [28, 28])[0]

        return new_image, example[1]

class MNISTTrain(MNISTDataset):
    def load_raw_data(self):
        x, y = mnist.load_data()[0]
        return x, y

class MNISTTest(MNISTDataset):
    def load_raw_data(self):
        x, y = mnist.load_data()[1]
        return x, y