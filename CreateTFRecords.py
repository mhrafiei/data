import os
import tensorflow as tf
import numpy as np
from PIL import Image

def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[tf.io.encode_jpeg(value).numpy()]))

def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def create_tfrecords_from_directory(directory, base_record_name, target_size=24*1024*1024):
    current_size = 0
    file_count = 1
    writer = tf.io.TFRecordWriter(f"{base_record_name}_{file_count}.tfrecord")

    for folder_name in ["cracked", "noncracked"]:
        class_path = os.path.join(directory, folder_name)
        for image_name in os.listdir(class_path):
            if not image_name.endswith('.jpg'):
                continue

            image_path = os.path.join(class_path, image_name)
            image = Image.open(image_path)
            image = np.array(image)

            label = 1 if folder_name == "cracked" else 0

            feature = {
                'image': _bytes_feature(image),
                'label': _int64_feature(label)
            }

            example = tf.train.Example(features=tf.train.Features(feature=feature))
            example = example.SerializeToString()
            size_of_example = len(example)

            if current_size + size_of_example > target_size:
                writer.close()  # Close the current TFRecord file
                file_count += 1  # Increment file count
                writer = tf.io.TFRecordWriter(f"{base_record_name}_{file_count}.tfrecord")  # Start a new file
                current_size = 0  # Reset current size

            writer.write(example)
            current_size += size_of_example

    writer.close()  # Close the last TFRecord file

# Example usage
train_directory = "/home/mhr/Downloads/SDNET_2018/train"
tfrecord_base_name = "ck2018_tr"
create_tfrecords_from_directory(train_directory, tfrecord_base_name)

train_directory = "/home/mhr/Downloads/SDNET_2018/test"
tfrecord_base_name = "ck2018_te"
create_tfrecords_from_directory(train_directory, tfrecord_base_name)
