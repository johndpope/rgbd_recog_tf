import tensorflow as tf
import numpy as np

def batch_input(dataset, batch_size, train, num_preprocess_threads=None, num_readers=1):
    """Contruct batches of training or evaluation examples from the image dataset.

    Args:
        dataset: instance of Dataset class specifying the dataset.
            See dataset.py for details.
        batch_size: integer
        train: boolean
        num_preprocess_threads: integer, total number of preprocessing threads
        num_readers: integer, number of parallel readers

    Returns:
        images: 4-D float Tensor of a batch of images
        labels: 1-D integer Tensor of [batch_size].
    
    Raises:
        ValueError: if data is not found
    """
    with tf.name_scope('batch_processing'):
        data_files = dataset.data_files() #TODO: data_files = list of files
        if data_files is None:
            raise ValueError('No data files found for this dataset')

        # Create filename_queue
        if train:
            filename_queue = tf.train.string_input_producer(data_files, shuffle=True, capacity=16)
        else:
            filename_queue = tf.train.string_input_producer(data_files, shuffle=False, capacity=1)

        if num_preprocess_threads is None:
            num_preprocess_threads = FLAGS.num_preprocess_threads
        if num_preprocess_threads % 4:
            raise ValueError('Please make num_preprocess_threads a multiple of 4 (%d % 4 != 0)', num_preprocess_threads)

        if num_readers is None:
            num_readers = FLAGS.num_readers
        if num_readers < 1:
            raise ValueError('Please make num_readers at least 1')

        # Approximate number of examples per shard
        examples_per_shard = 1024
        min_queue_examples = examples_per_shard * input_queue_memory_factor
        if train:
            examples_queue = tf.RandomShuffleQueue(
                    capacity=min_queue_examples + 3*batch_size, 
                    min_after_dequeue=min_queue_examples, 
                    dtypes=[tf.string])
        else:
            examples_queue = tf.FIFOQueue(
                    capacity=examples_per_shard + 3*batch_size,
                    dtypes=[tf.string])

        # Create multiple readers to populate the queue of examples
        if num_readers > 1:
            enqueue_ops = []
            for _ in range(num_readers):
                reader = dataset.reader() # TODO
                _, value = reader.read(filename_queue) #TODO
                enqueue_ops.append(examples_queue.enqueue([value]))

            tf.train.queue_runner.add_queue_runner(tf.train.queue_runner.QueueRunner(examples_queue, enqueue_ops))
            example_serialized = examples_queue.dequeue()
        else:
            reader = dataset.reader()
            _, example_serialized = reader.read(filename_queue)

        images_and_labels = []
        for thread_id in range(num_preprocess_threads):
            # Parse a serialized Example proto to extract the image and metadata
            image_buffer, label_index, bbox = parse_example_proto(example_serialized) #TODO
            image = image_preprocessing(image_buffer, bbox, train, thread_id) #TODO
            images_and_labels.appen([image, label_index])

        images, label_index_batch = tf.train.batch_join(
                images_and_labels,
                batch_size=batch_size,
                capacity=2*num_preprocess_threads*batch_size)

        # TODO
        # Reshape images into these desired dimensions
        height, width = FLAGS.image_size, FLAGS.image_size
        depth = 3

        images = tf.cast(images, tf.float32)
        images = tf.reshape(images, shape=[batch_size, height, width, depth])

        # Display the training images in the visualizer
        tf.image_summary('images', images)
        return images, tf.reshape(label_index_batch, [batch_size])



batch_size = 32
image_size = 227
num_preprocess_threads = 4
num_readers = 4
input_queue_memory_factor = 16
'''
N = 20
lbl_lst = np.arange(1,N+1) 
dat_lst = np.array([chr(i+ord('a')-1) for i in lbl_lst])
print lbl_lst, dat_lst
'''
