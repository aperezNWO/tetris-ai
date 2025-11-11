# logs.py
import tensorflow as tf

class CustomTensorBoard:
    def __init__(self, log_dir):
        self.writer = tf.summary.create_file_writer(log_dir)

    def log(self, step, **metrics):
        """Log multiple metrics at once.
        
        Args:
            step: The timestep or episode number
            **metrics: Keyword arguments of metric names and values
                      e.g., log(100, avg_score=42.5, min_score=10)
        """
        with self.writer.as_default():
            for name, value in metrics.items():
                tf.summary.scalar(name, value, step=step)
        self.writer.flush()  # Ensure data is written immediately

    def log_scalar(self, tag, value, step):
        """Log a single scalar value."""
        with self.writer.as_default():
            tf.summary.scalar(tag, value, step=step)
        self.writer.flush()

    def log_histogram(self, tag, values, step):
        """Log a histogram."""
        with self.writer.as_default():
            tf.summary.histogram(tag, values, step=step)
        self.writer.flush()