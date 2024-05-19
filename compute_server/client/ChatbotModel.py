import os
import keras_nlp
import keras
import tensorflow as tf
import numpy as np

os.environ["KERAS_BACKEND"] = "tensorflow"
keras.mixed_precision.set_global_policy("mixed_float16")

seq_length = 2048


class ChatbotModel:
    """
    Class abstracting the 'article_10m' model.

    Attributes:
        model: the model object loaded into memory.
        tokenizer: the model's tokenizer.
        model_name: the name of the model.
    """

    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.model_name = 'chatbot_125m'

    @staticmethod
    def get_tensorflow_device():
        """
        Gets the devices tensorflow is using to run the model.

        :return: A list of all devices the model can run on.
        """
        return tf.config.list_physical_devices()

    def load_model(self):
        """
        Loads the model checkpoint.

        The checkpoint only contains weights, and therefore the model needs to be built beforehand. In this case the model is downloaded.
        """
        self.model = keras_nlp.models.OPTCausalLM.from_preset("opt_125m_en", preprocessor=None)
        self.model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                           optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
                           metrics=['accuracy'])

        self.model.load_weights('checkpoint.weights.h5')

    def load_tokenizer(self):
        """
        Loads the pretrained tokenizer.
        """
        self.tokenizer = keras_nlp.models.OPTTokenizer.from_preset("opt_125m_en")

    def compute(self, prompt, temp=0.5, top_p=0.5):
        """
        Generates a single token given a context.

        :param prompt: the context used to generate to token.
        :param temp: the sampler's temperature.
        :param top_p: the sampler's top_p value.
        :return: a single token sampled from the model's output distribution.
        """
        if self.model is None:
            raise Exception('Tried computing with a model that hasn\'t been loaded yet!')

        if self.tokenizer is None:
            raise Exception('Tried using a tokenizer that hasn\'t been loaded yet!')

        window = [0] + tokenizer(prompt).numpy().tolist()[:seq_length - 1]

        padding_mask = [1] * len(window)

        model_input = {
            'token_ids': np.array([window]),
            'padding_mask': np.array([padding_mask])
        }

        pred = model(model_input, training=False)
        pred = tf.squeeze(pred[0, -1, :])  # only consider next position predictions
        pred /= temp

        pred = top_p_function(pred, top_p)

        token = np.argmax(np.random.multinomial(1, pred))
        decoded_token = tokenizer.detokenize([token]).numpy().decode("utf-8")

        return decoded_token

    @staticmethod
    def top_p_function(logits, threshold):
        """
        Applies the top_p function to a vector of logits. Returns the normalized probability of the most probable token's whos combined probability doesn't exceed threshold.

        :param logits: the logits to apply the top_p function to.
        :param threshold: the top_p value.
        :return: the probability distribution of the logits after applying the top_p function.
        """
        # type must be float64 due to softmax function rounding errors
        logits = tf.cast(logits, dtype=tf.float64)
        probs = tf.nn.softmax(logits)

        # sort and calculate cumulative sums
        sorted_probs, sorted_indices = tf.math.top_k(probs, k=tf.shape(probs)[-1], sorted=True)
        cum_probs = tf.cumsum(sorted_probs, axis=-1, exclusive=True)

        # mask out tokens outside the top_p threshold
        mask = tf.cast(cum_probs <= threshold, dtype=tf.float64)
        masked_probs = sorted_probs * mask
        renorm_probs = masked_probs / tf.reduce_sum(masked_probs, axis=-1, keepdims=True)

        # create new probabilities
        new_probs = tf.scatter_nd(indices=tf.expand_dims(sorted_indices, axis=-1),
                                  updates=renorm_probs,
                                  shape=tf.shape(probs))
        return new_probs
