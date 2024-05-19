import tensorflow as tf
import numpy as np
from Transformer import TransformerBlock, Multi_Head_Attention, FeedForwardNetwork
from PositionalEmbedding import PositionalEmbedding
from Tokenizer import Tokenizer


# CHECKPOINT_DIR = 'C:/Users/Arzi0/PycharmProjects/chatbot_website/client/trained_checkpoint/checkpoint.ckpt'
CHECKPOINT_DIR = 'E:/language model/article_big/checkpoint.ckpt'
TOKENIZER_DIR = 'C:/Users/Arzi0/PycharmProjects/chatbot_website/client/tokenizer_2500_vocab.json'
# CHECKPOINT_DIR = '/checkpoints/article_checkpoint.ckpt'
# TOKENIZER_DIR = '/tokenizers/tokenizer_2500_vocab.json'

PAD_TOKEN = '<pad>'
START_TOKEN = '<s>'
END_TOKEN = '</s>'

# model hyperparamaters
BATCH_SIZE = 64
BUFFER_SIZE = 50_000
train_split = 0.8
validation_split = 0.1
test_split = 0.1
vocab_size = 2500
transformer_units = 6
d_model = 384
seq_length = 256
attention_heads = 8
dff = 1024 # transformer ffn hidden layer size
dropout_rate = 0.1
learning_rate = 1e-6


class ArticleModel:
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
        self.model_name = 'article_10m'

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

        Because the checkpoint only contains the weights of the model, it first needs to be built.
        """
        inputs = tf.keras.Input(shape=(seq_length,))
        # token embedding layer
        x = tf.keras.layers.Embedding(vocab_size, d_model, mask_zero=True)(inputs)
        # positonal embedding layer
        x = PositionalEmbedding(5000, d_model)(x)

        # stack transformers on top of each other
        for _ in range(transformer_units):
            x = TransformerBlock(attention_heads, dff, d_model, dropout_rate)(x)

        x = tf.keras.layers.LayerNormalization()(x)

        outputs = tf.keras.layers.Dense(vocab_size)(x)

        model = tf.keras.Model(inputs=inputs, outputs=outputs)

        model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                      optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=learning_rate),
                      metrics=['accuracy'])

        model.load_weights(CHECKPOINT_DIR)

        self.model = model

    def load_tokenizer(self):
        """
        Loads the pretrained tokenizer.
        """
        tokenizer = Tokenizer(2500)
        tokenizer.deserialize_from_file(TOKENIZER_DIR)

        self.tokenizer = tokenizer

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

        tokenized = [2] + self.tokenizer.tokenize(prompt)
        tokenized = tokenized[-256:] # use the last 256 tokens
        index = len(tokenized) - 1
        window = tokenized + [0] * (seq_length - len(tokenized))

        pred = self.model(np.array([window]))
        pred = np.squeeze(pred)[index].astype('float64')
        pred /= float(temp)

        pred = self.top_p_function(pred, top_p)

        # sample a token using the multinomial function
        token = np.argmax(np.random.multinomial(1, pred))
        decoded_token = self.tokenizer.detokenize([token])

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


