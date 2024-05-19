import numpy as np
import tensorflow as tf


def scaled_dot_product_attention(Q, K, V, mask):
  """
  Implementation of scaled dot product attention.

  The shape of the queries, values and keys is not known. We only care about
  the last two dimensions which must be of shape (tokens, d_k) for the queries
  and keys, or (tokens, d_v) for the values. This allows us to use the function
  both for normal attention and for multi head attention. Additionally, this
  means the function doesn't need batch dimension boilerplate.

  Args:
    Q: the queries. (..., seq_length, d_k)
    K: the keys. (..., seq_length, d_k)
    V: the values. (..., seq_length, d_v)
    mask: a boolean mask of elements not to attend to. (batches, heads, seq_length, seq_length)

  Returns:
    The scaled dot product attention of the queries, keys and values provided. If
    the input was comprised of multple batches and heads the result will retain
    the number of batches and heads. (..., seq_length, d_v)
  """
  # matmul transposition uses `tf.linalg.matrix_transpose` which only transposes
  # the inner-most 2d matrix
  x = tf.matmul(Q, K, transpose_b=True) # (..., seq_length, seq_length)

  # scale the dot product with of a factor of 1 over the standard deviation,
  # therefore the standard deviation is now unit
  d_k = Q.shape[-1]
  x /= d_k**0.5

  # apply look-ahead and padding masks, ones represent values we want to keep
  # x += (1. - mask) * -1e9
  replace_with = tf.expand_dims(tf.ones(x.shape[-1:], dtype=tf.float32) * -1e9, axis=0)
  x = tf.where(mask, x, replace_with)

  # apply the softmax functin across each token, horizontally
  x = tf.nn.softmax(x, axis=-1)

  x = tf.matmul(x, V) # (..., seq_length, d_v)
  return x


class Multi_Head_Attention(tf.keras.layers.Layer):
   """
  Implementation of multi head attention. This model is identical to the one proposed 
  in the "Attention is all you need" paper.

  Attributes:
    supports_masking: tells tensorflow that this layer supports masking.
    d_model: the model's embedding dimension size.
    heads: the number of attention heads to be used.
    d_k: the size of keys and queries vectors.
    d_v: the size of the value vector.
    Q_w: the projection metrix for calculating queries, consists of learned parameters.
    K_w: the projection metrix for calculating keys, consists of learned parameters.
    V_w: the projection metrix for calculating values, consists of learned parameters.
    Q_w: the projection metrix for concatenating the attention heads, consists of learned parameters.
    dropout: the dropout layer.
  """

  def __init__(self, heads, d_model, dropout_rate):
    """
    Creates the Multi_Head_Attentin object.

    Args:
      heads: the number of attention heads we want to use.
      d_model: the model's embedding dimension.
      dropout_rate: the attention mechinism's dropout rate.
    """
    super().__init__()
    self.supports_masking = True

    if d_model % heads != 0:
      raise ValueError('Model embedding dimensions must be divisible by the number of attention heads!')

    self.d_model = d_model
    self.heads = heads

    # in this case d_k and d_v are equal, however that's just for simplicity's sake
    self.d_k = d_model // heads
    self.d_v = d_model // heads

    # Xavier weight initialization
    seed = np.random.randint(1e9, size=1)[0]
    initializer = tf.keras.initializers.GlorotNormal(seed)
    # we need to add a 1 for the batch dimension, tensorflow's matmul doesn't do this for us
    self.Q_w = tf.Variable(initializer(shape=(1, heads, d_model, self.d_k)))
    self.K_w = tf.Variable(initializer(shape=(1, heads, d_model, self.d_k)))
    self.V_w = tf.Variable(initializer(shape=(1, heads, d_model, self.d_v)))
    self.O_w = tf.Variable(initializer(shape=(1, heads * self.d_v, d_model)))

    self.dropout = tf.keras.layers.Dropout(dropout_rate)


  def call(self, inputs, mask=None):
    # inputs (batches, tokens, d_model)

    # add heads dimension to inputs
    inputs = tf.expand_dims(inputs, axis=1) # (batches, 1, tokens, d_model)

    Q = tf.matmul(inputs, self.Q_w) # (batches, heads, tokens, d_k)
    K = tf.matmul(inputs, self.K_w) # (batches, heads, tokens, d_k)
    V = tf.matmul(inputs, self.V_w) # (batches, heads, tokens, d_v)

    seq_len = inputs.shape[-2]
    full_mask = tf.cast(np.tri(seq_len), dtype=tf.bool)
    if mask is not None:
      mask = tf.expand_dims(mask, axis=1) # (batches, 1, seq_len)
      mask = tf.expand_dims(mask, axis=1) # (batches, 1, 1, seq_len)
      mask = tf.tile(mask, tf.constant([1, 1, seq_len, 1])) # (batches, heads, seq_len, seq_len)
      full_mask = full_mask & mask # (batches, heads, seq_len, seq_len)

    x = scaled_dot_product_attention(Q, K, V, full_mask) # (batches, heads, tokens, d_v)

    # concatenate heads
    batches, heads, tokens, d_v = x.shape
    # batches could be None, we therefore pass -1 so that the batch size is calculated at runtime
    x = tf.transpose(x, perm=[0, 2, 1, 3])
    x = tf.reshape(x, (-1, tokens, heads*d_v)) # (batches, tokens, d_model)

    x = self.dropout(x)

    x = tf.matmul(x, self.O_w) # (batches, tokens, d_model)

    return x


class FeedForwardNetwork(tf.keras.layers.Layer):
  """
  A simple implementation of the feedforward layer found in the "Attention is all you need" 
  paper. The paper suggests implementing the layer using a convelution layer with a kernel 
  of one, however this implemention is based on element-wise dense layers.

  Attributes:
    supports_masking: tells tensorflow that this layer supports masks.
    dense1: the first dense layer in the FFN.
    dense2: the second dense layer in the FFN
    dropout: the dropout layer used to prevent overfitting.
  """

  def __init__(self, dff, d_model, dropout_rate):
    """
    Creates the FeedForwardNetwork object.

    Args:
      dff: the size of the hidden dense layer.
      d_mode: the embedding dimension of the model.
      dropout_rate: the dropout rate used.
    """
    super().__init__()
    self.supports_masking = True

    self.dense1 = tf.keras.layers.Dense(dff, activation='relu')
    self.dense2 = tf.keras.layers.Dense(d_model)
    self.dropout = tf.keras.layers.Dropout(dropout_rate)


  def call(self, inputs):
    """
    Applies the feed forward network. Called during forward propagation and inference.

    Args:
      inputs: the layer's inputs, must of of shape (batches, seq_length, d_model)
    """
    x = self.dense1(inputs)
    x = self.dense2(x)
    x = self.dropout(x)

    return x


class TransformerBlock(tf.keras.layers.Layer):
"""
Implementation of the full transformer block, consisting of attention mechanisms and a feed 
forward network along with layer normalization and residual connections.

Attributes:
  supports_masking: tells tensorflow that this layer supports masks.
  layer_norm_1: the first normalization layer.
  residual_connection: the residual connection component. The same object is used twice as 
    it doesn't have learnable parameters.
  layer_norm_2: the second normalization layer.
  feed_forward_network: the trasnformer's feed forward network.
"""

  def __init__(self, attention_heads, dff, d_model, dropout_rate):
    """
    Creates a transformer object.

    Args:
      attention_heads: the desired number of heads in the attention mechanism.
      dff: the size of the hidden layer in the feed forward network.
      dropout_rate: the dropout rate for the different parts of the transformer.
    """
    super().__init__()
    self.supports_masking = True

    self.layer_norm_1 = tf.keras.layers.LayerNormalization()
    self.multi_head_attention = Multi_Head_Attention(attention_heads, d_model, dropout_rate)
    self.residual_connection = tf.keras.layers.Add()
    self.layer_norm_2 = tf.keras.layers.LayerNormalization()
    self.feed_forward_network = FeedForwardNetwork(dff, d_model, dropout_rate)


  def call(self, inputs):
    """
    Applies the trasnformer logic to an input. Used during forward propagation and inference.

    Args:
      inputs: the input matrix, must be of shape (batches, seq_length, d_model)
    """
    norm_1_out = self.layer_norm_1(inputs)
    attention_out = self.multi_head_attention(norm_1_out)
    residual_1_out = self.residual_connection([norm_1_out, attention_out])

    norm_2_out = self.layer_norm_2(residual_1_out)
    ffn_out = self.feed_forward_network(norm_2_out)
    residual_2_out = self.residual_connection([norm_2_out, ffn_out])

    return residual_2_out
