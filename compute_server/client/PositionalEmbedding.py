import numpy as np
import tensorflow as tf

def positional_encoding(length, depth):
  """
  Creates the positional encoding matrix from  the sinusoidal encoding formulas.

  Args:
    length: number of possible sequence positions.
    depth: the depth of the encoding, equal to the token embedding dimension.
  """
  depth = depth / 2

  positions = np.arange(length)[:, np.newaxis] # (seq, 1)
  depths = np.arange(depth)[np.newaxis, :] / depth # (1, depth)

  angle_rates = 1 / (10000**depths) # (1, depth)
  angle_rads = positions * angle_rates # (pos, depth)

  pos_encoding = np.concatenate(
      [np.sin(angle_rads), np.cos(angle_rads)],
      axis=-1)

  pos_encoding = np.expand_dims(pos_encoding, axis=0)

  return tf.cast(pos_encoding, dtype=tf.float32)


class PositionalEmbedding(tf.keras.layers.Layer):
  """
  Implementation of a positional embedding layer based on the sinusoidal encoding formulas from the "Attention is all you need" paper.

  Attributes:
    supports_masking: tells tensorflow that this layer supports masking.
    pe_table: the table containing the positional encoding values.
  """

  def __init__(self, max_length, d_model):
    """
    Creates a PostionalEmbedding object.

    Args:
      max_length: the maximum sequence length.
      d_model: the token embedding dimension.
    """
    super().__init__()
    self.supports_masking = True

    self.pe_table = positional_encoding(max_length, d_model)


  def call(self, inputs):
    """
    Adds the positional embedding values to an input. Called during forward propagatin and inference.

    Args:
      input: the input matrix, must be of shape (batches, seq_length, d_model)
    """
    batches, seq_length, d_model = inputs.shape

    # scale up token embeddings so that they have more influence than positional embedding
    x = inputs * np.sqrt(d_model, dtype=np.float32)

    x = x + 0.1 * self.pe_table[:, :seq_length, :]

    return x
