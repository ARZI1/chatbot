import json


base_vocab = 'abcdefghijklmnopqrstuvwxyz!"#$£%&\'()*,-./0123456789:;<>?[\\]{}\n '
MERGES_PER_EPOCH = 15


class Tokenizer:
  """
  An implementation of a BPE (Byte-Pair-Encoding) tokenizer.

  The tokenizer allows us to create a token vocabulary size of our choosing while preventing Out-Of-Vocabulary problems. This is done by starting with a base vocabulary comprised of basic characters. The tokenizer then goes over the training corpus, finds the most frequent token pair, and merges them into a new token. This process is done until our desired vocabulary size is reached.

  Attributes:
    vocab_size (int): The maximum of the tokenizer's vocabulary.
    oov_token (str): The predefined Out-Of-Vocabulary token.
    tuple_delimiter (char): The delimiter used to seperate tuple elements when
      serializing the tokenizer.
    vocab (str): the tokenizer's vocabulary.
    reserved_tokens (list[str]): Tokens reserved for other purposes.
    merges (dict[tuple[str], str]): The token merges learned turing training.
  """

  def __init__(self, vocab_size: int, pad_token: str = '<pad>', oov_token: str = '<oov>', reserved_tokens: list[str] = list()) -> None:
    """
    Tokenizer constructor.

    Initializes the tokenizer's attributes and loads the base vocabulary. Training
    the tokenizer is done by calling the ```train_on_corpus``` function, not by
    calling the constructor.

    Args:
      vocab_size: The maximum size of the tokenizer's vocabulary.
      oov_token: The predefined Out-Of-Vocabulary token.
      reserved_tokens: Predefined tokens such as padding, start and end.
    """
    self.vocab_size = vocab_size
    self.oov_token = oov_token
    self.tuple_delimiter = chr(0x1d)

    self.vocab = [pad_token] + [oov_token]
    for token in reserved_tokens:
      self.vocab.append(token)

    self.vocab += list(base_vocab)
    self.vocab_set = set(self.vocab)

    reserved_tokens.append(pad_token)
    reserved_tokens.append(oov_token)
    self.reserved_tokens = reserved_tokens

    self.create_lookup_maps()

    self.merges = dict()


  def tokenize(self, input: str) -> list[int]:
    """
    Tokenizes a string using the tokenizer's learned merge table.

    The input is first split up into base tokens found in the ```base_vocab``` vocabulary. A loop then iterates over the tokens and checks whether the concatenation of adjacent pairs is in the vocabulary. If it is, the tokens are merged. Once the tokenizer can no longer merge tokens, the tokens are vectorized and returned.

    Args:
      input: The string we want to tokenize.

    Returns:
      A list containing the tokenized input in vector form.
    """

    split = [c for c in input if c in self.vocab_set]

    flag = True
    while flag:
      flag = False
      i = 0
      while len(split) - 1 > i:
        merge = split[i] + split[i + 1]
        if merge in self.vocab_set:
          split[i:i+2] = [merge]
          flag = True
        else:
          i += 1

    vectorized = list(map(lambda t: self.token_to_index[t], split))

    return vectorized


  def detokenize(self, input: list[int]) -> str:
    """
    Detokenizes a list of tokens in vector form into a string.

    Each token vector is mapped to its corresponding token. The tokens are then joined together to form the final string.

    Args:
      input: The list of token vectors we want to detokenize.

    Returns:
      A string derived from the input tokens.
    """
    # map each token vector, the index, to its string form
    devectorized = list(map(lambda i: self.index_to_token[i], input))

    return ''.join(devectorized)


  def train_on_corpus(self, corpus: list[str], verbose=False) -> None:
    """
    Trains the tokenizer on a corpus.

    The corpus is first normalized, a process in which characters not present in the base vocabulary are removed. It is then pretokenized, meaning the corpus is split up into words and their frequencies to reduce computation later. After that, the words are split up into their base token representations. Finally we search for the most frequent token pair, merge them into a new token, add it to the vocabulary and merges dictionary, and update our word splits. This is done until our desired vocabulary size is reached.

    In order to reduce training time a number of pairs are merges in each "epoch". These pairs are picked by sorting the pair frequencies by highest lowest.

    Args:
      corpus: The corpus we want to train on. It's recommended to preprocess the
        text before tokenizer training, but not necessary.
    """
    # normalization & pretokenization
    word_freq = dict()
    for e in corpus:
      for word in self.normalize_and_split(e):
        word_freq[word] = word_freq.get(word, 0) + 1

    # initial word split
    splits = { word:[c for c in word] for word in word_freq.keys() }

    # merging
    while len(self.vocab) < self.vocab_size:
      pair_freq = self.calculate_pair_freq(word_freq, splits)
      best_merges = sorted(pair_freq.items(), key=lambda x: -x[1])

      left = self.vocab_size - len(self.vocab)
      for i in range(min(MERGES_PER_EPOCH, left)):
        new_merge = best_merges[i][0]
        self.merges[new_merge] = new_merge[0] + new_merge[1]
        self.vocab.append(new_merge[0] + new_merge[1])
        self.vocab_set.add(new_merge[0] + new_merge[1])
        self.merge_pairs(new_merge[0], new_merge[1], splits)

      if verbose:
        percent = len(self.vocab) / self.vocab_size * 100
        print(f'\rCurrent vocab: {len(self.vocab)}/{self.vocab_size} {percent:.2f}%', end='  ')

    if verbose:
      print()

    self.create_lookup_maps()


  def normalize_and_split(self, text: str) -> list[str]:
    """
    Normalizes a text and splits it by whitespace.

    Characters not found in the tokenizer's base vocabulary are filtered out. The text is then seperated into individual words by whitespace while keeping the delimiter.

    Args:
      text: The string we want to normalize and split.
    """
    text = ''.join([c for c in text if c in base_vocab])
    words = [' ' + word for word in text.split(' ') if word]
    if len(words) > 0:
      words[0] = words[0][1:]  # remove space from first element

    return words


  def calculate_pair_freq(self, word_freq: dict[str, int], splits: dict[str, list[str]]) -> dict[tuple[str, str], int]:
    """
    Calculates all pair frequencies in a pretokenized corpus.

    All token pairs of the pretokenized words are iterated over and their frequency tracked.

    Args:
      word_freq: The pretokenized words with their frequencies.
      splits: The tokens comprising the pretokenized words.

    Returns:
      A dictionary of all token pairs with their respective frequencies.
    """
    pair_freq = dict()
    for word, freq in word_freq.items():
      split = splits[word]
      for pair in zip(split, split[1:]):  # zip adjacent tokens
        pair_freq[pair] = pair_freq.get(pair, 0) + freq

    return pair_freq


  def merge_pairs(self, t1: str, t2: str, splits: dict[str, list[str]]) -> None:
    """
    Merges tokens comprising pretokenized words.

    All token pairs in the pretokenized words are compared against the merge tokens. If a match is found the tokens are compined and replaced with the merged token.

    Args:
      t1: The first token to merge.
      t2: The seconds token to merge.
      splits: The pretokenized words with their token splits.
    """
    for word, split in splits.items():
      i = 0
      while len(split) - 1 > i:
        if split[i] == t1 and split[i + 1] == t2:
          split[i:i+2] = [t1 + t2]
        else:
          i += 1

      splits[word] = split


  def create_lookup_maps(self) -> None:
    """
    Creates mappings for token to index and index to token.

    The vocabulary is enumerated over and the appropriate key and values are placed in a map.
    """
    self.token_to_index = { token:i for i, token in enumerate(self.vocab) }
    self.index_to_token = { i:token for i, token in enumerate(self.vocab) }


  def serialize_to_file(self, file_path: str) -> None:
    """
    Serializes the tokenizer and saves it to a file.

    The tokenizer's attributes are turned into JSON form and then saved to an file. The merges map contains a tuple type which cannot be serialized to JSON. Therefore, we place the tupple elements in a string and separate them with the ```self.tuple_delimiter``` char.

    Args:
      file_path: The file path for saving the serialized file.
    """
    data = dict()
    data['vocab_size'] = self.vocab_size
    data['oov_token'] = self.oov_token
    data['vocab'] = self.vocab
    data['reserved_tokens'] = self.reserved_tokens
    serialized_merges = { k[0]+self.tuple_delimiter+k[1] : v for k, v in self.merges.items() }
    data['merges'] = serialized_merges

    with open(file_path, 'w') as f:
      json.dump(data, f)


  def deserialize_from_file(self, file_path: str) -> None:
    """
    Deserializes the tokenizer from a serialized JSON file.

    The JSON file containing the serialized tokenizer is read from and the tokenizer's attributes are set. The tuples in the merges map are represented as a string with their elements separated by a delimiter. During deserialization we split this string by the delimiter and recreate the tupple.
    """
    with open(file_path, 'r') as f:
      data = json.load(f)
      self.vocab_size = data['vocab_size']
      self.oov_token = data['oov_token']
      self.vocab = data['vocab']
      self.vocab_set = set(self.vocab)
      self.reserved_tokens = data['reserved_tokens']
      serialized_merges = data['merges']
      self.merges = dict()
      for k, v in serialized_merges.items():
        delimiter_idx = k.index(self.tuple_delimiter)
        key = (k[:delimiter_idx], k[delimiter_idx+1:])
        self.merges[key] = v

    self.create_lookup_maps()

  def get_vocab(self):
    return self.vocab