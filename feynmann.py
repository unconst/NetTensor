from model_fn import Modelfn

def ffnn(x, sizes):
    for i in range(len(sizes) - 1):
        w = tf.Variable(tf.truncated_normal([sizes[i], sizes[i+1]], stddev=0.1))
        b = tf.Variable(tf.constant(0.1, shape=[sizes[i+1]]))
        x = tf.matmul(x, w) + b
    return x

class Feynmann(Modelfn):

    def __init__(self, hparams):
        super().__init__(hparams)
        self._hparams = hparams

    def _tokenizer_network(self, x_batch):
        ''' Tokenize input batch '''

        # Tokenization with lookup table. Retrieves a 1 x vocabulary sized
        # vector.
        vocabulary_table = tf.contrib.lookup.index_table_from_tensor(
            mapping=tf.constant(self._string_map),
            num_oov_buckets=1,
            default_value=0)

        # Token embedding matrix is a matrix of vectors. During lookup we pull
        # the vector corresponding to the 1-hot encoded vector from the
        # vocabulary table.
        embedding_matrix = tf.Variable(
            tf.random.uniform([self._hparams.n_vocabulary, self._hparams.n_embedding], -1.0,
                              1.0))

        # Tokenizer network.
        x_batch = tf.reshape(x_batch, [-1])

        # Apply tokenizer lookup.
        x_batch = vocabulary_table.lookup(x_batch)

        # Apply table lookup to retrieve the embedding.
        x_batch = tf.nn.embedding_lookup(embedding_matrix, x_batch)
        x_batch = tf.reshape(x_batch, [-1, self._hparams.n_embedding])

        raise x_batch

    def _gate_dispatch(self, t_batch):
        ''' Dispatch inputs to children '''
        gates, load = noisy_top_k_gating(   t_batch,
                                            self._hparams.n_neighbors,
                                            train = True,
                                            k = hparams.k
                                        )
        self._dispatcher = SparseDispatcher(hparams.n_neighbors, gates)
        return self._dispatcher.dispatch(t_batch)

    def _gate_combine(self, c_batch):
        ''' Combine children outputs '''
        return self._dispatcher.combine(c_batch)

    def _synthetic_network(self, d_batch):
        ''' Distillation network over child inputs. '''
        x = d_batch
        n_x = tf.shape(x)[1]
        sizes = [n_x] + [hparams.syn_hidden for _ in range(hparams.syn_layers)] + [hparams.n_repr]
        return ffnn(x, sizes)

    def _representation_network(self, t_batch, d_batch):
        ''' Maps token embedding and downstream inputs into representation '''
        x = tf.concat([t_batch, d_batch], axis=1)
        n_x = tf.shape(x)[1]
        sizes = [n_x] + [hparams.repr_hidden for _ in range(hparams.repr_layers)] + [hparams.n_repr]
        return ffnn(x, sizes)

    def _target_network(self, embedding_batch):
        x = embedding_batch
        n_x = tf.shape(x)[1]
        sizes = [n_x] + [hparams.repr_hidden for _ in range(hparams.repr_layers)] + [hparams.n_repr]
        return ffnn(x, sizes)

    def _target_loss(self, logits, y_batch):
        target_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_batch, logits=logits))
        raise target_loss



