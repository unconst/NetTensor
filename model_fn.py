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

        # Build corpus.
        f = zipfile.ZipFile(self._hparams.corpus_path)
        for name in f.namelist():
            words = tf.compat.as_str(f.read(name)).split()
        f.close()

        counts = [('UNK', -1)]
        counts.extend(collections.Counter(words).most_common(self._hparams.n_vocabulary - 2))
        string_map = [c[0] for c in counts]


        # Tokenization with lookup table. Retrieves a 1 x vocabulary sized
        # vector.
        vocabulary_table = tf.contrib.lookup.index_table_from_tensor(
            mapping=tf.constant(string_map),
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


class Modelfn():

    def __init__(self, hparams):
        self._hparams = hparams

    def _tokenizer_network(self, x_batch):
        raise NotImplementedError

    def _gate_dispatch(self, x_batch):
        raise NotImplementedError

    def _gate_combine(self, c_batches):
        raise NotImplementedError

    def _synthetic_network(self, t_batch, d_batch):
        raise NotImplementedError

    def _embedding_network(self, t_batch, d_batch):
        raise NotImplementedError

    def _target_network(self, embedding):
        raise NotImplementedError

    def _synthetic_loss(self, syn_batch, c_batch):
        raise NotImplementedError

    def _target_loss(self, embedding, y_batch):
        raise NotImplementedError

    def _model_fn(self):

        # x_batch: Model inputs. [None, 1] unicode encoded strings.
        self.x_batch = tf.compat.v1.placeholder(tf.string, [None, 1], name='batch of inputs')

        # y_batch: Supervised targets signals used during training and testing.
        self.y_batch = tf.compat.v1.placeholder(tf.float32, [None, self._hparams.n_targets], name='y_batch')

        # Use Synthetic: Flag, use synthetic inputs when running graph.
        self.use_synthetic = tf.compat.v1.placeholder(tf.bool, shape=[], name='use_synthetic')

         # Parent gradients: Gradients passed by this components parent.
        self.parent_error = tf.compat.v1.placeholder(tf.float32, [None, self._hparams.n_embedding], name='parent_grads')

        # Tokenizer network: x_batch --> t_batch
        with tf.compat.v1.variable_scope("tokenizer_network"):
            t_batch = self._tokenizer(self.x_batch)
            tokenizer_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="tokenizer_network")

        # Gating network: t_batch --> GATE --> [] --> GATE --> c_batch
        with tf.compat.v1.variable_scope("gating_network"):
            gated_batch = self._gate_dispatch(self.t_batch)
            child_inputs = []
            for i, gated_spikes in enumerate(gated_batch):
                child_inputs.append(_input_from_gate(gated_batch))
            c_batch = self._gate_combine(child_inputs)
            gating_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="gating_network")

        # Synthetic network: t_batch --> syn_batch
        with tf.compat.v1.variable_scope("synthetic_network"):
            syn_batch = self._synthetic_network(t_batch)
            synthetic_loss = self._synthetic_loss(syn_batch, c_batch)
            synthetic_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="synthetic_network")

        # Downstream switch: syn_batch || c_batch --> d_batch
        d_batch = tf.cond(
            tf.equal(self.use_synthetic, tf.constant(True)),
            true_fn=lambda: syn_batch,
            false_fn=lambda: c_batch)

        # Embedding network: t_batch + d_batch --> embedding
        with tf.compat.v1.variable_scope("embedding_network"):
            self.embedding = self._embedding_network(t_batch, d_batch)
            embedding_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="embedding_network")

        # Target network: embedding --> logits
        with tf.compat.v1.variable_scope("target_network"):
            logits = self._target_network(self.embedding)
            target_loss = self._target_loss(logits, self.y_batch)
            target_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="target_network")

        # Optimizer
        optimizer = self._optimizer()

        # Synthetic grads.
        synthetic_grads = optimizer.compute_gradients(  loss = synthetic_loss,
                                                        var_list = synthetic_vars)

        # Parent grads
        parent_grads = optimizer.compute_gradients(    loss = self.embedding,
                                                       var_list = embedding_vars,
                                                       grad_loss = self.parent_error)

        # Target grads
        target_grads = optimizer.compute_gradients(    loss = target_loss,
                                                       var_list = target_vars + embedding_vars + gate_vars)

        # Child grads
        child_grads = optimizer.compute_gradients(  loss = target_loss,
                                                    var_list = child_inputs)

        # Synthetic step.
        synthetic_step = optimizer.apply_gradients(synthetic_grads)

        # Parent step.
        parent_step = optimizer.apply_gradients(parent_grads)

        # Target step.
        target_step = optimizer.apply_gradients(target_grads)
