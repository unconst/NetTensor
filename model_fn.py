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
