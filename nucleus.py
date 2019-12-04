

def Nucleus:
    def __init__(self, hparams, metagraph):
        self._hparams = hparams
        self._children = metagraph.nodes[self.hparams.identity].children
        self._graph = tf.Graph()
        self._session = tf.compat.v1.Session(graph=self._graph)
        with self._graph.as_default(), tf.device('/cpu:0'):
            self._model_fn()
            self._session.run(tf.compat.v1.global_variables_initializer())

    def spike(self, pspikes, cspikes):
        # Build Feeds dictionary.
        feeds = {self._spikes : pspikes}
        for _ in self._children:
            feeds[self._cspikes[i]] = cspikes[i]
        fetches = {'embedding' : self._output}
        return self._session.run(fetches, feeds)['embedding']

    def grade(self, pgrades, pspikes, cspikes):
        # Build Feeds dictionary.
        feeds = {}
        feeds[self._pspikes] = pspikes
        feeds[self._pgrads] = pgrades
        for _ in self._children:
            feeds[self._cspikes[i]] = cspikes[i]

        # Compute gradients for the children and apply the local step.
        cgrads = self._session.run([self._cgrads, self._train_step], feeds)[0]

        # Return downstream grads
        return cgrads

    def _model_fn(self):

        self._input = tf.compat.v1.placeholder(tf.float32, [-1, self._hparams.n_inputs])

        self._cspikes = []
        for _ in self._children:
            self._cspikes.append(tf.compat.v1.placeholder(tf.float32, [-1, self._hparams.n_embedding]))

        self._pgrads = tf.compat.v1.placeholder(
            tf.float32, [None, self._hparams.n_embedding], 'g')

        # Model weights and biases
        l_weights = {
            'w1':
                tf.Variable(
                    tf.random.truncated_normal([
                        self._hparams.n_embedding + self._hparams.n_embedding,
                        self._hparams.n_hidden1
                    ],
                                               stddev=0.1)),
            'w2':
                tf.Variable(
                    tf.random.truncated_normal(
                        [self._hparams.n_hidden1, self._hparams.n_hidden2],
                        stddev=0.1)),
            'w3':
                tf.Variable(
                    tf.random.truncated_normal(
                        [self._hparams.n_hidden2, self._hparams.n_embedding],
                        stddev=0.1)),
        }
        l_biases = {
            'b1':
                tf.Variable(tf.constant(0.1, shape=[self._hparams.n_hidden1])),
            'b2':
                tf.Variable(tf.constant(0.1, shape=[self._hparams.n_hidden2])),
            'b3':
                tf.Variable(tf.constant(0.1,
                                        shape=[self._hparams.n_embedding])),
            'b4':
                tf.Variable(tf.constant(0.1, shape=[self._hparams.n_targets])),
        }
        local_network_variables = list(l_weights.values()) + list(l_biases.values())

        # Local embedding network.
        input_layer = tf.concat([text_embedding, downstream_embedding], axis=1)
        hidden_layer1 = tf.nn.relu(
            tf.add(tf.matmul(input_layer, l_weights['w1']), l_biases['b1']))
        hidden_layer2 = tf.nn.relu(
            tf.add(tf.matmul(hidden_layer1, l_weights['w2']), l_biases['b2']))
        self._embedding = tf.nn.relu(
            tf.add(tf.matmul(hidden_layer2, l_weights['w3']), l_biases['b3']))

        # Optimizer: The optimizer for this component.
        optimizer = tf.compat.v1.train.AdamOptimizer(
            self._hparams.learning_rate)


        # Local + joiner network grads from upstream.
        self._gradients = optimizer.compute_gradients(loss=self._embedding,
                                                    var_list=local_network_variables
                                                    grad_loss=self._pgrads)

        # Train step from target Local + joiner network grads.
        self._train_step = optimizer.apply_gradients(self._gradients)
