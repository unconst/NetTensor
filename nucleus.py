

def Nucleus:
    def __init__(self, hparams, model_fn, metagraph):
        self._hparams = hparams
        self._children = metagraph.nodes[self.hparams.identity].children
        self._graph = tf.Graph()
        self._session = tf.compat.v1.Session(graph=self._graph)
        self._model_fn = model_fn
        with self._graph.as_default(), tf.device('/cpu:0'):
            self._model_fn.init()
            self._session.run(tf.compat.v1.global_variables_initializer())

    def spike(self, parent_spikes, child_spikes):
        # Build Feeds dictionary.
        feeds = {self._spikes : parent_spikes}
        for _ in self._children:
            feeds[self._child_spikes[i]] = child_spikes[i]
        fetches = {'embedding' : self._output}
        return self._session.run(fetches, feeds)['embedding']

