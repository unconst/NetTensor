

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

    def spike(self, x_batch, c_batches):
        feeds = {
            self.x_batch : x_batch
        }
        for _ in self._children:
            feeds[self.c_batch[i]] = c_batches[i]
        fetches = {'y_batch' : self.y_batch}
        return self._session.run(fetches, feeds)['y_batch']


    def grade(self, y_grads, x_batch):
        feeds = {
            self.x_batch: x_batch,
            self.y_grads: y_grads
        }
        self._session.run(feeds=feeds)


