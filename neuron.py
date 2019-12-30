from loguru import logger

class Neuron(self):
    def __init__(self, logger, dataset, nucleus, dendrite, metagraph):
        self._logger = looger
        self._dendrite = dendrite
        self._nucleus = nucleus
        self._dataset = dataset
        self._metagraph = metagraph

    def train(self):
        while self.running:
            # Next batch.
            x_batch, y_batch = self.dataset.next_batch()

            # Gate inputs.
            c_outputs = self._nucleus.gate_batch(x_batch)

            # Query children.
            c_batches = self._dendrite.query_children(c_inputs)

            # Train graph.
            outputs, metrics = self._nucleus.train(y_batch, x_batch, c_batches)

            logger.info(metrics)
