from metagraph import Metagraph
from nucleus import Nucleus
from neuron import Neuron
from modelfn import Modelfn
from synapse import Synapse
from hprams import Hparams

import argparse
from datetime import timedelta
from loguru import logger
import time
from timeloop import Timeloop

def set_timed_loops(tl, config, neuron, metagraph):

    # Pull the updated graph state (Vertices, Edges, Weights)
    @tl.job(interval=timedelta(seconds=7))
    def pull_metagraph():
        metagraph.gossip()

    # Reselect channels.
    @tl.job(interval=timedelta(seconds=10))
    def connect():
        neuron.connect()

def main(hparams):

    dataset = Dataset(hparams)

    metagraph = Metagraph(hparams)

    dendrite = Dendrite(hparams)

    modelfn = Modelfn(hparams)

    nucleus = Nucleus(hparams, modelfn)

    neuron = Neuron(hparams, nucleus, metagraph)

    synapse = Synapse(hparams, neuron, metagraph)

    synapse.serve()

    tl = Timeloop()
    set_timed_loops(tl, hparams, neuron, metagraph)
    tl.start(block=False)

    def tear_down(_hparams, _neuron, _nucleus, _metagraph):
        del _neuron
        del _nucleus
        del _metagraph
        del _hparams

    try:
        while True:
            neuron.train()

    except KeyboardInterrupt:
        tear_down(hparams, neuron, nucleus, metagraph)

    except Exception as e:
        tear_down(hparams, neuron, nucleus, metagraph)


if __name__ == '__main__':
    hparams = Hparams.get_hparams()
    main(hparams)

