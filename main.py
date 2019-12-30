from metagraph import Metagraph
from nucleus import Nucleus
from neuron import Neuron
from modelfn import Modelfn
from feynman import Feynman
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

    metagraph = Metagraph(hparams)

    feynmann = Feynmann(hparams)

    nucleus = Nucleus(hparams, modelfn)

    neuron = Neuron(hparams, nucleus, metagraph)

    neuron.serve()

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
            time.sleep(100)

    except KeyboardInterrupt:
        logger.debug('Neuron stopped with keyboard interrupt.')
        tear_down(hparams, neuron, nucleus, metagraph)

    except Exception as e:
        logger.error('Neuron stopped with interrupt on error: ' + str(e))
        tear_down(hparams, neuron, nucleus, metagraph)


if __name__ == '__main__':
    logger.debug("started neuron.")
    # Server parameters.
    hparams = Hparams.get_hparams()
    main(hparams)

