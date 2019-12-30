
class hparams:
    def get_hparams():
        parser = argparse.ArgumentParser()
        global_params()
        vocab_params()
        training_params()
        architecture_params()
        hparams = parser.parse_args()
        return hparams

    def global_params():
        parser.add_argument(
            '--identity',
            default='abcd',
            type=str,
            help="network identity. Default identity=abcd"
        )

        parser.add_argument(
            '--serve_address',
            default='0.0.0.0',
            type=str,
            help="Address to server neuron. Default serve_address=0.0.0.0"
        )

        parser.add_argument(
            '--bind_address',
            default='0.0.0.0',
            type=str,
            help="Address to bind neuron. Default bind_address=0.0.0.0"
        )

        parser.add_argument(
            '--port',
            default='9090',
            type=str,
            help="Port to serve neuron on. Default port=9090"
        )

        parser.add_argument(
            '--logdir',
            default="/tmp/",
            type=str,
            help="logging output directory. Default logdir=/tmp/"
        )

    def vocab_params():
        parser.add_argument(
            '--corpus_path',
            default='text8.zip',
            type=str,
            help='Path to corpus of text. Default corpus_path=neurons/Mach/data/text8.zip'
        )

        parser.add_argument(
            '--n_vocabulary',
            default=50000,
            type=int,
            help='Size fof corpus vocabulary. Default vocabulary_size=50000'
        )

        parser.add_argument(
            '--n_sampled',
            default=64,
            type=int,
            help='Number of negative examples to sample during training. Default num_sampled=64'
        )

    def training_params(self):
        parser.add_argument(
            '--batch_size',
            default=50,
            type=int,
            help='The number of examples per batch. Default batch_size=128'
        )

        parser.add_argument(
            '--learning_rate',
            default=1e-4,
            type=float,
            help='Component learning rate. Default learning_rate=1e-4'
        )

    def architecture_params():

        parser.add_argument(
            '--n_repr',
            default=128,
            type=int,
            help='Size of representation. Default n_representation=128'
        )

        parser.add_argument(
            '--n_children',
            default=5,
            type=int,
            help='The number of graph neighbors. Default n_children=5'
        )

        parser.add_argument(
            '--syn_hidden',
            default=512,
            type=int,
            help='Size of synthetic network hidden layers. Default n_hidden1=512'
        )

        parser.add_argument(
            '--syn_layers',
            default=2,
            type=int,
            help='Number of synthetic layers. Default syn_layers=2'
        )

        parser.add_argument(
            '--repr_hidden',
            default=512,
            type=int,
            help='Size of represnetation network hidden layers. Default n_hidden1=512'
        )

        parser.add_argument(
            '--repr_layers',
            default=2,
            type=int,
            help='Number of representation layers. Default syn_layers=2'
        )

        parser.add_argument(
            '--target_hidden',
            default=512,
            type=int,
            help='Size of target network hidden layers. Default n_hidden1=512'
        )

        parser.add_argument(
            '--target_layers',
            default=2,
            type=int,
            help='Number of target network layers. Default syn_layers=2'
        )


