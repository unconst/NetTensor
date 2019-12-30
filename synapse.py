from concurrent import futures
import grpc
from loguru import logger
import pickle

class Synapse(nettensor.proto.nettensor_pb2_grpc.NettensorServicer):

    def __init__(self, hparams, neuron):
        self.hparams = hparams
        self.neuron = neuron

    def Spike(self, request, context):
        # 1. Unpack message.
        version = request.version
        source_id = request.source_id
        parent_id = request.parent_id
        message_id = request.message_id
        x_batch = pickle.loads(request.x_batch)

        # 2. Inference local neuron.
        repr_batch = self.neuron.spike(x_batch)

        # 3. Build response.
        payload = pickle.dumps(repr_batch, protocol=0)
        response = bittensor.proto.bittensor_pb2.SpikeResponse(
            version=version,
            source_id=source_id,
            child_id=self.config.identity,
            message_id=message_id,
            payload=payload)

        # Return.
        return response

    def Grade(self, request, context):
        # 1. Unpack request.
        source_id = request.source_id
        parent_id = request.parent_id
        message_id = request.message_id
        x_batch = pickle.loads(request.spikes)
        repr_grads = pickle.loads(request.grads)

        # 2. Grad nucleus.
        self.nucleus.neuron(repr_grads, x_batch)

        # 3. Return.
        return bittensor.proto.bittensor_pb2.GradeResponse(accept=True)
