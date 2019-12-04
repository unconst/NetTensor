from concurrent import futures
import grpc
from loguru import logger
import pickle
from threading import Lock

class Neuron(bittensor.proto.bittensor_pb2_grpc.BittensorServicer):

    def __init__(self, hparams, metagraph, nucleus):
        self.hparams = hparams
        self.nucleus = nucleus
        self.children = metagraph.nodes[self.hparams.identity].children
        self.lock = Lock()
        self.mem = {}

    def Spike(self, request, context):
        # 1. Unpack message.
        version = request.version
        source_id = request.source_id
        parent_id = request.parent_id
        message_id = request.message_id
        pspikes = pickle.loads(request.spikes)

        # 2. Make recursive calls to children.
        futures = []
        for child in self.metagraph.children:
            futures.append(self._spike_future(child, request))

        # 3. Fill child futures.
        cspikes = self._fill_futures(futures)

        # 4. Inference local neuron.
        lspikes = self.nucleus.spike(pspikes, cspikes)

        # 5. Build response.
        payload = pickle.dumps(lspikes, protocol=0)
        response = bittensor.proto.bittensor_pb2.SpikeResponse(
            version=version,
            source_id=source_id,
            child_id=self.config.identity,
            message_id=message_id,
            payload=payload)

        # 6. Save to memory
        self.mem[message_id] = types.SimpleNamespace(
            message_id = message_id,
            pspikes = pspikes,
            cspikes = cspikes,
            lspikes = lspikes,
        )

        # Return.
        return response

    def Grade(self, request, context):
        # Unpack request.
        source_id = request.source_id
        parent_id = request.parent_id
        message_id = request.message_id
        pgrades = pickle.loads(request.grads)

        # 6. Get memory
        # NOTE (const) So many lacking checks here.
        mem_buffer = self.mem[message_id]

        # Get local spikes.
        lspikes = mem_buffer.lspikes

        # Get child spikes.
        cspikes = mem_buffer.cspikes

        # Get parent spikes
        pspikes = mem_buffer.pspikes

        # Get child grads and local grads.
        cgrades = self.nucleus.grade(pgrades, pspikes, cspikes)

        # Make recursive calls.
        futures = []
        for i, child in enumerate(self.children):
            futures.append(self._grade_future(child, request, cgrades[i]))

        # Fill futures
        # NOTE(const): Grade responses can be disregarded.
        self._fill_grade_futures(futures)

        # delete memory:
        del self.memory[message_id]

        # Return boolean flag.
        return bittensor.proto.bittensor_pb2.GradeResponse(accept=True)

    def _fill_grade_futures(self, grad_futures):
        start = time.time()
        returned = [False for _ in range(len(grad_futures))]
        while True:
            for i, future in enumerate(grad_futures):
                if future.done():
                    returned[i] = True
            if time.time() - start > 1:
                break
            if sum(returned) == len(grad_futures):
                break

    def _fill_spike_futures(self, futures):
        start = time.time()
        returned = [False for _ in range(len(grad_futures))]
        responses = [None for _ in range(len(grad_futures))]
        while True:
            for i, future in enumerate(grad_futures):
                if future.done():
                    returned[i] = True
                    result = futures[i].result()
                    responses[i] = pickle.loads(result.payload)
            if time.time() - start > 1:
                break
            if sum(returned) == len(grad_futures):
                break
        return responses

    def _spike_future(self, child, request):
        try:
            # Build Stub and request proto.
            stub = bittensor.proto.bittensor_pb2_grpc.BittensorStub(child.channel)

            # Create spike request proto.
            request = bittensor.proto.bittensor_pb2.SpikeRequest(
                version=request.version,
                source_id=request.source_id,
                parent_id=self.config.identity,
                message_id=request.message_id,
                spikes=request.spikes)

            # Send TCP spike request with futures callback.
            return stub.Spike.future(request)
        except:
            return None

    def _grade_future(self, child, request, grades):
        try:
            # Build Stub and request proto.
            stub = bittensor.proto.bittensor_pb2_grpc.BittensorStub(child.channel)

            # Create spike request proto.
            request = bittensor.proto.bittensor_pb2.GradeRequest(
                version=request.version,
                source_id=request.source_id,
                parent_id=self.config.identity,
                message_id=request.message_id
                grades=pickle.dumps(grades, protocol=0))

            # Send TCP spike request with futures callback.
            return stub.Spike.future(request)
        except:
            return None
