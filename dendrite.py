


class Dendrite(self):

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
