"""Retry malformed depth transport packets without changing OpticalProbe v1.

Only image acquisition is retried. No synthetic image is supplied, no missing
measurement is accepted, and geometric assertions remain in the frozen probe.
"""
import time


class InvalidDepthResponse(RuntimeError):
    pass


class ValidatedDepthClient:
    def __init__(self, client, report, attempts=8, delay=.1):
        assert attempts > 0 and delay >= 0
        self.client, self.report = client, report
        self.attempts, self.delay = attempts, delay
        self.invalid_packets = 0

    def __getattr__(self, name):
        return getattr(self.client, name)

    def simGetImages(self, requests, *args, **kwargs):
        depth = [i for i, request in enumerate(requests)
                 if request.pixels_as_float and int(request.image_type) == 2]
        if not depth:
            return self.client.simGetImages(requests, *args, **kwargs)
        for attempt in range(1, self.attempts + 1):
            responses = self.client.simGetImages(requests, *args, **kwargs)
            invalid = []
            if len(responses) != len(requests):
                invalid.append({'response_count': len(responses), 'request_count': len(requests)})
            else:
                for index in depth:
                    response = responses[index]
                    if ((response.height, response.width) != (256, 256)
                            or len(response.image_data_float) != 256 * 256):
                        invalid.append({'index': index, 'height': response.height,
                                        'width': response.width,
                                        'float_count': len(response.image_data_float),
                                        'timestamp_ns': int(response.time_stamp)})
            if not invalid:
                return responses
            self.invalid_packets += 1
            self.report({'event': 'malformed_depth_transport_response', 'attempt': attempt,
                         'attempt_limit': self.attempts, 'invalid': invalid, 'at': time.time()})
            if attempt == self.attempts:
                raise InvalidDepthResponse('No valid depth response after %d attempts: %r'
                                           % (self.attempts, invalid))
            time.sleep(self.delay)

