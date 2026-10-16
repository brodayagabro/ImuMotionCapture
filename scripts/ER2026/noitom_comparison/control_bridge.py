"""Move optional recorder-control requests from TCP workers onto the Qt thread."""
from collections import OrderedDict
import queue
import threading


class ControlBridge:
    def __init__(self):
        self.pending = queue.Queue()
        self.responses = OrderedDict()

    def request(self, payload, timeout=2.):
        item = {'payload': payload, 'done': threading.Event(), 'cancelled': False}
        self.pending.put(item)
        if not item['done'].wait(timeout):
            item['cancelled'] = True
            return {'accepted': False, 'reason': 'control_timeout'}
        return item['response']

    def process(self, callback):
        for _ in range(50):
            try:
                item = self.pending.get_nowait()
            except queue.Empty:
                break
            if item['cancelled']:
                continue
            payload = item['payload']
            key = payload['request_id']
            if key in self.responses:
                original, response = self.responses[key]
                if original != payload:
                    response = {'accepted': False, 'reason': 'request_id_reused'}
            else:
                try:
                    response = {'accepted': True, **callback(payload)}
                except (ValueError, OSError) as error:
                    response = {'accepted': False, 'reason': str(error)}
                self.responses[key] = (payload, response)
                if len(self.responses) > 2048:
                    self.responses.popitem(last=False)
            item['response'] = response
            item['done'].set()
