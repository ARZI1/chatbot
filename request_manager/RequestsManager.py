import time


class RequestManager:
    """
    Connects the web server and compute server by managing requests. The web server creates requests and adds them to the request manager. The request manager in turn sorts the them by priority and dispatches compute tasks when resources are available.

    Attributes:
        compute_server_manager: a reference to the compute server manager, used for dispatching compute tasks.
        active_requests: a dictionary containing all requests, the user's id acts as the key.
        queue: contains the requests sorted by their priority in the form of a queue.
    """

    def __init__(self, compute_server_manager):
        self.compute_server_manager = compute_server_manager
        self.active_requests = dict()
        self.queue = list()

    def add_request(self, sid, request):
        """
        Adds a request to the request manager.

        :param sid: the id of the user who added the request.
        :param request: the request object itself
        """
        if sid in self.active_requests:
            raise Exception('TODO: implement sid collision logic')

        self.active_requests[sid] = request
        self.queue.append(request)

    def remove_request(self, sid):
        """
        Removes a request from the request manager.

        :param sid: the id of the request's user
        """
        if sid not in self.active_requests:
            raise Exception(f'Tried to remove request but no matching sid was found sid={sid}')

        request = self.active_requests.pop(sid)
        self.queue.remove(request)

    def get_request_for_server(self, server):
        """
        Finds a request which can be handled by a given compute server. Request with higher priority in the queue will be tended to first.

        :param server: the compute server.
        :return: a reference to the request object, if not found None will be returned.
        """
        for request in self.queue:
            if not request.waiting_for_response and request.request_type == server.request_type:
                return request

        return None

    def dispatch_requests(self):
        """
        Continuously checks for available compute servers and dispatches compute requests. This method runs in its own thread.
        """
        while True:
            compute_servers = self.compute_server_manager.get_available_servers()
            for server in compute_servers:
                request = self.get_request_for_server(server)
                if request is None:
                    continue

                # move request to the back of the queue
                self.queue.remove(request)
                self.queue.append(request)

                request_data = request.get_compute_data()
                callback = request.handle_compute_result
                server.compute(request_data, callback)

            time.sleep(0.01)
