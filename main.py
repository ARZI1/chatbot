from RequestsManager import RequestManager
from ComputeServerManager import ComputeServerManager
from threading import Thread
import WebServer


def main():
    """
    Initializes the backend process. This process contains the web server, request manager and compute server.
    :return:
    """
    compute_server_manager = ComputeServerManager('0.0.0.0', 1337)
    compute_server_manager.start()

    request_dispatcher = RequestManager(compute_server_manager)
    dispatcher_thread = Thread(target=request_dispatcher.dispatch_requests)
    dispatcher_thread.start()

    WebServer.run(request_dispatcher, compute_server_manager)


if __name__ == '__main__':
    main()

