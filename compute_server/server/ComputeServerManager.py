import socket
import threading
from ComputeMachine import ComputeMachine


class ComputeServerManager(threading.Thread):
    """
    Manages active compute servers and handles new connections.

    Attributes:
        host: the compute server manager's ip.
        port: the port on which the compute server manager will listen for new connections.
        lock: a thread lock object to prevent race conditions.
        compute_machines: a list containing all the active compute servers.
    """

    def __init__(self, host, port):
        """
        Create the computer server manager.

        :param host: the compute server manager's host ip.
        :param port: the port on which the compute server manager will listen for new connections.
        """
        super().__init__()

        self.host = host
        self.port = port

        self.lock = threading.Lock()
        self.compute_machines = list()

    def get_available_servers(self):
        """
        Returns a list of compute servers which are not handling a compute task.

        :return: a list of compute servers that can handle new compute tasks.
        """
        res = list()

        for s in self.compute_machines:
            if s.available:
                res.append(s)

        return res

    def remove_server(self, server):
        """
        Removes a compute server from the compute server manager.
        :param server: the server to remove.
        """
        if server not in self.compute_machines:
            raise Exception(f'Tried removing server which is not in the server list. server={server}')

        self.compute_machines.remove(server)

    def run(self):
        """
        Listens for new connections from compute servers and adds them to the manager. This method runs in a different thread.
        """

        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.bind((self.host, self.port))
        s.listen(5)

        print('Compute server listening on port:', self.port)

        while True:
            # wait for a new connection
            client_socket, address = s.accept()

            with self.lock:
                compute_machine = ComputeMachine(client_socket, self)
                compute_machine.start()
                self.compute_machines.append(compute_machine)

            print('New compute machine added with address:', address)

        s.close()
