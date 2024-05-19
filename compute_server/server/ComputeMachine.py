from Requests import RequestType
import threading
from shared.Packets import *


class ComputeMachine(threading.Thread):
    """
    Represents a compute machine. Handles the traffic between the compute machine and the compute server manager. Acts as an interface for the request manager to dispatch compute tasks.

    Attributes:
        socket: the socket used to communicate with the compute server.
        ip: the ip address of the compute server.
        port: the port of the compute server.
        compute_server_manager: a reference to the compute server manager object.
        available: whether the compute server can handle new compute tasks.
        lock: a thread lock object to prevent race conditions.
        result_callback: a callback for the current compute task result.
        request_type: the request type this compute server can handle.
        machine_name: the name of the compute server.
        model_name: the name of the model the compute server is running.
        tensorflow_device: the device the compute server is running the model on.
    """

    def __init__(self, socket, computer_server_manager):
        """
        Creates a compute server object.

        :param socket: the socket for communicating with the compute server.
        :param computer_server_manager: a reference to the compute server manager.
        """
        super().__init__()
        self.socket = socket
        self.ip, self.port = socket.getpeername()

        self.computer_server_manager = computer_server_manager

        self.available = True

        self.lock = threading.Lock()
        self.result_callback = None

        self.request_type = RequestType.NONE

        self.machine_name = 'N/A'
        self.model_name = 'N/A'
        self.tensorflow_device = 'N/A'

    def run(self):
        """
        Listens for traffic from the compute server. This method runs in a separate thread.
        """
        try:
            while True:
                # wait for data from the compute server
                data = self.socket.recv(1024)

                if not data:
                    print('Closed connection with:', self.socket.getpeername())
                    self.computer_server_manager.remove_server(self)
                    return

                packet = decode_packet(data)

                # handle the packet
                if isinstance(packet, CPacketComputeResult):
                    with self.lock:
                        if self.result_callback is None:
                            raise Exception(f'Result callback not found for result {packet.result}')

                        self.result_callback(packet.result)
                        self.result_callback = None
                        self.available = True
                elif isinstance(packet, CPacketMachineInfo):
                    with self.lock:
                        self.machine_name = packet.machine_name

                        self.model_name = packet.model_name
                        if self.model_name == 'article_10m':
                            self.request_type = RequestType.ARTICLE
                        elif self.model_name == 'chatbot_125m':
                            self.request_type = RequestType.QUESTION
                        else:
                            raise Exception(f'Unknown model name model_name={self.model_name}')

                        self.tensorflow_device = packet.tensorflow_device
                else:
                    print(f'No handle logic for packet: {packet}')

        except ConnectionError:
            print('Closed connection with:', self.socket.getpeername())
            self.computer_server_manager.remove_server(self)

    def compute(self, request_data, callback):
        """
        Sends a compute task to the compute server.

        :param request_data: the compute task data, prompt and inference config.
        :param callback: a reference to the result callback method.
        """
        with self.lock:
            if not self.available:
                raise Exception('Tried computing with an unavailable machine!')

            self.available = False
            self.result_callback = callback

            prompt, (temp, top_p) = request_data

            packet = SPacketComputeTask(prompt, temp, top_p)
            self.socket.send(encode_packet(packet))
