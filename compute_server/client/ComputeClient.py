import sys
import socket
from Packets import *


def main(host, port, machine_name, model_name):
    """
    Initializes the compute server and maintains connections with the main compute server.

    :param host: the address of the main compute server.
    :param port: the port of the main compute server.
    :param machine_name: the name of this compute server.
    :param model_name: the name of the model to run on this compute server.
    """

    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.connect((host, port))

    # models are imported separately as they need different libraries
    model = None
    if model_name == 'article_10m':
        from ArticleModel import ArticleModel
        model = ArticleModel()
    elif model_name == 'chatbot_125m':
        from ChatbotModel import ChatbotModel
        model = ChatbotModel()
    else:
        raise Exception(f'Invalid model name, model_name={model_name}')

    model.load_model()
    model.load_tokenizer()

    tensorflow_device = model.get_tensorflow_device()

    info_packet = CPacketMachineInfo(machine_name, model_name, tensorflow_device)
    s.send(encode_packet(info_packet))

    # main compute loop
    while True:
        data = s.recv(65536)
        packet = decode_packet(data)

        result = model.compute(packet.prompt, packet.temp, packet.top_p)
        print(f'Generated token with temp={packet.temp} top_p={packet.top_p} => {result}')

        response = CPacketComputeResult(result)
        s.send(encode_packet(response))


if __name__ == '__main__':
    """
    Script entry point.
    
    The script needs to get the following arguments:
    1. address of main compute server
    2. port of main compute server
    3. name of this compute server
    4. name of the model to run
    """
    args = sys.argv[1:]

    if len(args) != 4:
        raise Exception('Invalid number of script arguments! Must be 4.')

    args[1] = int(args[1])  # convert port to integer

    main(*args)
