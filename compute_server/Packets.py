import json
import abc


class Packet:
    """
    An abstract class of a packet.
    """

    @abc.abstractmethod
    def to_json(self):
        """
        Serializes the packet to json form.
        :return: the json representation of the packet.
        """

    @staticmethod
    @abc.abstractmethod
    def from_json(json_str):
        """
        Deserialized the packet from json form.

        :param json_str: the packet in json form.
        :return: the deserialized packet object.
        """


# ---------- Client-bound packets ----------


class SPacketSetModel(Packet):
    """
    Tells the compute server to change its current model.

    !!! Note: the logic for this packet has yet to be implemented !!!

    Attributes:
        model_name: the name of the new model.
    """
    def __init__(self, model_name):
        self.model_name = model_name

    def to_json(self):
        return json.dumps({'model_name': self.model_name})

    @staticmethod
    def from_json(json_str):
        parsed = json.loads(json_str)
        return SPacketSetModel(parsed['model_name'])


class SPacketComputeTask:
    """
    Sends a compute task to the compute server.

    Attributes:
        prompt: the task prompt.
        temp: the sampler's temperature.
        top_p the sampler's top_p value.
    """

    def __init__(self, prompt, temp, top_p):
        if not isinstance(temp, float) or not isinstance(top_p, float):
            raise Exception('Tried creating a SPacketComputeTask with temperature or top_p that is not a floating point number')

        self.prompt = prompt
        self.temp = temp
        self.top_p = top_p

    def to_json(self):
        return json.dumps({'prompt': self.prompt, 'temp': self.temp, 'top_p': self.top_p})

    @staticmethod
    def from_json(json_str):
        parsed = json.loads(json_str)
        return SPacketComputeTask(parsed['prompt'], parsed['temp'], parsed['top_p'])


# ---------- Server-bound packets ----------


class CPacketComputeResult:
    """
    Contains a compute task's result.

    Attributes:
        result: the result of the compute task.
    """
    def __init__(self, result):
        self.result = result

    def to_json(self):
        return json.dumps({'result': self.result})

    @staticmethod
    def from_json(json_str):
        parsed = json.loads(json_str)
        return CPacketComputeResult(parsed['result'])


class CPacketMachineInfo:
    """
    Tells the compute server manager basic information about the compute server.

    Attributes:
        machine_name: the name of the compute server.
        model_name: the name of the model currently running on the compute server.
        tensorflow_device: the device being used to run the model.
    """
    def __init__(self, machine_name, model_name, tensorflow_device):
        self.machine_name = machine_name
        self.model_name = model_name
        self.tensorflow_device = tensorflow_device

    def to_json(self):
        return json.dumps({'machine_name': self.machine_name,
                           'model_name': self.model_name,
                           'tensorflow_device': self.tensorflow_device})

    @staticmethod
    def from_json(json_str):
        parsed = json.loads(json_str)
        return CPacketMachineInfo(parsed['machine_name'], parsed['model_name'], parsed['tensorflow_device'])


packet_registry = {
    0: SPacketSetModel,
    1: SPacketComputeTask,
    2: CPacketComputeResult,
    3: CPacketMachineInfo
}


def get_packet_by_id(packet_id):
    """
    Find a packet using it's registry ID.

    :param packet_id: the packet's ID.
    :return: the class of the packet corresponding to the given ID.
    """
    if packet_id not in packet_registry:
        raise Exception(f'Tried handling a packet with an invalid id! Got ID={packet_id}')

    return packet_registry[packet_id]


def get_id_from_packet(packet):
    """
    Gets the ID from a packet object.

    :param packet: the packet object.
    :return: the ID corresponding to the packet object.
    """
    for packet_id, packet_class in packet_registry.items():
        if isinstance(packet, packet_class):
            return packet_id

    raise Exception(f'Tried resolving the ID for a packet which does not exist! Got packet: {packet}')


def encode_packet(packet):
    """
    Encodes the packet in order for it to be sent. Contains the ID and serialized packet.

    :param packet: the packet to encode.
    :return: the encoded packet in utf-8 form.
    """
    packet_id = get_id_from_packet(packet)
    serialized = packet.to_json()

    return bytes([packet_id]) + serialized.encode('utf-8')


def decode_packet(data):
    """
    Decodes an encoded packet. The encoded data contains the id and serialized packet.

    :param data: the encoded data.
    :return: a packet object from the encoded data.
    """
    packet_id = data[0]
    serialized = data[1:]

    return get_packet_by_id(packet_id).from_json(serialized)
