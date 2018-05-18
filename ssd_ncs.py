# import mvnc
import mvnc.mvncapi as fx
import sys
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(BASE_DIR, 'caffe/python'))
import caffe
import numpy as np

def ncs_prepare():
    print("[INFO] finding NCS devices...")
    devices = fx.EnumerateDevices()

    if len(devices) == 0:
        print("[INFO] No devices found. Please plug in a NCS")
        quit()

    print("[INFO] found {} devices. device0 will be used. "
          "opening device0...".format(len(devices)))
    device = fx.Device(devices[0])
    # try to open the device.  this will throw an exception if someone else has it open already
    try:
        device.OpenDevice()
    except:
        print("Error - Could not open NCS device.")
        quit()
    return device

def graph_prepare(PATH_TO_CKPT, device):
    print("[INFO] loading the graph file into RPi memory...")
    with open(PATH_TO_CKPT, mode="rb") as f:
        graph_in_memory = f.read()

    # load the graph into the NCS
    print("[INFO] allocating the graph on the NCS...")
    detection_graph = device.AllocateGraph(graph_in_memory)
    return detection_graph

transformer = caffe.io.Transformer({'data': (1, 3, 300, 300)})
transformer.set_transpose('data', (2, 0, 1))
transformer.set_mean('data', np.array([104,117,123])) # mean pixel
transformer.set_raw_scale('data', 255)  # the reference model operates on images in [0,255] range instead of [0,1]
transformer.set_channel_swap('data', (2,1,0))  # the reference model has channels in BGR order instead of RGB


def preprocess_image(PATH):
    image = caffe.io.load_image(PATH)
    return image

def ncs_clean(detection_graph, device):
    detection_graph.DeallocateGraph()
    device.CloseDevice()


exmaple_image = os.path.join(BASE_DIR, "pic/example.jpg")

graph_path = os.path.join(BASE_DIR, "models/ncs_ssd_graph")

image = preprocess_image(exmaple_image)

device = ncs_prepare()
graph = graph_prepare(graph_path, device)
graph.LoadTensor(image, None)
(output, _) = graph.GetResult()

print(output)

ncs_clean(graph, device)
