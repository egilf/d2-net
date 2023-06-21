import time
import zmq
import d2net_msg_pb2 as d2
import numpy as np
from matplotlib import pyplot as plt

#d2net imports:
import argparse
import numpy as np
from PIL import Image
import torch
import math
from tqdm import tqdm
from os import path
from lib.model_test import D2Net
from lib.utils import preprocess_image
from lib.pyramid import process_multiscale
import imageio
import scipy
import json
import os

def GetFloatMap(pmap: d2.ProtoFloatMap) -> np.ndarray:
    shape = (height, width) = (pmap.height, pmap.width)
    bytes_per_pixel = 4
    data = pmap.data
    if height*width*bytes_per_pixel != len(data):
        raise ValueError("Incompatible number of bytes")
    return np.frombuffer(data, dtype=np.float32).reshape(shape)

if __name__=="__main__":
    
    parser = argparse.ArgumentParser(description='D2Net extraction server')

    parser.add_argument(
        '--pipe_address', type=str, default='tcp://*:5555',
        help='address for pipe communication'
    )

    parser.add_argument(
        '--model', type=str, default='models//d2_ots.pth',
        help='path to the model'
    )

    args = parser.parse_args()
    print(args)

    #pipe init:
    # https://zeromq.org/socket-api/
    context = zmq.Context()                                 
    socket = context.socket(zmq.REP)
    address = args.pipe_address         #"tcp://*:5555"
    socket.bind(address)
    print("Socket.Bind() to: %s" % address)
    
    print("Working Directory: " + os.getcwd())

    #d2net init:
    use_cuda = torch.cuda.is_available()
    print("CUDA available: " + str(use_cuda))
    device = torch.device("cuda:0" if use_cuda else "cpu")
    max_edge=1600
    max_sum_edges=2800
    multiscale=False
    preprocessing='caffe'
    model = D2Net(
        model_file=args.model,
        use_relu=True,
        use_cuda=use_cuda
    )

    print("Waiting for requests...")
    while True:
        received = socket.recv()
        try:
            print("Message received")
            t = time.time()
            msg = d2.D2NetMessage.FromString(received)
            image = GetFloatMap(msg.image)
            print("Image received")
            print("Image Size: " + str(image.shape))

            if len(image.shape) == 2:
                image = image[:, :, np.newaxis]
                image = np.repeat(image, 3, -1)
                resized_image = image
            
            input_image = preprocess_image(
                resized_image,
                preprocessing=preprocessing
            )

            with torch.no_grad():
                if multiscale:
                    keypoints, scores, descriptors = process_multiscale(
                        torch.tensor(
                            input_image[np.newaxis, :, :, :].astype(np.float32),
                            device=device
                        ),
                        model
                    )
                else:
                    keypoints, scores, descriptors = process_multiscale(
                        torch.tensor(
                            input_image[np.newaxis, :, :, :].astype(np.float32),
                            device=device
                        ),
                        model,
                        scales=[1]
                    )
        
            ret = d2.D2NetMessage()
            ret.keypoints.width = keypoints.shape[1]
            ret.keypoints.height = keypoints.shape[0]
            ret.keypoints.data = keypoints.tobytes() 

            ret.scores.width = 1 #scores.shape[1]
            ret.scores.height = scores.shape[0]
            ret.scores.data = scores.tobytes()

            ret.descriptors.width = descriptors.shape[1]
            ret.descriptors.height = descriptors.shape[0]
            ret.descriptors.data = descriptors.tobytes()

            #ret.scores = d2.ProtoFloatMap()
            #ret.descriptors = d2.ProtoFloatMap()

            #elapsed = time.time() - t
            #print("Processed in: %f [sec]" % elapsed)

            #plt.imshow(image, interpolation='nearest')
            #plt.show()
            #time.sleep(1)

            socket.send(ret.SerializeToString());
        except Exception as e:
            socket.send(("CE: " + str(e)).encode('UTF-8'))

        #socket.send(b"Continue")                            #Send 'Continue' if you dont want to upload data to DIACore
