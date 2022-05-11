
import gym
import d3rlpy
import torch
dataset, env = d3rlpy.datasets.get_dataset("urban_env","two-way-v0")

import time
import zmq, zlib, pickle
import numpy as np


def send_zipped_pickle(socket, obj, flags=0, protocol=-1):
    """pickle an object, and zip the pickle before sending it"""
    p = pickle.dumps(obj, protocol)
    z = zlib.compress(p)
    return socket.send(z, flags=flags)

def recv_zipped_pickle(socket, flags=0, protocol=-1):
    """inverse of send_zipped_pickle"""
    z = socket.recv(flags)
    p = zlib.decompress(z)
    return pickle.loads(p)



TRAINING=False
agent =  d3rlpy.algos.DiscreteSAC()

if TRAINING:
    agent.fit(dataset, n_steps=100000)
    agent.save_model('model.pt')
# agent.save_policy('policy.pt')
else:
    # initialize with dataset
    agent.build_with_dataset(dataset)
    agent.load_model('model.pt')

    print("Connecting to hello world server…")
    context = zmq.Context()
    socket = context.socket(zmq.REQ)
    socket.connect("tcp://localhost:5555")
    #return_obj=np.random.rand(2,2)

    request=1
    action=[0]
    while True:

        print("Sending action %s …" % request, "action ", action)
        send_zipped_pickle(socket, action, protocol=4)

        message = recv_zipped_pickle(socket, protocol=4)
        state = message.squeeze()
    #    print("Received reply %s [ %s ]" % (request, message), type(message))
        # print("state is", state, type(state))

        action[0]=agent.predict(state.astype(np.float64))


        request=request+1
