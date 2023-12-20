import urllib.request
import urllib.error
import urllib.parse
import base64
import json
import os
import sys
import subprocess
import random
from time import sleep
import multiprocessing

RPC_URL = os.environ.get("RPC_URL", "http://127.0.0.1:8332")
RPC_USER = os.environ.get("RPC_USER", "replaceme")
RPC_PASS = os.environ.get("RPC_PASS", "replaceme") 

################################################################################
# Bitcoin Daemon JSON-HTTP RPC
################################################################################
def rpc(method, params=None):
    """ 
    Make an RPC call to the Bitcoin Daemon JSON-HTTP server.

    Arguments:
        method (string): RPC method
        params: RPC arguments

    Returns:
        object: RPC response result.
    """

    rpc_id = random.getrandbits(32)
    data = json.dumps({"id": rpc_id, "method": method, "params": params}).encode()
    auth = base64.encodebytes((RPC_USER + ":" + RPC_PASS).encode()).decode().strip()

    request = urllib.request.Request(RPC_URL, data, {"Authorization": "Basic {:s}".format(auth)})

    with urllib.request.urlopen(request) as f:
        response = json.loads(f.read())

    if response['id'] != rpc_id:
        raise ValueError("Invalid response id: got {}, expected {:u}".format(response['id'], rpc_id))
    elif response['error'] is not None:
        raise ValueError("RPC error: {:s}".format(json.dumps(response['error'])))

    return response['result']

################################################################################
# Bitcoin Daemon RPC Call Wrappers
################################################################################
def rpc_getblockhash(height):
    return rpc( "getblockhash", [height] )

def rpc_getblock( Hash ):
    return rpc( "getblock", [Hash, 2] )
def rpc_getblockcount():
    return rpc( "getblockcount" )

################################################################################
#                                          Main 
################################################################################
if __name__ == "__main__":
    last = rpc_getblockcount()

    while True:
        if last != rpc_getblockcount():
            print(last)
            last = rpc_getblockcount()
            print(last)
            parse = subprocess.run( "pgrep -a python3 | grep cado" , capture_output=True, shell=True ).stdout.split( "\n".encode() )
            for line in parse:
                print(line)
                if line:
                    pid = line.decode("utf-8").split()[0]  
                    print(pid, flush=True)
                    parse = subprocess.run( "kill -9 " + pid, capture_output=True, shell=True )

            parse = subprocess.run( "pkill cuda-ecm", capture_output=True, shell=True )
            print()
