import ctypes
import urllib.request
import urllib.error
import urllib.parse
import base64
import json
import hashlib
import struct
import random
import os
import sys
import struct
import hashlib
import base58
import secrets
import shutil 
import nvidia_smi
import random 
import subprocess
import concurrent.futures
import sympy
from time import time
from math import gcd
from sympy import isprime

RPC_URL  = os.environ.get("RPC_URL" , "http://127.0.0.1:8332")
RPC_USER = os.environ.get("RPC_USER", "replaceme")
RPC_PASS = os.environ.get("RPC_PASS", "replaceme") 

nvidia_smi.nvmlInit()

def create_config_file(level, idx, filename ):
    ecmparams = {     "b1": [11000, 50000, 250000, 1000000, 3000000, 11000000, 43000000 ],
                  "curves": [   86,   214,    430,     910,    2351,     4482,     7557 ]  }    

    conf_data = """
; Example config file for co-ecm
[general]

; server or file
mode = file

; Logfile location
logfile = ./log.txt

; Output file of abandoned, i.e. unsolved tasks.
; Format is the same as the output format, without listing factors
;
; Example line:
; 44800523911798220433379600867; # effort 112 
output_abandoned = ./abandoned.txt

; Log level
;
;   1: "VERBOSE",
;   2: "DEBUG",
;   3: "INFO",
;   4: "WARNING",
;   5: "ERROR",
;   6: "FATAL",
;   7: "NONE"
; Default is set at compile time.
loglevel = 3    

; Use a random seed for the random number generator used to generate points and 
; curves. If set to 'false', each run of the program will behave the same
; provided the same input data.
; Default: true
random = true

[server]
port = 11111

[file]
; Input file.
; The input file should contain a single number to be factored per line. Lines
; starting with anything but a digit are skipped.
;
; Example line:
; 44800523911798220433379600867
input = ./input{}.txt


; Output file.
; Each fully factored input number is appended to the output on its own line in
; the format 
; (input number);(factor),(factor),(factor), # effort: (number of curves)
;
; Example line:
; 44800523911798220433379600867;224536506062699,199524454608233, # effort: 12
output = ./output{}.txt


[cuda]

; Number of concurrent cuda streams to issue to GPU 
; Default: 2
streams = 1 

; Number of threads per block for cuda kernel launches.
; Set to auto to determine setting for maximum parallel resident blocks per SM at runtime.
; Note: The settings determined by 'auto' are not always automatically the optimal setting for maximum throughput.
; Default: auto
threads_per_block = auto 

; Constant memory is used for (smaller) scalars during point multiplication.
; When the scalar is too large to fit into constant memory or this option is set 
; to 'false', global device memory is used.
; Default: true
use_const_memory = true

[ecm]
; Redo ECM until numbers are fully factored.
; If set to false, only the first factor is returned.
; Default: false
;find_all_factors = true
iind_all_factors = false

; Set the computation of the scalar s for point multiplication. With
; 'powersmooth' set to 'true', then s = lcm(2, ..., b1). If set to false,
; s = primorial(2, ..., b1), i.e. the product of all primes less than or equal
; to b1. 
; Default: true
powersmooth = true

b1 = {}
b2 = 100000


; Maximum effort per input number.
; With each curve, the already spent effort is incremented. Thus, with effort
; set to 100, ecm stage1 (and stage2) will be executed on 100 curves per input
; number.
; Default: 10
effort = {} 

; Set the curve generator function. 
; Use 2 under normal circumstances.
;   0: "Naive"
;   1: "GKL2016_j1"
;   2: "GKL2016_j4"
; Default: 2
curve_gen = 2   

; Use only points for finding factors that are off curve.
; After point multiplication, use all resulting points to find factors. If set 
; to 'false' coordinates of points will be checked that do not fulfill the curve
; equation.
; Settings for stage1 and stage2 respectively.
; Default: true
stage1.check_all = false
stage2.check_all = true

; Enable/Disable stage 2.
; If set to 'false', only stage 1 of ECM is performed.
; Default: true
stage2.enabled = false

; Set the window size for stage 2
; Default: 2310
;stage2.window_size = 2310
""".format( idx, idx, ecmparams["b1"][level], ecmparams["curves"][level]  )

    with open( filename , 'w') as inputfile:
        inputfile.write( conf_data  )
        inputfile.close()


################################################################################
# CTypes and utility functions
################################################################################
class CParams(ctypes.Structure):
    _fields_=[("hashRounds",ctypes.c_uint32 ),
              ("MillerRabinRounds",ctypes.c_uint32 )  
             ]
    
class uint1024(ctypes.Structure):
    _fields_=[("data", ctypes.c_uint64 * 16 )]

class uint256(ctypes.Structure):
    _fields_=[("data", ctypes.c_uint64 * 4 )]
    
def uint256ToInt( m ):
    ans = 0    
    for idx,a in enumerate(m):
        ans += a << (idx*64)
    return ans

def uint1024ToInt( m ):
    ans = 0    

    if hasattr(m, 'data'):
        for idx in range(16):
            ans += m.data[idx] << (idx*64)
    else:
        for idx,a in enumerate(m):
            ans += a << (idx*64)
    
    return ans

def IntToUint1024( m ):
    ans = [0]*16
    n = int(m)
    MASK = (1<<64)-1
    
    for idx in range(16):
        ans[idx] = (m >> (idx*64)) & MASK
    
    return (ctypes.c_uint64 * 16)(*ans)
    
    
def hashToArray( Hash ):
    if Hash == 0:
        return [0,0,0,0]
    
    number = int(Hash,16)
    MASK = (1 << 64) - 1
    arr = [ ( number >> 64*(jj) )&MASK for jj in range(0, 4) ]    
    
    return arr


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

def rpc_getblocktemplate():
    try:
        return rpc("getblocktemplate", [{"rules": ["segwit"]}])
    except ValueError:
        return {}

def rpc_submitblock(block_submission):
    return rpc("submitblock", [block_submission])

def rpc_getblockcount():
    return rpc( "getblockcount" )


################################################################################
# Representation Conversion Utility Functions
################################################################################
def int2lehex(value, width):
    """
    Convert an unsigned integer to a little endian ASCII hex string.
    Args:
        value (int): value
        width (int): byte width
    Returns:
        string: ASCII hex string
    """

    return value.to_bytes(width, byteorder='little').hex()


def int2varinthex(value):
    """
    Convert an unsigned integer to little endian varint ASCII hex string.
    Args:
        value (int): value
    Returns:
        string: ASCII hex string
    """

    if value < 0xfd:
        return int2lehex(value, 1)
    elif value <= 0xffff:
        return "fd" + int2lehex(value, 2)
    elif value <= 0xffffffff:
        return "fe" + int2lehex(value, 4)
    else:
        return "ff" + int2lehex(value, 8)


def bitcoinaddress2hash160(addr):
    """
    Convert a Base58 Bitcoin address to its Hash-160 ASCII hex string.
    Args:
        addr (string): Base58 Bitcoin address
    Returns:
        string: Hash-160 ASCII hex string
    """

    table = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz"

    hash160 = 0
    addr = addr[::-1]
    for i, c in enumerate(addr):
        hash160 += (58 ** i) * table.find(c)

    # Convert number to 50-byte ASCII Hex string
    hash160 = "{:050x}".format(hash160)

    # Discard 1-byte network byte at beginning and 4-byte checksum at the end
    return hash160[2:50 - 8]

################################################################################
# Transaction Coinbase and Hashing Functions
################################################################################
def tx_encode_coinbase_height(height):
    """
    Encode the coinbase height, as per BIP 34:
    https://github.com/bitcoin/bips/blob/master/bip-0034.mediawiki
    Arguments:
        height (int): height of the mined block
    Returns:
        string: encoded height as an ASCII hex string
    """

    width = (height.bit_length() + 7 )//8 

    return bytes([width]).hex() + int2lehex(height, width)

def make_P2PKH_from_public_key( publicKey = "03564213318d739994e4d9785bf40eac4edbfa21f0546040ce7e6859778dfce5d4" ):
    from hashlib import sha256 as sha256
   
    address   = sha256( bytes.fromhex( publicKey) ).hexdigest()
    address   = hashlib.new('ripemd160', bytes.fromhex( address ) ).hexdigest()
    address   = bytes.fromhex("00" + address)
    addressCS = sha256(                address     ).hexdigest()
    addressCS = sha256( bytes.fromhex( addressCS ) ).hexdigest()
    addressCS = addressCS[:8]
    address   = address.hex() + addressCS
    address   = base58.b58encode( bytes.fromhex(address))
    
    return address
    
def tx_make_coinbase(coinbase_script, pubkey_script, value, height, wit_commitment ):
    """
    Create a coinbase transaction.
    Arguments:
        coinbase_script (string): arbitrary script as an ASCII hex string
        address (string): Base58 Bitcoin address
        value (int): coinbase value
        height (int): mined block height
    Returns:
        string: coinbase transaction as an ASCII hex string
    """
    # See https://en.bitcoin.it/wiki/Transaction
    coinbase_script = tx_encode_coinbase_height(height) + coinbase_script

    tx = ""
    # version
    tx += "02000000"
    # in-counter
    tx += "01"
    # input[0] prev hash
    tx += "0" * 64
    # input[0] prev seqnum
    tx += "ffffffff"
    # input[0] script len
    tx += int2varinthex(len(coinbase_script) // 2)
    # input[0] script
    tx += coinbase_script
    # input[0] seqnum
    tx += "00000000"
    # out-counter
    #tx += "02" if default_witness_commitment else "01"
    tx += "02"
    # output[0] value
    tx += int2lehex(value, 8)
    # output[0] script len
    tx += int2varinthex(len(pubkey_script) // 2)
    # output[0] script
    tx += pubkey_script
    # witness commitment value
    tx += int2lehex(0, 8)
    # witness commitment script len
    tx += int2varinthex(len(wit_commitment) // 2)
    # witness commitment script
    tx += wit_commitment    
    # lock-time
    tx += "00000000"

    return tx

def tx_compute_hash(tx):
    """
    Compute the SHA256 double hash of a transaction.
    Arguments:
        tx (string): transaction data as an ASCII hex string
    Return:
        string: transaction hash as an ASCII hex string
    """

    return hashlib.sha256(hashlib.sha256(bytes.fromhex(tx)).digest()).digest()[::-1].hex()

def tx_compute_merkle_root(tx_hashes):
    """
    Compute the Merkle Root of a list of transaction hashes.
    Arguments:
        tx_hashes (list): list of transaction hashes as ASCII hex strings
    Returns:
        string: merkle root as a big endian ASCII hex string
    """
    
    # Convert list of ASCII hex transaction hashes into bytes
    tx_hashes = [bytes.fromhex(tx_hash)[::-1] for tx_hash in tx_hashes]

    # Iteratively compute the merkle root hash
    while len(tx_hashes) > 1:
        # Duplicate last hash if the list is odd
        if len(tx_hashes) % 2 != 0:
            tx_hashes.append(tx_hashes[-1])

        tx_hashes_new = []

        for i in range(len(tx_hashes) // 2):
            # Concatenate the next two
            concat = tx_hashes.pop(0) + tx_hashes.pop(0)
            # Hash them
            concat_hash = hashlib.sha256(hashlib.sha256(concat).digest()).digest()
            # Add them to our working list
            tx_hashes_new.append(concat_hash)

        tx_hashes = tx_hashes_new

    # Format the root in big endian ascii hex
    return tx_hashes[0][::-1].hex()

################################################################################
# Bitcoin Core Wrappers
################################################################################
gHash = ctypes.CDLL("./gHash.so").gHash
gHash.restype = uint1024

class CBlock(ctypes.Structure):
    blocktemplate = {}
    _hash = "0"*32
    _fields_ = [("nP1",                 ctypes.c_uint64 * 16),
              ("hashPrevBlock",       ctypes.c_uint64 * 4 ),
              ("hashMerkleRoot",      ctypes.c_uint64 * 4 ),
              ("nNonce",   ctypes.c_uint64),
              ("wOffset",  ctypes.c_int64),
              ("nVersion", ctypes.c_uint32),
              ("nTime",    ctypes.c_uint32),
              ("nBits",    ctypes.c_uint16),
             ]
    block = None
    param = None
    scriptPubKey = None
 
    def get_next_block_to_work_on(self):
        blocktemplate      = rpc_getblocktemplate()
        self.blocktemplate = blocktemplate 

        prevBlock = blocktemplate["previousblockhash"]
        prevBlock = hashToArray(prevBlock)

        merkleRoot = blocktemplate["merkleroothash"]
        merkleRoot = hashToArray(merkleRoot)

        self.nP1                 = (ctypes.c_uint64 * 16)(*([0]*16))
        self.hashPrevBlock       = (ctypes.c_uint64 * 4)(*prevBlock)
        self.hashMerkleRoot      = (ctypes.c_uint64 * 4)(*merkleRoot )
        self.nNonce   = 0
        self.nTime    = ctypes.c_uint32( blocktemplate["curtime"] )
        self.nVersion = ctypes.c_uint32( blocktemplate["version"] )
        self.nBits    = ctypes.c_uint16( blocktemplate["bits"] )
        self.wOffset  = 0
        
        return self
    
    def serialize_block_header(self):
        #Get the data
        nP1                 = hex(uint1024ToInt(self.nP1)                 )[2:].zfill(256)
        hashPrevBlock       = hex(uint256ToInt( self.hashPrevBlock)       )[2:].zfill(64)
        hashMerkleRoot      = hex(uint256ToInt( self.hashMerkleRoot)      )[2:].zfill(64)
        nNonce              = struct.pack("<Q", self.nNonce)
        wOffset             = struct.pack("<q", self.wOffset)
        nVersion            = struct.pack("<L", self.nVersion)
        nTime               = struct.pack("<L", self.nTime)
        nBits               = struct.pack("<H", self.nBits)
        
        #Reverse bytes of the hashes as little-Endian is needed for bitcoind
        nP1                 = bytes.fromhex(nP1)[::-1]
        hashPrevBlock       = bytes.fromhex(hashPrevBlock)[::-1] 
        hashMerkleRoot      = bytes.fromhex(hashMerkleRoot)[::-1]
                                                
        #Serialize in the right order
        CBlock1 = bytes()
        CBlock1 += nP1
        CBlock1 += hashPrevBlock
        CBlock1 += hashMerkleRoot
        CBlock1 += nNonce
        CBlock1 += wOffset
        CBlock1 += nVersion
        CBlock1 += nTime
        CBlock1 += nBits
        
        return CBlock1
    
    def __str__(self):
        
        #Get the data
        nP1                 = hex(uint1024ToInt(self.nP1)                 )[2:].zfill(256)
        hashPrevBlock       = hex(uint256ToInt( self.hashPrevBlock)       )[2:].zfill(64)
        hashMerkleRoot      = hex(uint256ToInt( self.hashMerkleRoot)      )[2:].zfill(64)
        nNonce              = struct.pack("<Q", self.nNonce).hex()
        wOffset             = struct.pack("<q", self.wOffset).hex()
        nVersion            = struct.pack("<L", self.nVersion).hex()
        nTime               = struct.pack("<L", self.nTime).hex()
        nBits               = struct.pack("<H", self.nBits).hex()
        
        #Reverse bytes of the hashes as little-Endian is needed for bitcoind
        nP1                 = bytes.fromhex(nP1)[::-1].hex()
        hashPrevBlock       = bytes.fromhex(hashPrevBlock)[::-1].hex() 
        hashMerkleRoot      = bytes.fromhex(hashMerkleRoot)[::-1].hex()
        
        s  = "CBlock class: \n"
        s += "                    nP1: " + str(nP1)                 + "\n"
        s += "          hashPrevBlock: " + str(hashPrevBlock)       + "\n"
        s += "         hashMerkleRoot: " + str(hashMerkleRoot)      + "\n"
        s += "                 nNonce: " + str(nNonce)              + "\n"
        s += "                wOffset: " + str(wOffset)             + "\n"
        s += "               nVersion: " + str(nVersion)            + "\n"
        s += "                  nTime: " + str(nTime)               + "\n"
        s += "                  nBits: " + str(nBits)               + "\n"
    
        return s
    
    def int2lehex(self, value, width):
        """
        Convert an unsigned integer to a little endian ASCII hex string.
        Args:
            value (int): value
            width (int): byte width
        Returns:
            string: ASCII hex string
        """

        return value.to_bytes(width, byteorder='little').hex()

    def int2varinthex(self, value):
        """
        Convert an unsigned integer to little endian varint ASCII hex string.
        Args:
            value (int): value
        Returns:
            string: ASCII hex string
        """

        if value < 0xfd:
            return self.int2lehex(value, 1)
        elif value <= 0xffff:
            return "fd" + self.int2lehex(value, 2)
        elif value <= 0xffffffff:
            return "fe" + self.int2lehex(value, 4)
        else:
            return "ff" + self.int2lehex(value, 8)

    def prepare_block_for_submission(self):
        #Get block header
        submission = self.serialize_block_header().hex()
        
        # Number of transactions as a varint
        submission += self.int2varinthex(len(self.blocktemplate['transactions']))
        
        # Concatenated transactions data
        for tx in self.blocktemplate['transactions']:
            submission += tx['data']
            
        return submission
    
    def rpc_submitblock(self):
        submission = self.prepare_block_for_submission()
        print( "Submission: ", submission)

        return rpc_submitblock(submission), submission
    
    def compute_raw_hash(self):
        """
        Compute the raw SHA256 double hash of a block header.
        Arguments:
            header (bytes): block header
        Returns:
            bytes: block hash
        """

        return hashlib.sha256(hashlib.sha256(self.serialize_block_header()).digest()).digest()[::-1]


    def prepare_block(self):
        #Get parameters and candidate block
        block = self.get_next_block_to_work_on()
        param = getParams()

        # Add an coinbase transaction to the block template transactions
        coinbase_tx = {}

        # Update the coinbase transaction with coinbase txn information
        coinbase_script = ""
        coinbase_tx['data'] = tx_make_coinbase( coinbase_script, 
                                                self.scriptPubKey, 
                                                block.blocktemplate['coinbasevalue'], 
                                                block.blocktemplate['height'], 
                                                block.blocktemplate.get("default_witness_commitment") )
        coinbase_tx['txid'] = tx_compute_hash(coinbase_tx['data'])
        
        #Add transaction to our block
        block.blocktemplate['transactions'].insert(0, coinbase_tx)
       
        # Recompute the merkle root
        block.blocktemplate['merkleroot'] = tx_compute_merkle_root([tx['txid'] for tx in block.blocktemplate['transactions']])   
        merkleRoot = uint256()
        merkleRoot = (ctypes.c_uint64 * 4)(*hashToArray( block.blocktemplate["merkleroot"] )) 
        block.hashMerkleRoot = merkleRoot

        #Update internal copy of block and params
        self.block = block
        self.param = param 

    def gpu_sieving( self, level, gpu_idx, config, candidates, verbose ):

        #Clean input/output files
        subprocess.run( "rm -rf output"+ str(gpu_idx) + ".txt" , capture_output=True, shell=True )
        subprocess.run( "rm -rf input" + str(gpu_idx) + ".txt"  , capture_output=True, shell=True )
        
        #Create input file for GPU work
        with open("input"+str(gpu_idx)+".txt", 'w') as inputfile:
            inputfile.write( "\n".join( [ str(idx) + " " + str(n) for idx, n in enumerate( candidates ) ] ) )
            inputfile.close()

        #Create a copy of the config file for this running instance
        output = "output"+str(gpu_idx)+".txt"
        config_name = "gpu_config_" + str(level) + "_" + str(gpu_idx) + ".txt"

        start = time()
        parse = subprocess.run( "CUDA_VISIBLE_DEVICES="+str(gpu_idx) + " ./ecmongpu/build/bin/cuda-ecm -c " + config_name, capture_output=True, shell=True )
        endf = time()
        if parse.returncode != 0:
            return [ -1 ]
            
        survivors = []
        solutions = []
        index = 0
        block = self.block
        with open(output, 'r') as outputfile:
            read           = [ line for line in outputfile if "DONE" not in line ] 
            survivor_index = [ int(g[0]) for line in read if (g:= line.split())[1] == "1"  ]
            survivors      = [ candidates[index] for index in survivor_index ]
            solution_index = [ ( int(g[0]), int(g[1]) ) for line in read if  len( g := line.split() ) == 3 and  int(g[1]).bit_length() == ( block.nBits//2 + (block.nBits&1)) ]
            solutions      = [ candidates[ index[0] ] for index in solution_index ]
            
            #Check if we got lucky and found a solution
            for i_p, n in zip(solution_index, solutions ):
                tp,tq = int(i_p[1]), int(n)//int(i_p[1])
                p = min(tp,tq)
                q = max(tp,tq)
                n = p*q

                if p.bit_length() == ( block.nBits//2 + (block.nBits&1)) and ( isprime(p) == isprime(q) ) == True : 
                     #Update values for the found block
                     block.nP1     = IntToUint1024(p)
                     block.nNonce  = nonce
                     block.wOffset = n - W

                     #Compute the block hash
                     block_hash = block.compute_raw_hash()

                     #Update block
                     block._hash = block_hash

                     print(" Height: ", block.blocktemplate["height"] )   
                     print("      N: ", n)
                     print("      W: ", W)
                     print("      P: ", p)
                     print("  Nonce: ", nonce)
                     print("wOffset: ", n - W)
                     print("Total Block Mining Runtime: ", time() - START, " Seconds." )

                     return block

        if verbose:
            print("GPU ID " + str(gpu_idx) + " Total Time: ", time() - start, " Seconds.   Survivors: ", len( survivors )  )
        
        return survivors    
   
    def mine(self, scriptPubKey = None, verbose = False):
        #Start measing time for mining this block 
        START = time()

        #Prepare block for mining
        self.scriptPubKey=scriptPubKey
        self.prepare_block()

        #Get random nonce
        self.block.nNonce = secrets.randbelow( 1 << 64 ) 

        #Get the W
        W = gHash(self.block,self.param)
        W = uint1024ToInt(W)

        #Compute limit range around W
        wInterval = 16 * self.block.nBits 
        wMAX = int(W + wInterval)
        wMIN = int(W - wInterval)         
        
        if verbose:
            print("Total Number of candidates:", 2*wInterval)

        #Candidates for admissible semiprime
        L1 = sympy.primorial( 1 << 6, False )
        candidates = [ n for n in range( wMIN, wMAX) if gcd( n, L1 ) == 1  ]
        L2 = sympy.primorial( 1 << 10, False )
        L3 = sympy.primorial( 1 << 15, False )//(L1*L2)
        L2 = L2//L1
        candidates = [ n for n in candidates         if gcd( n, L2 ) == 1  ] 
        candidates = [ n for n in candidates         if gcd( n, L3 ) == 1  ] 
        
        if verbose:
            print("      Surviving candidates for small sieve:", len(candidates))

        #Make sure the candidates have exactly nBits as required by this block
        candidates = [ k for k in candidates if k.bit_length() == self.block.nBits ] #This line requires python >= 3.10
        candidates = [ k for k in candidates if not isprime(k)                     ]
        
        if verbose:
            print("Surviving candidates after removing primes:", len(candidates))

        #############################################################################################
        #                                       GPU Filtering                                       #
        #############################################################################################
        deviceCount = nvidia_smi.nvmlDeviceGetCount()

        #Clean input/output files
        subprocess.run( "rm -rf output*.txt" , capture_output=True, shell=True )
        subprocess.run( "rm -rf input*.txt"  , capture_output=True, shell=True )
        levels = sys.argv[2]

        for level in range(levels):
            for idx in range( deviceCount ):
                config_name = "gpu_config_" + str(level) + "_" + str(idx) + ".txt"
                create_config_file( level, idx, config_name )
 
        for level in range(levels):
                cand_size = len(candidates)//deviceCount
                gpu_inputs = [ ( level, gpu_idx, config[level], candidates[gpu_idx*cand_size: min( (gpu_idx+1)*cand_size, len(candidates) ) ],  True )   for gpu_idx in range( deviceCount)  ]
    
                with concurrent.futures.ThreadPoolExecutor( max_workers = deviceCount ) as executor:
                    future       = [ executor.submit( self.gpu_sieving, *indata ) for indata in gpu_inputs ]
                    return_value = [ future[idx].result() for idx in range( len(future))  ]
                    return_value = [ x for xs in return_value for x in xs ]
                    if any([ a == -1 for a in return_value] ):
                        return None
                    
                    candidates   = return_value 
                    print( "Total Survivors:", len(return_value) )
            
        #############################################################################################

        block = self.block
        for idx,cand in enumerate( candidates):
            if rpc_getblockcount() >= block.blocktemplate["height"]:
                print("Race was lost. Next block.")
                print("Total Block Mining Runtime: ", time() - START, " Seconds." )
                return None

            #Note: the block requires the smaller of the two prime factors to be submitted.
            #By default, cypari2 lists the factors in ascending order so choose the first factor listed. 
            fstart  = time()
            
            run_command = "./cado-nfs/build/cado-nfs.py " + str(cand) 
            print(run_command)
            try:
                startf = time()
                PARSE = subprocess.run( run_command, capture_output=True, shell=True  )
                endf = time()
            except Exception as e:
                print(e)
                continue

            parse = PARSE.stdout.decode('utf-8').split( ) 
            flag=False
            for line in parse:
                print("Factorization", line, flush=True)

            #Check if there are any winners in this batch
            factorData = []
            print("  Candidate: ", str(idx) +"/" + str(len(candidates)), "Factoring Time: ", endf - startf, flush=True )
            if len(parse) == 2:
                tp,tq = [ int(a) for a in parse ] 
                p = min(tp,tq)
                q = max(tp,tq)
                n = p*q
                print("|p1|_2=",p.bit_length(),"|p2|_2=",q.bit_length(), "|n|_2",n.bit_length())
                if ( p.bit_length() ==  ( block.nBits//2 + (block.nBits&1)) ):
                        if( (isprime(p) == isprime(q)) == True ):
                            factorData.append( [n,p,q] )
            else:
                continue

            for solution in factorData:
                solution.sort()
                factors = [ solution[0], solution[1] ]
                n = solution[2]

                #Update values for the found block
                block.nP1     = IntToUint1024(factors[0])
                block.nNonce  = nonce
                block.wOffset = n - W

                #Compute the block hash
                block_hash = block.compute_raw_hash()

                #Update block
                block._hash = block_hash

                print(" Height: ", block.blocktemplate["height"] )   
                print("      N: ", n)
                print("      W: ", W)
                print("      P: ", factors[0])
                print("  Nonce: ", nonce)
                print("wOffset: ", n - W)
                print("Total Block Mining Runtime: ", time() - START, " Seconds." )

                return block
		
def getParams():
    param = CParams()
    param.hashRounds = 1
    param.MillerRabinRounds = 50
    return param

def mine():
    if len(sys.argv) != 3:
        print("Usage: python blockchain.py <ScriptPubKey> <ECM Level>")
        print("  ScriptPubKey is a hexadecimal string and ECM Level is an integer from 1 to 5.")
        print("         ECM Levels")
        print("         Level 1:    Remove factors with up to 20 decimal digits.")
        print("         Level 2:    Remove factors with up to 25 decimal digits.")
        print("         Level 3:    Remove factors with up to 30 decimal digits.")
        print("         Level 4:    Remove factors with up to 35 decimal digits")
        print("         Level 5:    Remove factors with up to 40 decimal digits.")
        print("         Level 5:    Remove factors with up to 45 decimal digits.")
        print("  Choose the appropiate one for the computational power in GPU you have." )
        print()
        sys.exit(1)

    scriptPubKey = sys.argv[1]
    while True:
        B = CBlock()
        block = B.mine( scriptPubKey, True )
        if block:
            block.rpc_submitblock()

if __name__ == "__main__":
    mine()
