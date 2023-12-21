# GPUDispersion

This version uses GPU to help eliminate candidates at scale much faster.

Roughly speaking, the situation is this: we get an interval of about 12,000 integers to sieve per nonce. Multiples of 2,3,5 and 7 reduces
this amount of 12K candidates to by 79% to 2500 candidates. Next we sieve by multiples of 11, 13, and so on. We remove candidates by checking
if they are a multiple of ever larger primes. 

In all previous testing of ECM on GPUs we did not see any advantage over CPU for this task of sieving until recently when someone in the team
investigating this stumble upon the fact that, if we remove phase 2 from ECM, then we get a huge speed up and can sieve by much larger primes
in the same amount of time we perviously had available.

In practice, when before we tested and reduced 12000 candidates to ~350 now, in that same amount of time, we hav left ~100 or so candidates. If
we add three GPUs of the same type this number comes down to ~60 or so candidates. We tested with 14 RTX 4090 GPUs and this number comes down to
~20 candidates. 

Where now we can perform GNFS using CUDA-NFS on the remaining 100 or 20 survivors of this proccess -- depending on how much GPU power your system
has available.

This miner automagically uses ALL gpus available on the system -- though, only nvidia GPUs. The ECM GPU implementation we use is here[https://github.com/FACT0RN/ecmongpu]. 
Feel free to improve it.

# Installation

You will need the following python modules:

``` pip install numpy base58 nvidia-ml-py3 ```

To build ecmongpu you will the cuda toolkit. We will publish a release built for x86 and ubuntu.

# Running

Once you have CUDA-NFS and ecmongpu ready to go -- run the ``build.sh`` script. You will need to set the ``RPC_USER`` and ``RPC_PASS`` in the environment so the miner can talk to the daemon.

```
export RPC_USER=youruserforfactornd
export RPC_PASS=yourpasswordforfactornd
```

Then to run, you will need your ScriptPubKey and the ecm level to run. For example:

```
python blockchain.py 00scriptpubkeygoeshereff 4
```

The ECM levels are as follows:

```
Usage: python blockchain.py <ScriptPubKey> <ECM Level>")
       ScriptPubKey is a hexadecimal string and ECM Level is an integer from 1 to 5.
         ECM Levels
         Level 1:    Remove factors with up to 20 decimal digits.
         Level 2:    Remove factors with up to 25 decimal digits.
         Level 3:    Remove factors with up to 30 decimal digits.
         Level 4:    Remove factors with up to 35 decimal digits.
         Level 5:    Remove factors with up to 40 decimal digits.
        Level 5:    Remove factors with up to 45 decimal digits.
 Choose the appropiate one for the computational power in GPU you have
```

The higher the level the more you sieve but the longer it takes. The number of CUDA cores you have the higher you can for a fixed amount of time.

So, for example if CUDA-NFS takes 4 minutes to factor a block solver candidate and the average block time is 10 minutes...you should now take more
than 6 minuets on GPU sieving -- choose a level that takes less than that to leave yourselve some buffer room for running CADO-NFS.


# Improvements

Please report any bugs or funny behaviour, in terms of timing or not sieving as much as expected given a number of GPus, and we will take a look at it.
Scaling should be linear in terms of GPUs.






