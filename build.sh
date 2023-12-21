git clone https://gitlab.inria.fr/cado-nfs/cado-nfs.git
cd cado-nfs
mkdir build
cd build
cmake ..
make -j 4
cd ../../
git clone https://github.com/FACT0RN/ecmongpu.git 
cd ecmongpu
mkdir build
cd build
cmake ..
make -j4

