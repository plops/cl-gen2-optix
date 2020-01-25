CXX=g++ 
LDFLAGS=-lglfw  -lGL \
-L/opt/cuda/targets/x86_64-linux/lib -lcudart
CXXFLAGS=-ffast-math -std=gnu++1z -O0 \
-I/opt/cuda/targets/x86_64-linux/include \
-I/home/martin/optix/include \
-march=native #-ggdb 
# -fsanitize=address

CFILES:=$(shell find .| grep .cpp$)
CUFILES:=$(shell ls | grep .cu$)
OBJ:=$(CFILES:%.cpp=%.o)
CUOBJ:=$(CUFILES:%.cu=%.o)
HFILES:=$(CFILES:%.cpp=%.h)
PROTOS:=$(patsubst %, proto_%, $(HFILES))
%.o: %.cpp globals.h
	$(CXX) -c -o $@ $< $(CXXFLAGS)

%.o: %.cu
	/opt/cuda/bin/nvcc -c --std=c++14 -O3 -g -Xcompiler=-march=native -Xcompiler=-ggdb  --compiler-bindir=/usr/x86_64-pc-linux-gnu/gcc-bin/8.3.0 -o $@ $< 


all: vis
vis: $(OBJ) #$(CUOBJ)
	$(CXX) -o $@ $^ $(CXXFLAGS) $(LDFLAGS)



genclean:
	rm *.cpp *.o globals.h utils.h proto2.h vis *.s imgui/*.o 

clean:
	rm *.o vis *.s imgui/*.o 

