CXX=g++ 
LDFLAGS=-lglfw  -lGL -ldl \
-L/opt/cuda/targets/x86_64-linux/lib -lcudart -lcuda \
-lnvoptix
CFLAGS=-I/opt/cuda/targets/x86_64-linux/include \
-I/home/martin/optix/include \
-march=native #-ggdb -fsanitize=address

CXXFLAGS=-ffast-math -std=gnu++1z -O0 $(CFLAGS)

CFILES:=$(shell find .| grep .cpp$)
CUFILES:=$(shell ls | grep .cu$)
OBJ:=$(CFILES:%.cpp=%.o)
CUOBJ:=$(CUFILES:%.cu=%.o)
HFILES:=$(CFILES:%.cpp=%.h)
PROTOS:=$(patsubst %, proto_%, $(HFILES))

all:  ptx_code.c vis


# FindCUDA.cmake
# dont use -G or -g for optix
# -Xcompiler=-ggdb -g
%.ptx: %.cu
	/opt/cuda/bin/nvcc -c --std=c++14 -O3 -Xcompiler=-march=native   --compiler-bindir=/usr/x86_64-pc-linux-gnu/gcc-bin/8.3.0 -I/home/martin/optix/include --ptx -o $@ $< 

# configure_optix.cmake
ptx_code.c: vis_05_cuda_device_programs.ptx
	bin2c -c --padd 0 --type "char" --name ptx_code $< > $@

ptx_code.o: ptx_code.c
	gcc $(CFLAGS) -c -o $@ $^ -fpic #--no-pic --no-pie

%.o: %.cpp globals.h 
	$(CXX) -c -o $@ $< $(CXXFLAGS)


vis: $(OBJ) ptx_code.o
	$(CXX) -rdynamic -o $@ $^ $(CXXFLAGS) $(LDFLAGS)



genclean:
	rm *.cpp *.o globals.h utils.h proto2.h vis *.s imgui/*.o 

clean:
	rm *.o vis *.s imgui/*.o  *.ptx ptx_code.c

