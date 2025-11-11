
NVCC        = nvcc
NVCC_FLAGS  = -O3 -I/usr/local/cuda/include `pkg-config --cflags opencv4`
LD_FLAGS    = -lcudart -L/usr/local/cuda/lib64 `pkg-config --libs opencv4`
EXE	        = imgProcess
OBJ	        = main.o 


default: $(EXE)

main.o: main.cu gaussian_kernel.cu sobel_kernel.cu readPgm.c writePgm.c
	$(NVCC) -c -o $@ main.cu $(NVCC_FLAGS)



$(EXE): $(OBJ)
	$(NVCC) $(OBJ) -o $(EXE) $(LD_FLAGS)

clean:
	rm -rf *.o $(EXE)
	rm -rf *t.jpg