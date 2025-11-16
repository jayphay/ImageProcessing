# =========================================================
#  Makefile for CUDA Image Processing Project
#  Manual OpenCV include/link paths (no pkg-config needed)
# =========================================================

NVCC        := nvcc
# Suppress harmless OpenCV and fscanf warnings
CXXFLAGS := -O2 -std=c++11 -diag-suppress=611
CUDAFLAGS   := -arch=sm_50      # Change to your GPU's capability (sm_61, sm_75, sm_86, etc.)

# ---- OpenCV Include/Lib Paths ----
OPENCV_INC  := -I/usr/include/opencv4
OPENCV_LIBS := -lopencv_core -lopencv_imgproc -lopencv_highgui -lopencv_imgcodecs

# ---- Source Files ----
SRC := main.cu
OBJ         := $(SRC:.cu=.o)
OBJ         := $(OBJ:.c=.o)
TARGET      := imgProcess

# ---- Build Rules ----
all: $(TARGET)

$(TARGET): $(OBJ)
	@echo "Linking target: $(TARGET)"
	$(NVCC) $(CXXFLAGS) $(OBJ) -o $(TARGET) $(OPENCV_INC) $(OPENCV_LIBS)

%.o: %.cu
	@echo "Compiling CUDA source: $<"
	$(NVCC) $(CXXFLAGS) $(CUDAFLAGS) $(OPENCV_INC) -c $< -o $@

%.o: %.c
	@echo "Compiling C source: $<"
	gcc -O2 -c $< -o $@

run: $(TARGET)
	./$(TARGET) dog_img.jpg

clean:
	rm -f *.o $(TARGET) blur_output_*.jpg edge_output_*.jpg

info:
	@echo "CUDA compiler: $(NVCC)"
	@echo "Include path:  $(OPENCV_INC)"
	@echo "Linked libs:   $(OPENCV_LIBS)"
