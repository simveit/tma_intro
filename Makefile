NVCC_FLAGS = -std=c++17 -O3 -DNDEBUG -w
NVCC_LDFLAGS = -lcublas -lcuda
OUT_DIR = out

CUDA_OUTPUT_FILE = -o $(OUT_DIR)/$@
NCU_PATH := $(shell which ncu)
NCU_COMMAND = sudo $(NCU_PATH) --set full --import-source yes

NVCC_FLAGS += --expt-relaxed-constexpr --expt-extended-lambda --use_fast_math -Xcompiler=-fPIE -Xcompiler=-Wno-psabi -Xcompiler=-fno-strict-aliasing
NVCC_FLAGS += -arch=sm_90a

NVCC_BASE = nvcc $(NVCC_FLAGS) $(NVCC_LDFLAGS) -lineinfo

intro: intro.cu 
	$(NVCC_BASE) $^ $(CUDA_OUTPUT_FILE)

intro_swizzle: intro_swizzle.cu 
	$(NVCC_BASE) $^ $(CUDA_OUTPUT_FILE)

compile_all: 
	make intro
	make intro_swizzle

run_all: compile_all
	./$(OUT_DIR)/intro
	./$(OUT_DIR)/intro_swizzle

clean:
	rm $(OUT_DIR)/*