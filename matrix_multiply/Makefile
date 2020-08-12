all: naive_mm.out
NVCC = nvcc
NVCC_FLAGS = -I../include

ifeq ($(dbg),1)
	NVCC_FLAGS += -g -O0
else
	NVCC_FLAGS += -O3
endif

naive_mm.out: naive_mm.cu
	$(NVCC) $(NVCC_FLAGS) $^ -o $@


clean:
	rm naive_mm.out

