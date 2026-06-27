# =============================================================================
# Configuration
# =============================================================================
SHELL := /bin/bash

# MPI host compiler (used for C++ TUs and as nvcc -ccbin so <mpi.h> resolves).
CXX  := $(shell which mpic++ 2>/dev/null)
ifeq ($(CXX),)
  CXX := mpic++
endif
NVCC := $(shell which nvcc 2>/dev/null)

# --- Repo paths ---
CP_ROOT     := .
CP_DIR      := context_parallel
PG_DIR      := process_group
DL_DIR      := Data_Loader
SCRIPTS_DIR := Scripts/BluTrain

OBJDIR := build/objects
BINDIR := build

# Main training entrypoint + resulting executable.
MAIN_SRC := $(SCRIPTS_DIR)/gpt2_cp_test.cpp
TARGET   := $(BINDIR)/gpt2_cp_test_exec

# --- Tensor-Implementations submodule (provides libtensor) ---
TENSOR_DIR    := Tensor-Implementations
TENSOR_INC    := $(TENSOR_DIR)/include
TENSOR_LIBDIR := $(TENSOR_DIR)/lib
TENSOR_LIB_A  := $(TENSOR_LIBDIR)/libtensor.a
TENSOR_LIB_SO := $(TENSOR_LIBDIR)/libtensor.so

# =============================================================================
# CUDA auto-detection
# =============================================================================
ifeq ($(NVCC),)
  $(error nvcc not found. CUDA toolkit is not installed or not in PATH)
endif

# Evaluate nvidia-smi at most once (a recursive `?=` would re-run it on every
# reference to NVCCFLAGS).
ifeq ($(origin SM_ARCH),undefined)
  SM_ARCH := $(shell nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d '.')
endif
ifeq ($(strip $(SM_ARCH)),)
  SM_ARCH := 86
endif

CUDA_ROOT := $(shell dirname $(shell dirname $(NVCC)))
CUDA_INC  := $(CUDA_ROOT)/include
CUDA_LIB  := $(CUDA_ROOT)/lib64

CUDA_TARGETS_LIB := $(shell find $(CUDA_ROOT)/targets -type d -name "lib" 2>/dev/null | head -1)
ifeq ($(CUDA_TARGETS_LIB),)
  CUDA_TARGETS_LIB := $(CUDA_LIB)
endif

# MPI library directory (best-effort; from the mpic++ wrapper).
MPI_LIBDIR := $(shell $(CXX) --showme:libdirs 2>/dev/null | tr ' ' '\n' | head -1)

# =============================================================================
# Compiler / Linker Flags
# =============================================================================
INCLUDES := -I$(CP_ROOT) -I$(SCRIPTS_DIR) -I$(TENSOR_INC) -I$(CUDA_INC) -DWITH_CUDA

CXXFLAGS  := -std=c++17 -fPIC -O3 -g $(INCLUDES)
NVCCFLAGS := -std=c++17 -Xcompiler="-fPIC" -arch=sm_$(SM_ARCH) -O3 -g \
             --use_fast_math --expt-relaxed-constexpr -ccbin=$(CXX) $(INCLUDES)

LINKFLAGS := -std=c++17 -Xcompiler="-fPIC" -arch=sm_$(SM_ARCH) -O3 -g -ccbin=$(CXX)

LDFLAGS := -L$(CUDA_LIB) -L$(CUDA_TARGETS_LIB) -L$(TENSOR_LIBDIR)
ifneq ($(MPI_LIBDIR),)
  LDFLAGS += -L$(MPI_LIBDIR) -Xlinker -rpath -Xlinker $(MPI_LIBDIR)
endif
LDFLAGS += -Xlinker -rpath -Xlinker '$$ORIGIN/$(TENSOR_LIBDIR)'

# NVTX on CUDA 12+ is header-only -> no -lnvToolsExt.
LDLIBS := -lmpi -lnccl -lcudart -lcublas -lcublasLt -lcuda -lcurand \
          -lgomp -lstdc++ -lnvidia-ml -lz -lpthread -ldl

# Keep the link from being OOM-killed on the large -O3 training TU.
LINK_MEMFLAGS := -Xlinker --no-keep-memory -Xlinker --reduce-memory-overheads

# =============================================================================
# Source / Object Discovery
#   - All CP-owned .cu under context_parallel/ and process_group/.
#   - All CP-owned .cpp under process_group/ (e.g. processGroupNCCL.cpp).
#   - Exclude alternates / standalone test files via CU_EXCLUDE / CPP_EXCLUDE.
#   - Data_Loader/dl_test.cpp and the main TU are NOT discovered here
#     (dl_test.cpp is #included directly by the main TU).
# =============================================================================
# sm89 attention variants are arch-specific alternates of the default WMMA
# kernels; exclude them from the default build to avoid duplicate symbols.
CU_EXCLUDE := \
    $(CP_DIR)/AttentionForward_sm89.cu \
    $(CP_DIR)/AttentionBackward_sm89.cu

CPP_EXCLUDE := \
    $(PG_DIR)/test.cpp

CU_SOURCES := $(filter-out $(CU_EXCLUDE), \
                $(shell find $(CP_DIR) $(PG_DIR) -name '*.cu' 2>/dev/null))
CPP_SOURCES := $(filter-out $(CPP_EXCLUDE), \
                $(shell find $(PG_DIR) -name '*.cpp' 2>/dev/null))

OBJECTS_FROM_CU  := $(patsubst %.cu,$(OBJDIR)/%.o,$(CU_SOURCES))
OBJECTS_FROM_CPP := $(patsubst %.cpp,$(OBJDIR)/%.o,$(CPP_SOURCES))
MAIN_OBJ         := $(patsubst %.cpp,$(OBJDIR)/%.o,$(MAIN_SRC))

ALL_OBJECTS := $(OBJECTS_FROM_CPP) $(OBJECTS_FROM_CU) $(MAIN_OBJ)

# =============================================================================
# Top-level Targets
# =============================================================================
.PHONY: all libtensor run run-snippet run-folder \
        clean rebuild print_sm help

all: $(TARGET)
	@echo -e "\n\nBuild complete: $(TARGET)"

# =============================================================================
# Link the context-parallel training binary
# =============================================================================
$(TARGET): $(ALL_OBJECTS) | $(TENSOR_LIB_A)
	@echo -e "\n--- Linking executable (sm_$(SM_ARCH)): $@"
	@mkdir -p $(BINDIR)
	$(NVCC) $(LINKFLAGS) $(LDFLAGS) -o $@ $(ALL_OBJECTS) \
	        -Xlinker --start-group $(TENSOR_LIB_A) -Xlinker --end-group \
	        $(LDLIBS) $(LINK_MEMFLAGS)
	@echo -e "[SUCCESS] $@\n\nRun with: make run NP=2"

# C++ objects (host / MPI)
$(OBJDIR)/%.o: %.cpp
	@mkdir -p $(@D)
	@echo "Compiling [CXX  | cp]: $<"
	$(CXX) $(CXXFLAGS) -c $< -o $@

# CUDA objects
$(OBJDIR)/%.o: %.cu
	@mkdir -p $(@D)
	@echo "Compiling [CUDA | cp]: $<"
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

# =============================================================================
# libtensor -- delegate to the Tensor-Implementations submodule Makefile
# =============================================================================
libtensor:
	@echo "--- Building libtensor in $(TENSOR_DIR) (SM_ARCH=$(SM_ARCH)) ---"
	$(MAKE) -C $(TENSOR_DIR) tensor SM_ARCH=$(SM_ARCH)

$(TENSOR_LIB_A):
	@if [ ! -f $(TENSOR_LIB_A) ]; then \
		echo -e "\n[ERROR] $(TENSOR_LIB_A) not found."; \
		echo    "        Build the submodule first:  make libtensor"; \
		echo -e "        (or: make -C $(TENSOR_DIR) tensor)\n"; \
		exit 1; \
	fi

# =============================================================================
# Run helpers
# =============================================================================
# Number of ranks for mpirun (override: make run NP=4). Extra args via ARGS=.
NP ?= 2

run: $(TARGET)
	@echo "--- Running $(TARGET) on $(NP) rank(s) ---"
	LD_LIBRARY_PATH=$(TENSOR_LIBDIR):$$LD_LIBRARY_PATH \
	    mpirun -np $(NP) ./$(TARGET) $(ARGS)

# Compile + run a single standalone .cpp against libtensor, then keep the binary
# in $(BINDIR). Usage: make run-snippet FILE=path/to/file.cpp [NP=2]
run-snippet: | $(TENSOR_LIB_A)
	@if [ -z "$(FILE)" ]; then \
		echo "ERROR: specify a file.  Usage: make run-snippet FILE=path/to/file.cpp [NP=2]"; \
		exit 1; \
	fi
	@mkdir -p $(BINDIR)
	@echo "--- Compiling (sm_$(SM_ARCH)): $(FILE) ---"
	$(NVCC) $(NVCCFLAGS) $(LDFLAGS) -o $(BINDIR)/snippet_runner $(FILE) \
	        -Xlinker --start-group $(TENSOR_LIB_A) -Xlinker --end-group \
	        $(LDLIBS) $(LINK_MEMFLAGS)
	@echo -e "\n--- Running snippet_runner ---"
	LD_LIBRARY_PATH=$(TENSOR_LIBDIR):$$LD_LIBRARY_PATH \
	    mpirun -np $(NP) ./$(BINDIR)/snippet_runner $(ARGS)

# Compile + run every .cpp in a folder. Usage: make run-folder FOLDER=dir [NP=2]
run-folder: | $(TENSOR_LIB_A)
	@if [ -z "$(FOLDER)" ]; then \
		echo "ERROR: specify a folder.  Usage: make run-folder FOLDER=dir [NP=2]"; \
		exit 1; \
	fi
	@mkdir -p $(BINDIR)
	@for file in $(FOLDER)/*.cpp; do \
		[ -f "$$file" ] || continue; \
		echo -e "\n---------------------------------------------------------"; \
		echo "Running: $$file"; \
		echo    "---------------------------------------------------------"; \
		$(NVCC) $(NVCCFLAGS) $(LDFLAGS) -o $(BINDIR)/snippet_runner "$$file" \
		        -Xlinker --start-group $(TENSOR_LIB_A) -Xlinker --end-group \
		        $(LDLIBS) $(LINK_MEMFLAGS) || exit 1; \
		LD_LIBRARY_PATH=$(TENSOR_LIBDIR):$$LD_LIBRARY_PATH \
		    mpirun -np $(NP) ./$(BINDIR)/snippet_runner $(ARGS) || exit 1; \
	done
	@rm -f $(BINDIR)/snippet_runner
	@echo -e "\nAll tests in $(FOLDER) passed!"

# =============================================================================
# Clean / Rebuild / Utility
# =============================================================================

clean:
	@echo "--- Cleaning CP build artifacts ---"
	rm -rf $(OBJDIR) $(TARGET) $(BINDIR)/snippet_runner
	@echo "--- Removing stray in-tree object files ---"
	find $(CP_DIR) $(PG_DIR) $(SCRIPTS_DIR) -name '*.o' -delete 2>/dev/null || true

rebuild:
	@$(MAKE) clean && $(MAKE) all

print_sm:
	@echo "------Detected Compute Capability: $(SM_ARCH)------"

help:
	@echo ""
	@echo "Available targets:"
	@echo "  make [all]                   Build the CP training binary ($(TARGET))"
	@echo "  make libtensor               Build libtensor in the Tensor-Implementations submodule"
	@echo "  make run [NP=2] [ARGS=...]   mpirun the CP training binary on NP ranks"
	@echo "  make run-snippet FILE=<f>    Compile + run a single .cpp against libtensor"
	@echo "  make run-folder  FOLDER=<d>  Compile + run every .cpp in a folder"
	@echo "  make rebuild                 clean + all"
	@echo "  make clean                   Remove CP build artifacts (not libtensor)"
	@echo "  make print_sm                Print detected GPU compute capability"
	@echo ""
	@echo "Flags:"
	@echo "  SM_ARCH=<nn>                 Override GPU SM arch (default: auto-detected = $(SM_ARCH))"
	@echo "  NP=<n>                       mpirun rank count for run targets (default: 2)"
	@echo "  ARGS=\"...\"                   Extra args passed to the binary"
	@echo ""
	@echo "Output:"
	@echo "  $(TARGET)"
	@echo ""
