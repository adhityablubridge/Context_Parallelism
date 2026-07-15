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
SCRIPTS_DIR := Scripts/Blutrain

OBJDIR := build/objects
BINDIR := build

# Main training entrypoint + resulting executable.
MAIN_SRC := $(SCRIPTS_DIR)/gpt2_cp_test.cpp
TARGET   := $(BINDIR)/gpt2_cp_test_exec

# --- Tensor-Implementations (provides libtensor) ---
# Now nested under the BluTrain monorepo. libtensor itself links -lprofiler.
TENSOR_DIR    := BluTrain/Tensor-Implementations
TENSOR_INC    := $(TENSOR_DIR)/include
TENSOR_LIBDIR := $(TENSOR_DIR)/lib
TENSOR_LIB_A  := $(TENSOR_LIBDIR)/libtensor.a
TENSOR_LIB_SO := $(TENSOR_LIBDIR)/libtensor.so

# --- Profiler (provides libprofiler: AllocationTracker, Profiler, scopes) ---
# Sibling of Tensor-Implementations under BluTrain/. Has no Makefile of its own,
# so this Makefile builds libprofiler.so (pure host C++) from its sources.
PROFILER_DIR    := BluTrain/Profiler
PROFILER_INC    := $(PROFILER_DIR)/include
PROFILER_LIBDIR := $(PROFILER_DIR)/lib
PROFILER_LIB    := $(PROFILER_LIBDIR)/libprofiler.so
PROFILER_SRCS   := $(shell find $(PROFILER_DIR)/src -name '*.cpp' 2>/dev/null)

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
# -IBLUTRAIN_COMM_INC brings in the canonical (unified) ProcessGroupNCCL.h + CpuSync.hpp
# (BluTrain dist/communication) used by both DataParallel and ContextParallel.
BLUTRAIN_COMM_INC := BluTrain/dist/communication/include
BLUTRAIN_DP_INC   := BluTrain/dist/Data-Parallel/include
INCLUDES := -I$(CP_ROOT) -I$(SCRIPTS_DIR) -I$(TENSOR_INC) -I$(PROFILER_INC) -I$(CUDA_INC) -I$(BLUTRAIN_COMM_INC) -I$(BLUTRAIN_DP_INC) -DWITH_CUDA

# CP fused-RoPE kernels (context_parallel/GQA_fused_{fwd,bwd}_sm103_cp.cu) + the
# FusedRoPESDPA.h wrappers compile only under -DCP_FUSED_ROPE; the default build
# leaves those TUs empty (no libtensor symbol collision). CP_ROPE_DEBUG adds the
# out-of-range RoPE-index counter/check (bring-up only; adds a per-call sync).
#   make CP_FUSED_ROPE=1 ...            enable the CP fused-RoPE path
#   make CP_FUSED_ROPE=1 CP_ROPE_DEBUG=1 ...   + out-of-range checks
ifeq ($(CP_FUSED_ROPE),1)
  INCLUDES += -DCP_FUSED_ROPE
endif
ifeq ($(CP_ROPE_DEBUG),1)
  INCLUDES += -DCP_ROPE_DEBUG
endif

# Make the CP-RoPE macro state a real build dependency, so flipping
# CP_FUSED_ROPE / CP_ROPE_DEBUG rebuilds automatically (no manual `make clean`).
# The stamp file's contents are rewritten ONLY when the flag string changes, so
# its mtime bumps -> every object depending on it recompiles. Unchanged flags =>
# no rewrite => no spurious rebuilds.
CP_ROPE_STAMP := $(OBJDIR)/.cp_rope_flags
CP_ROPE_STATE := CP_FUSED_ROPE=$(CP_FUSED_ROPE)|CP_ROPE_DEBUG=$(CP_ROPE_DEBUG)
.PHONY: __cp_rope_flagcheck
$(CP_ROPE_STAMP): __cp_rope_flagcheck
	@mkdir -p $(@D)
	@[ -f $@ ] && [ "$$(cat $@)" = "$(CP_ROPE_STATE)" ] || echo "$(CP_ROPE_STATE)" > $@

# Automatic header-dependency tracking: -MMD emits a per-object .d file listing
# every (non-system) header the TU pulled in; -MP adds phony targets so deleting
# a header doesn't break the build. The .d files are -include'd below, so editing
# ANY header recompiles exactly the objects that include it -- no `make clean`.
DEPFLAGS := -MMD -MP

# c++2a (C++20) matches BluTrain/Makefile; checkpointing/Checkpointing.h uses
# C++20 std::string::starts_with/ends_with.
CXXFLAGS  := -std=c++2a -fPIC -O3 -g $(DEPFLAGS) $(INCLUDES)
NVCCFLAGS := -std=c++17 -Xcompiler="-fPIC" -arch=sm_$(SM_ARCH) -O3 -g \
             --use_fast_math --expt-relaxed-constexpr -ccbin=$(CXX) $(DEPFLAGS) $(INCLUDES)

LINKFLAGS := -std=c++17 -Xcompiler="-fPIC" -arch=sm_$(SM_ARCH) -O3 -g -ccbin=$(CXX)

LDFLAGS := -L$(CUDA_LIB) -L$(CUDA_TARGETS_LIB) -L$(TENSOR_LIBDIR) -L$(PROFILER_LIBDIR)
ifneq ($(MPI_LIBDIR),)
  LDFLAGS += -L$(MPI_LIBDIR) -Xlinker -rpath -Xlinker $(MPI_LIBDIR)
endif
LDFLAGS += -Xlinker -rpath -Xlinker '$$ORIGIN/$(TENSOR_LIBDIR)' \
           -Xlinker -rpath -Xlinker '$$ORIGIN/$(PROFILER_LIBDIR)'

# NVTX on CUDA 12+ is header-only -> no -lnvToolsExt.
# -lprofiler provides OwnTensor::AllocationTracker / Profiler (from BluTrain/Profiler).
LDLIBS := -lmpi -lnccl -lcudart -lcublas -lcublasLt -lcuda -lcurand \
          -lgomp -lstdc++ -lnvidia-ml -lz -lpthread -ldl -lprofiler

# Keep the link from being OOM-killed on the large -O3 training TU.
LINK_MEMFLAGS := -Xlinker --no-keep-memory -Xlinker --reduce-memory-overheads

# =============================================================================
# Source / Object Discovery
#   - All CP-owned .cu  under context_parallel/ and process_group/.
#   - All CP-owned .cpp under context_parallel/ and process_group/
#     (e.g. context_parallel/ContextParallel.cpp, process_group/processGroupNCCL.cpp).
#     Headers (LoadBalancer.h, *.h) need no rule -- they are found via -I. / -Iinclude.
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
    $(PG_DIR)/test.cpp \
    $(PG_DIR)/processGroupNCCL.cpp
# ^ CP's forked PG impl is superseded by the BluTrain canonical PG (below); the
#   header process_group/ProcessGroupNCCL.h + CpuSync_fixed.hpp stay on disk but
#   are no longer included anywhere, so nothing pulls them in.

# Canonical PG implementation (unified for CP + DP), compiled into the CP targets.
# processGroupNccl.cpp depends on the dist Error/registry infra (Error_logs.cpp).
BLUTRAIN_PG_SRC := BluTrain/dist/communication/src/processGroupNccl.cpp \
                   BluTrain/dist/Data-Parallel/src/Error_logs.cpp

CU_SOURCES := $(filter-out $(CU_EXCLUDE), \
                $(shell find $(CP_DIR) $(PG_DIR) -name '*.cu' 2>/dev/null))
CPP_SOURCES := $(filter-out $(CPP_EXCLUDE), \
                $(shell find $(CP_DIR) $(PG_DIR) -name '*.cpp' 2>/dev/null)) \
                $(BLUTRAIN_PG_SRC)

OBJECTS_FROM_CU  := $(patsubst %.cu,$(OBJDIR)/%.o,$(CU_SOURCES))
OBJECTS_FROM_CPP := $(patsubst %.cpp,$(OBJDIR)/%.o,$(CPP_SOURCES))
MAIN_OBJ         := $(patsubst %.cpp,$(OBJDIR)/%.o,$(MAIN_SRC))

ALL_OBJECTS := $(OBJECTS_FROM_CPP) $(OBJECTS_FROM_CU) $(MAIN_OBJ)

# =============================================================================
# Top-level Targets
# =============================================================================
.PHONY: all libtensor profiler run run-snippet run-folder \
        clean rebuild print_sm help

all: $(TARGET)
	@echo -e "\n\nBuild complete: $(TARGET)"

# =============================================================================
# Link the context-parallel training binary
# =============================================================================
$(TARGET): $(ALL_OBJECTS) | $(TENSOR_LIB_A) $(PROFILER_LIB)
	@echo -e "\n--- Linking executable (sm_$(SM_ARCH)): $@"
	@mkdir -p $(BINDIR)
	$(NVCC) $(LINKFLAGS) $(LDFLAGS) -o $@ $(ALL_OBJECTS) \
	        -Xlinker --start-group $(TENSOR_LIB_A) -Xlinker --end-group \
	        $(LDLIBS) $(LINK_MEMFLAGS)
	@echo -e "[SUCCESS] $@\n\nRun with: make run NP=2"

# C++ objects (host / MPI). $(CP_ROPE_STAMP) prereq => flag flips force a rebuild.
$(OBJDIR)/%.o: %.cpp $(CP_ROPE_STAMP)
	@mkdir -p $(@D)
	@echo "Compiling [CXX  | cp]: $<"
	$(CXX) $(CXXFLAGS) -c $< -o $@

# CUDA objects
$(OBJDIR)/%.o: %.cu $(CP_ROPE_STAMP)
	@mkdir -p $(@D)
	@echo "Compiling [CUDA | cp]: $<"
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

# Pull in the auto-generated header dependencies (*.d) for every built object.
# Absent on a clean tree (nothing built yet) -> the leading '-' ignores that.
-include $(shell find $(OBJDIR) -name '*.d' 2>/dev/null)

# =============================================================================
# libprofiler -- pure host-C++ shared lib (no Makefile in BluTrain/Profiler)
# =============================================================================
profiler: $(PROFILER_LIB)
	@echo "libprofiler is up-to-date: $(PROFILER_LIB)"

$(PROFILER_LIB): $(PROFILER_SRCS)
	@echo "--- Building libprofiler.so from $(PROFILER_DIR)/src ---"
	@mkdir -p $(PROFILER_LIBDIR)
	$(CXX) -std=c++17 -fPIC -O3 -shared -I$(PROFILER_INC) $(PROFILER_SRCS) -o $@

# =============================================================================
# libtensor -- delegate to the Tensor-Implementations Makefile.
# libtensor links -lprofiler, so libprofiler must exist first.
# =============================================================================
libtensor: $(PROFILER_LIB)
	@echo "--- Building libtensor in $(TENSOR_DIR) (SM_ARCH=$(SM_ARCH)) ---"
	$(MAKE) -C $(TENSOR_DIR) tensor SM_ARCH=$(SM_ARCH)

$(TENSOR_LIB_A):
	@if [ ! -f $(TENSOR_LIB_A) ]; then \
		echo -e "\n[ERROR] $(TENSOR_LIB_A) not found."; \
		echo    "        Build it first:  make libtensor   (builds libprofiler + libtensor)"; \
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
	LD_LIBRARY_PATH=$(TENSOR_LIBDIR):$(PROFILER_LIBDIR):$$LD_LIBRARY_PATH \
	    mpirun -np $(NP) ./$(TARGET) $(ARGS)

# =============================================================================
# Phase-2 de-fused RoPE CP stand-in parity test (additive; new target).
# Links the SAME CP/PG objects as the main exec (minus the main TU) + libtensor.
# Build: make cp-rope-standin    Run: make run-cp-rope-standin NP=2
# =============================================================================
STANDIN_SRC := Tests/cp_rope_standin_parity.cpp
STANDIN_OBJ := $(OBJDIR)/Tests/cp_rope_standin_parity.o
STANDIN_EXE := $(BINDIR)/cp_rope_standin_parity_exec
.PHONY: cp-rope-standin run-cp-rope-standin
cp-rope-standin: $(STANDIN_EXE)
$(STANDIN_EXE): $(STANDIN_OBJ) $(OBJECTS_FROM_CPP) $(OBJECTS_FROM_CU) | $(TENSOR_LIB_A) $(PROFILER_LIB)
	@mkdir -p $(BINDIR)
	@echo -e "\n--- Linking stand-in parity test (sm_$(SM_ARCH)): $@"
	$(NVCC) $(LINKFLAGS) $(LDFLAGS) -o $@ \
	        $(STANDIN_OBJ) $(OBJECTS_FROM_CPP) $(OBJECTS_FROM_CU) \
	        -Xlinker --start-group $(TENSOR_LIB_A) -Xlinker --end-group \
	        $(LDLIBS) $(LINK_MEMFLAGS)

run-cp-rope-standin: $(STANDIN_EXE)
	@echo "--- Running stand-in parity on $(NP) rank(s) ---"
	LD_LIBRARY_PATH=$(TENSOR_LIBDIR):$(PROFILER_LIBDIR):$$LD_LIBRARY_PATH \
	    mpirun -np $(NP) ./$(STANDIN_EXE) $(ARGS)

# ---- Phase 3 FUSED RoPE kernel parity (single GPU; needs CP_FUSED_ROPE=1). ----
# Build: make CP_FUSED_ROPE=1 CP_ROPE_DEBUG=1 cp-rope-fused
# Run:   make run-cp-rope-fused   (single process, no MPI ring)
FUSED_SRC := Tests/cp_rope_fused_parity.cpp
FUSED_OBJ := $(OBJDIR)/Tests/cp_rope_fused_parity.o
FUSED_EXE := $(BINDIR)/cp_rope_fused_parity_exec
.PHONY: cp-rope-fused run-cp-rope-fused
cp-rope-fused: $(FUSED_EXE)
$(FUSED_EXE): $(FUSED_OBJ) $(OBJECTS_FROM_CPP) $(OBJECTS_FROM_CU) | $(TENSOR_LIB_A) $(PROFILER_LIB)
	@mkdir -p $(BINDIR)
	@echo -e "\n--- Linking fused RoPE parity test (sm_$(SM_ARCH)): $@"
	$(NVCC) $(LINKFLAGS) $(LDFLAGS) -o $@ \
	        $(FUSED_OBJ) $(OBJECTS_FROM_CPP) $(OBJECTS_FROM_CU) \
	        -Xlinker --start-group $(TENSOR_LIB_A) -Xlinker --end-group \
	        $(LDLIBS) $(LINK_MEMFLAGS)

run-cp-rope-fused: $(FUSED_EXE)
	@echo "--- Running fused RoPE parity (single GPU) ---"
	LD_LIBRARY_PATH=$(TENSOR_LIBDIR):$(PROFILER_LIBDIR):$$LD_LIBRARY_PATH \
	    ./$(FUSED_EXE) $(ARGS)

# ---- bluscriptCP: context-parallel Llama training (DataParallel + ContextParallel
#      on the unified PG). Ring path needs CP_FUSED_ROPE=1.
# Build: make CP_FUSED_ROPE=1 bluscript-cp    Run: make CP_FUSED_ROPE=1 run-bluscript-cp NP=2
BLUSCRIPT_SRC := Scripts/Blutrain/bluscriptCP.cpp
BLUSCRIPT_OBJ := $(OBJDIR)/Scripts/Blutrain/bluscriptCP.o
BLUSCRIPT_EXE := $(BINDIR)/bluscriptCP_exec
# DataParallel + its profiler dep — scoped to this target only (parity tests don't need them).
BLUSCRIPT_DIST_SRCS := BluTrain/dist/Data-Parallel/src/DataParallel.cpp \
                       BluTrain/dist/Data-Parallel/src/profiler.cpp
BLUSCRIPT_DIST_OBJ  := $(patsubst %.cpp,$(OBJDIR)/%.o,$(BLUSCRIPT_DIST_SRCS))
.PHONY: bluscript-cp run-bluscript-cp
bluscript-cp: $(BLUSCRIPT_EXE)
$(BLUSCRIPT_EXE): $(BLUSCRIPT_OBJ) $(BLUSCRIPT_DIST_OBJ) $(OBJECTS_FROM_CPP) $(OBJECTS_FROM_CU) | $(TENSOR_LIB_A) $(PROFILER_LIB)
	@mkdir -p $(BINDIR)
	@echo -e "\n--- Linking bluscriptCP (sm_$(SM_ARCH)): $@"
	$(NVCC) $(LINKFLAGS) $(LDFLAGS) -o $@ \
	        $(BLUSCRIPT_OBJ) $(BLUSCRIPT_DIST_OBJ) $(OBJECTS_FROM_CPP) $(OBJECTS_FROM_CU) \
	        -Xlinker --start-group $(TENSOR_LIB_A) -Xlinker --end-group \
	        $(LDLIBS) $(LINK_MEMFLAGS)

run-bluscript-cp: $(BLUSCRIPT_EXE)
	@echo "--- Running bluscriptCP on $(NP) rank(s) ---"
	LD_LIBRARY_PATH=$(TENSOR_LIBDIR):$(PROFILER_LIBDIR):$$LD_LIBRARY_PATH \
	    mpirun -np $(NP) ./$(BLUSCRIPT_EXE) $(ARGS)

# =============================================================================
# Ulysses (DeepSpeed-style) CP attention parity test (additive; new target).
# Links the SAME CP/PG objects as the main exec (minus the main TU) + libtensor.
# Build: make cp-ulysses    Run: make run-cp-ulysses NP=2  (and NP=4)
# =============================================================================
ULYSSES_SRC := Tests/cp_ulysses_parity.cpp
ULYSSES_OBJ := $(OBJDIR)/Tests/cp_ulysses_parity.o
ULYSSES_EXE := $(BINDIR)/cp_ulysses_parity_exec
.PHONY: cp-ulysses run-cp-ulysses
cp-ulysses: $(ULYSSES_EXE)
$(ULYSSES_EXE): $(ULYSSES_OBJ) $(OBJECTS_FROM_CPP) $(OBJECTS_FROM_CU) | $(TENSOR_LIB_A) $(PROFILER_LIB)
	@mkdir -p $(BINDIR)
	@echo -e "\n--- Linking Ulysses parity test (sm_$(SM_ARCH)): $@"
	$(NVCC) $(LINKFLAGS) $(LDFLAGS) -o $@ \
	        $(ULYSSES_OBJ) $(OBJECTS_FROM_CPP) $(OBJECTS_FROM_CU) \
	        -Xlinker --start-group $(TENSOR_LIB_A) -Xlinker --end-group \
	        $(LDLIBS) $(LINK_MEMFLAGS)

run-cp-ulysses: $(ULYSSES_EXE)
	@echo "--- Running Ulysses parity on $(NP) rank(s) ---"
	LD_LIBRARY_PATH=$(TENSOR_LIBDIR):$(PROFILER_LIBDIR):$$LD_LIBRARY_PATH \
	    mpirun -np $(NP) ./$(ULYSSES_EXE) $(ARGS)

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
	LD_LIBRARY_PATH=$(TENSOR_LIBDIR):$(PROFILER_LIBDIR):$$LD_LIBRARY_PATH \
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
		LD_LIBRARY_PATH=$(TENSOR_LIBDIR):$(PROFILER_LIBDIR):$$LD_LIBRARY_PATH \
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
