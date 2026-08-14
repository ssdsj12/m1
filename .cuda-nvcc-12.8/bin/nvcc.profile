
TOP              = $(_HERE_)/../$(_TARGET_DIR_)

CICC_PATH        = $(TOP)/nvvm/bin
NVVMIR_LIBRARY_DIR = $(TOP)/nvvm/libdevice

LD_LIBRARY_PATH += $(TOP)/lib:
PATH            += $(TOP)/bin:$(CICC_PATH):$(TOP)/../../bin:$(TOP)/../../$(_NVVM_BRANCH_)/bin:

INCLUDES        +=  "-I$(TOP)/include" $(_SPACE_)

LIBRARIES        =+ $(_SPACE_) "-L$(TOP)/lib/stubs" "-L$(TOP)/lib"

CUDAFE_FLAGS    +=
PTXAS_FLAGS     +=
