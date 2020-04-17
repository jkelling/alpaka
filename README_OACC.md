# Building configuring for the OpenACC backend

```bash
cmake -DALPAKA_ACC_ANY_BT_OACC_ENABLE=on \
	-DALPAKA_ACC_CPU_BT_OMP4_ENABLE=off \
	-DALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLE=on \
	-DALPAKA_ACC_CPU_B_OMP2_T_SEQ_ENABLE=off \
	-DALPAKA_ACC_CPU_B_SEQ_T_OMP2_ENABLE=off \
	-DALPAKA_ACC_CPU_B_SEQ_T_FIBERS_ENABLE=off \
	-DALPAKA_ACC_CPU_B_SEQ_T_THREADS_ENABLE=off \
	-DALPAKA_ACC_CPU_B_TBB_T_SEQ_ENABLE=off \
	-DALPAKA_ACC_GPU_CUDA_ENABLE=off \
	-DALPAKA_ACC_GPU_HIP_ENABLE=off \
```
All other backends are disabled for faster compilation/testing and reduced
environment requirements. The cmake package OpenACC is used to detect the
required OpenACC flags for the compiler. Additional flags can be added, e.g:
- gcc, target x86:
  ```bash
    -DCMAKE_CXX_FLAGS="-foffload=disable -fopenacc"
  ```
  As of gcc 9.2 no test will compile if the nvptx backend is enabled. If cmake
  fials to set the `-fopenacc` flag, it can be set manually.
