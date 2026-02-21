#!/bin/bash
ncu \
  --metrics l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum,\
l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum,\
l1tex__data_bank_conflicts_pipe_tex_mem_shared_op_st.sum \
  --kernel-name "flash_fwd_kernel" \
  python tests/profile_bank_conflict.py
