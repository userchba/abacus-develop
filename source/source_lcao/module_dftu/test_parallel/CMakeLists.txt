# Register the DFT+U source and its tests with the shared MPI executable.
target_sources(MODULE_IO_ddla_postprocess PRIVATE
  ${CMAKE_CURRENT_LIST_DIR}/test_dftu_nao_pots.cpp
  ${ABACUS_SOURCE_DIR}/source_lcao/module_dftu/dftu_nao_pots.cpp)
