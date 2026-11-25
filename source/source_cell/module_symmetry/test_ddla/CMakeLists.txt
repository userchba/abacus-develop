# Register the symmetry source and its tests with the shared MPI executable.
target_sources(MODULE_IO_ddla_postprocess PRIVATE
  ${CMAKE_CURRENT_LIST_DIR}/test_symm_rotation_k.cpp
  ${ABACUS_SOURCE_DIR}/source_cell/module_symmetry/symm_rotation_k.cpp)
