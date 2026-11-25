# Distributed postprocessing tests share one MPI main and backend grid.
addtest(
  TARGET MODULE_IO_ddla_postprocess
  LIBS parameter base device ${BLACS_LIB}
  SOURCES ${ABACUS_SOURCE_DIR}/source_base/test/mpi_test_main.cpp
          ${ABACUS_SOURCE_DIR}/source_basis/module_ao/parallel_orbitals.cpp)
include(${ABACUS_SOURCE_DIR}/source_cell/module_symmetry/test_ddla/CMakeLists.txt)
include(${ABACUS_SOURCE_DIR}/source_lcao/module_dftu/test_parallel/CMakeLists.txt)
# Isolate these production matrix routines from unused geometry/file paths.
target_compile_options(MODULE_IO_ddla_postprocess PRIVATE -ffunction-sections -fdata-sections)
target_link_options(MODULE_IO_ddla_postprocess PRIVATE -Wl,--gc-sections)
if(ENABLE_DDLA)
  target_link_libraries(MODULE_IO_ddla_postprocess PRIVATE ddla::ddla)
endif()
