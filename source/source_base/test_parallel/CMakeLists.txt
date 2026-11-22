addtest(
  TARGET MODULE_BASE_ParaCommon
  LIBS parameter MPI::MPI_CXX
  SOURCES parallel_common_test.cpp
          ../global_variable.cpp
          ../parallel_common.cpp
          ../parallel_reduce.cpp
          ../parallel_comm.cpp
          ../parallel_global.cpp
          ../tool_quit.cpp
          ../global_file.cpp
          ../global_function.cpp
          ../memory_recorder.cpp
          ../timer.cpp)

addtest(
  TARGET MODULE_BASE_ParaGlobal
  LIBS parameter MPI::MPI_CXX
  SOURCES parallel_global_test.cpp
          ../global_variable.cpp
          ../parallel_global.cpp
          ../parallel_comm.cpp
          ../tool_quit.cpp
          ../global_file.cpp
          ../global_function.cpp
          ../memory_recorder.cpp
          ../timer.cpp
          ../parallel_reduce.cpp)

addtest(
  TARGET MODULE_BASE_ParaReduce
  LIBS parameter MPI::MPI_CXX
  SOURCES parallel_reduce_test.cpp
          ../global_variable.cpp
          ../parallel_global.cpp
          ../parallel_comm.cpp
          ../parallel_common.cpp
          ../parallel_reduce.cpp
          ../tool_quit.cpp
          ../global_file.cpp
          ../global_function.cpp
          ../memory_recorder.cpp
          ../timer.cpp)

abacus_copy_test_files(parallel_common_test.sh parallel_global_test.sh
                       parallel_reduce_test.sh)
find_program(BASH bash)
add_test(
  NAME MODULE_BASE_parallel_common_test
  COMMAND ${BASH} parallel_common_test.sh
          "$<TARGET_FILE:MODULE_BASE_ParaCommon>"
  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
add_test(
  NAME MODULE_BASE_parallel_global_test
  COMMAND ${BASH} parallel_global_test.sh
          "$<TARGET_FILE:MODULE_BASE_ParaGlobal>"
  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
add_test(
  NAME MODULE_BASE_parallel_reduce_test
  COMMAND ${BASH} parallel_reduce_test.sh
          "$<TARGET_FILE:MODULE_BASE_ParaReduce>"
  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})

addtest(
  TARGET MODULE_BASE_para_gemm
  LIBS MPI::MPI_CXX base device parameter
  SOURCES test_para_gemm.cpp)

addtest(
  TARGET MODULE_BASE_parallel_device
  LIBS MPI::MPI_CXX base device parameter
  SOURCES parallel_device_test.cpp ../test/mpi_test_main.cpp)

add_test(
  NAME MODULE_BASE_parallel_device_parallel
  COMMAND mpirun -np 4 "$<TARGET_FILE:MODULE_BASE_parallel_device>"
  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})

addtest(
  TARGET MODULE_BASE_math_chebyshev_mpi
  LIBS MPI::MPI_CXX parameter base device container
  SOURCES math_cheby_mpi_test.cpp)

add_test(
  NAME MODULE_BASE_para_gemm_parallel
  COMMAND mpirun -np 4 "$<TARGET_FILE:MODULE_BASE_para_gemm>"
  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})

AddTest(
  TARGET MODULE_BASE_parallel_domain_grid
  SOURCES parallel_domain_grid_test.cpp ../test/mpi_test_main.cpp
  LIBS parameter MPI::MPI_CXX base device)

add_test(
  NAME MODULE_BASE_parallel_domain_grid_parallel
  COMMAND mpirun -np 4 "$<TARGET_FILE:MODULE_BASE_parallel_domain_grid>"
  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})

find_program(BASH bash)

# Select the same ScaLAPACK/BLACS provider as the main build.
if(MKL_FOUND)
  set(BLACS_LIB ${MKL_LIBRARIES})
else()
  set(BLACS_LIB ScaLAPACK::ScaLAPACK)
endif()

addtest(
  TARGET MODULE_BASE_blacs_connector
  LIBS parameter MPI::MPI_CXX ${BLACS_LIB}
  SOURCES blacs_connector_test.cpp)
abacus_copy_test_files(blacs_connector_test.sh)
add_test(
  NAME MODULE_BASE_blacs_connector_test
  COMMAND ${BASH} blacs_connector_test.sh
          "$<TARGET_FILE:MODULE_BASE_blacs_connector>"
  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})

# This target also verifies the native path with DDLA entirely unavailable.
addtest(
  TARGET MODULE_BASE_ddla_resident
  LIBS parameter base device ${BLACS_LIB}
  SOURCES test_ddla_linalg_resident.cpp ../test/mpi_test_main.cpp
          ${ABACUS_SOURCE_DIR}/source_basis/module_ao/parallel_orbitals.cpp)
if(ENABLE_DDLA)
  target_link_libraries(MODULE_BASE_ddla_resident PRIVATE ddla::ddla)
endif()
foreach(rank_count IN ITEMS 2 4 6)
  add_test(
    NAME MODULE_BASE_ddla_resident_np${rank_count}
    COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${rank_count}
            ${MPIEXEC_PREFLAGS} $<TARGET_FILE:MODULE_BASE_ddla_resident> ${MPIEXEC_POSTFLAGS})
  set_tests_properties(MODULE_BASE_ddla_resident_np${rank_count}
    PROPERTIES ENVIRONMENT "OMP_NUM_THREADS=1;OPENBLAS_NUM_THREADS=1"
               PROCESSORS "${rank_count}" TIMEOUT 120)
endforeach()

if(ENABLE_DDLA)
  addtest(
    TARGET MODULE_BASE_ddla_context
    LIBS parameter base device ddla::ddla
    SOURCES test_ddla_context.cpp)
  addtest(
    TARGET MODULE_BASE_ddla_linalg
    LIBS parameter base device ddla::ddla
    SOURCES test_ddla_linalg.cpp
            ${ABACUS_SOURCE_DIR}/source_basis/module_ao/parallel_orbitals.cpp)
  foreach(ddla_test IN ITEMS MODULE_BASE_ddla_context MODULE_BASE_ddla_linalg)
    foreach(rank_count IN ITEMS 2 4 6)
      add_test(
        NAME ${ddla_test}_np${rank_count}
        COMMAND
          ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${rank_count}
          ${MPIEXEC_PREFLAGS} $<TARGET_FILE:${ddla_test}> ${MPIEXEC_POSTFLAGS}
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
      set_tests_properties(
        ${ddla_test}_np${rank_count}
        PROPERTIES ENVIRONMENT "OMP_NUM_THREADS=1;OPENBLAS_NUM_THREADS=1"
                   PROCESSORS "${rank_count}" TIMEOUT 120)
    endforeach()
  endforeach()
endif()
