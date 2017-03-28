if (FFTW3_FIND_COMPONENTS MATCHES "^$")
  set (_components double)
else ()
  set (_components ${FFTW3_FIND_COMPONENTS})
endif ()

set (_libraries)
foreach (_comp ${_components})
  if (_comp STREQUAL "double")
    list (APPEND _libraries fftw3)
  elseif (_comp STREQUAL "openmp")
   list (APPEND _libraries fftw3_omp)
  elseif (_comp STREQUAL "mpi")
   set (_use_mpi ON)
   list (APPEND _libraries fftw3_mpi)
  else (_comp STREQUAL "double")
    message (FATAL_ERROR "FindFFTW3: unknown component `${_comp}' specified. "
      "Valid components are `double', 'mpi' and `openmp'.")
  endif (_comp STREQUAL "double")
endforeach (_comp ${_components})

# Keep a list of variable names that we need to pass on to
# find_package_handle_standard_args().
set (_check_list)

# Search for all requested libraries.
foreach (_lib ${_libraries})
  string (TOUPPER ${_lib} _LIB)
  find_library (${_LIB}_LIBRARY NAMES ${_lib} ${_lib}-3
    HINTS ${FFTW3_ROOT_DIR} PATH_SUFFIXES lib)
  mark_as_advanced (${_LIB}_LIBRARY)
  list (APPEND FFTW3_LIBRARIES ${${_LIB}_LIBRARY})
  list (APPEND _check_list ${_LIB}_LIBRARY)
endforeach (_lib ${_libraries})

# Search for the header file (fftw3.h normally, fftw3-mpi.h for MPI should also be there)
find_path (FFTW3_INCLUDE_DIR fftw3.h 
  HINTS ${FFTW3_ROOT_DIR} PATH_SUFFIXES include)
mark_as_advanced (FFTW3_INCLUDE_DIR)
list(APPEND _check_list FFTW3_INCLUDE_DIR)

# Handle the QUIETLY and REQUIRED arguments and set FFTW_FOUND to TRUE if
# all listed variables are TRUE
include (FindPackageHandleStandardArgs)
find_package_handle_standard_args (FFTW3 DEFAULT_MSG ${_check_list})