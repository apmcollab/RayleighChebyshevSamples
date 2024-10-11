# 
#  CMakeLists.txt
#
#  Generator      : CMakeCreator 
#  XML data file  : CMakeLaplaceOp2dEigTest_SparseMatrix.xml
#  Date           : 2024-10-11 
#

cmake_minimum_required (VERSION 3.15)
set(CMAKE_CXX_STANDARD 17)
project (LaplaceOp2dEigTest_SparseMatrix)

message(STATUS "System Name ${CMAKE_SYSTEM_NAME}")

find_package (Git)
if (GIT_FOUND)
    #execute_process(
    #COMMAND ${GIT_EXECUTABLE} rev-list --all --count 
    #RESULT_VARIABLE RESULT
    #OUTPUT_VARIABLE GIT_BUILD_VERSION
    #OUTPUT_STRIP_TRAILING_WHITESPACE)
    #message(STATUS "Build Version ${GIT_BUILD_VERSION}")
    #add_compile_definitions(GIT_REV_COUNT_=${GIT_BUILD_VERSION})
    execute_process(
    COMMAND ${GIT_EXECUTABLE} rev-list --max-count=1 HEAD 
    RESULT_VARIABLE RESULT
    OUTPUT_VARIABLE GIT_HASH
    OUTPUT_STRIP_TRAILING_WHITESPACE)
    message(STATUS "Build hash ${GIT_HASH}")
    add_compile_definitions(GIT_HASH_="${GIT_HASH}")  
    execute_process(
    COMMAND ${GIT_EXECUTABLE} log -1 --format='%ci'
    RESULT_VARIABLE RESULT
    OUTPUT_VARIABLE GIT_DATE
    OUTPUT_STRIP_TRAILING_WHITESPACE)
    message(STATUS "Build date ${GIT_DATE}")
    add_compile_definitions(GIT_DATE_="${GIT_DATE}")  
endif (GIT_FOUND)

# Loads helper to print messages
 
include(CMakePrintHelpers)

# Suppresses the creation of the install script 

set(CMAKE_SKIP_INSTALL_RULES True) 

# Setup for debug and release targets 

ADD_CUSTOM_TARGET(debug
  COMMAND ${CMAKE_COMMAND} -DCMAKE_BUILD_TYPE=Debug ${CMAKE_SOURCE_DIR}
  COMMAND ${CMAKE_COMMAND} --build ${CMAKE_BINARY_DIR} --target all
  COMMENT "Creating the executable in the debug mode.")

ADD_CUSTOM_TARGET(release
  COMMAND ${CMAKE_COMMAND} -DCMAKE_BUILD_TYPE=Release ${CMAKE_SOURCE_DIR}
  COMMAND ${CMAKE_COMMAND} --build ${CMAKE_BINARY_DIR} --target all
  COMMENT "Creating the executable in the release mode.") 
  
#unset all options that may be set 
UNSET(USE_CPP11        CACHE)
UNSET(USE_OPENMP       CACHE)
UNSET(USE_LAPACK       CACHE)
UNSET(USE_OPENBLAS     CACHE)
UNSET(USE_OPENBLAS_THREADED   CACHE)
UNSET(USE_FFTW         CACHE)
UNSET(USE_OPENMP_FFTW  CACHE)
UNSET(USE_FLAME        CACHE)
UNSET(USE_MKL          CACHE)
UNSET(USE_MKL_FFTW     CACHE)
UNSET(USE_SQLITE3      CACHE)
UNSET(USE_MEMCHECK     CACHE)


#
# Global include directories 
#

list(APPEND IncludeDirs "${CMAKE_SOURCE_DIR}")
list(APPEND IncludeDirs "${CMAKE_SOURCE_DIR}/../Components")

#####################################################
# Collecting data for components specified by options 
#####################################################

#
# Override specified option values by appending to    
# the cmake build command 
#
# -DOption_Name=ON (or = OFF) 
#
# Multiple option specifications are allowed. 
#

OPTION(USE_OPENMP  "Option USE_OPENMP"  OFF)
OPTION(USE_MEMCHECK  "Option USE_MEMCHECK"  ON)

# OpenMP INCLUSION
 
if(USE_OPENMP MATCHES ON)
#
# Local variables set so OpenMP can be found on MacOS >= Sonoma (Darwin 23)
#
if (CMAKE_HOST_SYSTEM_NAME STREQUAL "Darwin" AND CMAKE_HOST_SYSTEM_VERSION VERSION_GREATER_EQUAL 23)
set(OpenMP_C_FLAGS "-fopenmp=lomp")
set(OpenMP_CXX_FLAGS "-fopenmp=lomp") 
set(OpenMP_C_LIB_NAMES "libomp")
set(OpenMP_CXX_LIB_NAMES "libomp")
set(OpenMP_libomp_LIBRARY "/usr/local/opt/libomp/lib/libomp.dylib")
set(OpenMP_CXX_FLAGS "-Xpreprocessor -fopenmp -I/usr/local/opt/libomp/include")
set(OpenMP_CXX_LIB_NAMES "libomp")
set(OpenMP_C_FLAGS "-Xpreprocessor -fopenmp  -I/usr/local/opt/libomp/include")
endif ()
    find_package(OpenMP)
endif()


# Enable ctest to run memory check (use ctest -T memcheck) 

if(USE_MEMCHECK)
include (CTest)
endif() 

#
# Enable testing and create directories for testing 
#

enable_testing()

file(MAKE_DIRECTORY "${CMAKE_SOURCE_DIR}/Testing/LaplaceOp2dEigTest_SparseMatrix")

#####################################################
# Specify list of target sources containing "main"  
#####################################################

set(Main_Sources "LaplaceOp2dEigTest_SparseMatrix.cpp" )

#
#   Loop over build commands for each target  
#

foreach(mainFile ${Main_Sources} )
string( REPLACE ".cpp" "" mainExecName ${mainFile} )

##################################################################
#   Target :  LaplaceOp2dEigTest_SparseMatrix
##################################################################

    if("${mainExecName}.cpp" STREQUAL "LaplaceOp2dEigTest_SparseMatrix.cpp")
      add_executable( ${mainExecName} "LaplaceOp2dEigTest_SparseMatrix.cpp" )
         
      set_property(TARGET ${mainExecName} PROPERTY CXX_STANDARD 17)
         
#
#     Supporting libraries and include directories 
#
      if(OpenMP_CXX_FOUND) 
         target_link_libraries(${mainExecName} PUBLIC OpenMP::OpenMP_CXX)
      endif()


      target_include_directories(${mainExecName} PUBLIC ${IncludeDirs} ) 

#      --- Commands for ctest setup ---  

#     Add target to test set 

      add_test(NAME  ${mainExecName} WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}/Testing/${mainExecName}"
      COMMAND "${CMAKE_SOURCE_DIR}/${CMAKE_BUILD_TYPE}/${mainExecName}")

    endif()

#----------------------------------------------------------------#

#########################################################################
#   Executable compile features applied to every target
##########################################################################

#   target_compile_features(${mainExecName} PUBLIC cxx_std_11)
    set_target_properties  (${mainExecName} PROPERTIES CXX_EXTENSIONS OFF)
#
#   Executable output properties applied to every target
#
    set_target_properties(${mainExecName}  PROPERTIES 
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/${CMAKE_BUILD_TYPE}" 
    OUTPUT_NAME "${mainExecName}")
#
#  Additional compiler options (operating system dependent)  
#  Visual studio options if "Windows", Mac options if "Darwin"  
#
    set(ADDITIONAL_DEBUG_OPTIONS "")
    if("${CMAKE_SYSTEM_NAME}" STREQUAL "Linux")
      set(ADDITIONAL_DEBUG_OPTIONS "-Wall" "-Wno-pessimizing-move" "-D _DEBUG" "-D RC_WITHOUT_LAPACK_" )
    elseif("${CMAKE_SYSTEM_NAME}" STREQUAL "Windows")
      set(ADDITIONAL_DEBUG_OPTIONS "/W3" "/wd\"4458" "-D _DEBUG" "-D RC_WITHOUT_LAPACK_" )
    elseif("${CMAKE_SYSTEM_NAME}" STREQUAL "Darwin")
      set(ADDITIONAL_DEBUG_OPTIONS "-Wall" "-Wno-pessimizing-move" "-D _DEBUG" "-D RC_WITHOUT_LAPACK_" )
    endif()

    set(ADDITIONAL_RELEASE_OPTIONS "")
    if("${CMAKE_SYSTEM_NAME}" STREQUAL "Linux")
      set(ADDITIONAL_RELEASE_OPTIONS "-D RC_WITHOUT_LAPACK_" )
    elseif("${CMAKE_SYSTEM_NAME}" STREQUAL "Windows")
      set(ADDITIONAL_RELEASE_OPTIONS "-D RC_WITHOUT_LAPACK_" )
    elseif("${CMAKE_SYSTEM_NAME}" STREQUAL "Darwin")
      set(ADDITIONAL_RELEASE_OPTIONS "-D RC_WITHOUT_LAPACK_" )
    endif()

    if("${CMAKE_SYSTEM_NAME}" STREQUAL "Linux")
        if((USE_OPENBLAS) AND (NOT USE_MKL) AND (NOT USE_FLAME) AND (DEFINED ENV{OPENBLAS_THREADED}))
            set(ADDITIONAL_RELEASE_OPTIONS ${ADDITIONAL_RELEASE_OPTIONS} "-D OPENBLAS_THREADED" )
            set(ADDITIONAL_DEBUG_OPTIONS   ${ADDITIONAL_DEBUG_OPTIONS} "-D OPENBLAS_THREADED" )
        endif()
    endif()

    if("${CMAKE_SYSTEM_NAME}" STREQUAL "Linux")
        if((USE_OPENBLAS_THREADED) AND (NOT USE_MKL))
            set(ADDITIONAL_RELEASE_OPTIONS ${ADDITIONAL_RELEASE_OPTIONS} "-D OPENBLAS_THREADED" )
            set(ADDITIONAL_DEBUG_OPTIONS   ${ADDITIONAL_DEBUG_OPTIONS} "-D OPENBLAS_THREADED" )
        endif()
    endif()

    if(("${CMAKE_SYSTEM_NAME}" STREQUAL "Linux") OR ("${CMAKE_SYSTEM_NAME}" STREQUAL "Darwin")) 
        if(USE_OPENMP_FFTW)
            set(ADDITIONAL_RELEASE_OPTIONS ${ADDITIONAL_RELEASE_OPTIONS} "-D OPENMP_FFTW" )
            set(ADDITIONAL_DEBUG_OPTIONS   ${ADDITIONAL_DEBUG_OPTIONS} "-D OPENMP_FFTW" )
        endif()
    endif()

    target_compile_options(${mainExecName} PUBLIC "$<$<CONFIG:DEBUG>:${ADDITIONAL_DEBUG_OPTIONS}>")
    target_compile_options(${mainExecName} PUBLIC "$<$<CONFIG:RELEASE>:${ADDITIONAL_RELEASE_OPTIONS}>")

endforeach()   # Loop over targets 
