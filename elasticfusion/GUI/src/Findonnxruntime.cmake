# This will define the following variables:
#   onnxruntime_FOUND        -- True if the system has the onnxruntime library
#   onnxruntime_INCLUDE_DIRS -- The include directories for onnxruntime
#   onnxruntime_LIBRARIES    -- Libraries to link against
#   onnxruntime_CXX_FLAGS    -- Additional (required) compiler flags

include(FindPackageHandleStandardArgs)

# Assume we are in <install-prefix>/share/cmake/onnxruntime/onnxruntimeConfig.cmake
get_filename_component(CMAKE_CURRENT_LIST_DIR "${CMAKE_CURRENT_LIST_FILE}" PATH)
get_filename_component(onnxruntime_INSTALL_PREFIX "${CMAKE_CURRENT_LIST_DIR}/../../../../libs/" ABSOLUTE)

set(onnxruntime_INCLUDE_DIRS ${onnxruntime_INSTALL_PREFIX}/onnxruntime/include)

find_library(onnxruntime_LIBRARY
             NAMES libonnxruntime.so
             PATHS
               "${onnxruntime_INSTALL_PREFIX}/onnxruntime/build/Linux/Release"
            #    "${CMAKE_SOURCE_DIR}/../../onnxruntime/build/Linux/Release"
            #    "${CMAKE_SOURCE_DIR}/../../../onnxruntime/build/Linux/Release"
            #    "${CMAKE_SOURCE_DIR}/../../../../onnxruntime/build/Linux/Release"
            #    "${CMAKE_SOURCE_DIR}/../../../onnxruntime/build/Linux/Release"
            #    "${CMAKE_SOURCE_DIR}/../../../../onnxruntime/build/Linux/Release"
)
set(onnxruntime_LIBRARIES ${onnxruntime_LIBRARY})
#set(onnxruntime_LIBRARIES onnxruntime)
set(onnxruntime_CXX_FLAGS "") # no flags needed


# find_library(onnxruntime_LIBRARY onnxruntime
#     PATHS "${onnxruntime_INSTALL_PREFIX}/onnxruntime/build/Linux/Release/"
# )

# add_library(onnxruntime SHARED IMPORTED)
# set_property(TARGET onnxruntime PROPERTY IMPORTED_LOCATION "${onnxruntime_LIBRARY}")
# set_property(TARGET onnxruntime PROPERTY INTERFACE_INCLUDE_DIRECTORIES "${onnxruntime_INCLUDE_DIRS}")
# set_property(TARGET onnxruntime PROPERTY INTERFACE_COMPILE_OPTIONS "${onnxruntime_CXX_FLAGS}")

find_package_handle_standard_args(onnxruntime DEFAULT_MSG onnxruntime_LIBRARY onnxruntime_INCLUDE_DIRS)

IF (onnxruntime_INCLUDE_DIRS AND onnxruntime_LIBRARIES)
   SET(onnxruntime_FOUND TRUE)
   MESSAGE(STATUS "Found onnxruntime")
ENDIF (onnxruntime_INCLUDE_DIRS AND onnxruntime_LIBRARIES)