cmake_minimum_required(VERSION 2.6.0)

project(ElasticFusion)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_SOURCE_DIR}")

if(WIN32)
  macro(CANONIFY_BOOL var)
    if(${var})
      set(${var} TRUE)
    else()
      set(${var} FALSE)
    endif()
  endmacro()
endif()

if(WIN32)
  find_package(JPEG REQUIRED)
endif()

set(Boost_USE_STATIC_LIBS OFF) 
set(Boost_USE_MULTITHREADED ON)  
set(Boost_USE_STATIC_RUNTIME OFF) 
set(BOOST_ALL_DYN_LINK ON)

find_package(OpenCV 3)
if(NOT OpenCV_FOUND)
   find_package(OpenCV 2.4.3 QUIET)
   if(NOT OpenCV_FOUND)
      message(FATAL_ERROR "OpenCV > 2.4.3 not found.")
   endif()
endif()

MESSAGE("OPENCV VERSION:")
MESSAGE(${OpenCV_VERSION})

find_package(Boost COMPONENTS thread system REQUIRED) 
find_package(LAPACK REQUIRED)
find_package(BLAS REQUIRED)
find_package(ZLIB REQUIRED)
find_package(Pangolin 0.1 REQUIRED)
find_package(CUDA REQUIRED)
find_package(OpenNI2 REQUIRED)
find_package(efusion REQUIRED)
find_package(SuiteSparse REQUIRED)
find_package(PkgConfig REQUIRED)
find_package(realsense2 REQUIRED)
find_package(PCL 1.8.1 EXACT REQUIRED COMPONENTS common features keypoints gpu_features search)
find_package(Boost COMPONENTS program_options REQUIRED)
find_package(ORBSLAM3 REQUIRED)
find_package(onnxruntime REQUIRED)
find_package(Eigen3 3.1.0 REQUIRED)

MESSAGE("Eigen Version found: ${EIGEN3_VERSION_STRING}")


pkg_check_modules(LCM REQUIRED lcm)

file(GLOB lcmtypes Tools/lcmtypes/*.lcm)
execute_process(COMMAND lcm-gen -x ${lcmtypes} --cpp-hpath ${CMAKE_CURRENT_SOURCE_DIR}/Tools/lcmtypes/)

if(WIN32)
  find_package(RealSense QUIET)
  CANONIFY_BOOL(REALSENSE_FOUND)
  message(STATUS "librealsense found: ${REALSENSE_FOUND}")
  option(WITH_REALSENSE "Build with Intel RealSense support?" ${REALSENSE_FOUND})
endif()

if(WIN32)
  include_directories(${JPEG_INCLUDE_DIR})
endif()

include_directories(${ZLIB_INCLUDE_DIR})
#include_directories(${EIGEN_INCLUDE_DIRS})
include_directories(${EIGEN3_INCLUDE_DIR})
include_directories(${Pangolin_INCLUDE_DIRS})
include_directories(${CUDA_INCLUDE_DIRS})
include_directories(${EFUSION_INCLUDE_DIR})
include_directories(${OPENNI2_INCLUDE_DIR})
include_directories(${LCM_INCLUDE_DIRS})
include_directories(${Boost_INCLUDE_DIR})
include_directories(${PCL_INCLUDE_DIRS})
include_directories(${ORBSLAM3_INCLUDE_DIRS})
include_directories(${onnxruntime_INCLUDE_DIRS})
#link_directories(${PCL_LIBRARY_DIRS})
#message("PCL lib dirs: ${PCL_LIBRARY_DIRS}")
#add_definitions(${PCL_DEFINITIONS})

if(WITH_REALSENSE)
  include_directories(${REALSENSE_INCLUDE_DIR})
  add_definitions(-DWITH_REALSENSE)
  set(EXTRA_LIBS ${EXTRA_LIBS} ${REALSENSE_LIBRARY})
endif()

file(GLOB srcs *.cpp)
file(GLOB tools_srcs Tools/*.cpp Tools/networking/*.cpp)

if(WIN32)
  file(GLOB hdrs *.h)
  file(GLOB tools_hdrs Tools/*.h Tools/networking/*.h)
endif()

if(WIN32)
  add_definitions(-Dlinux=0)
  add_definitions(-DWIN32_LEAN_AND_MEAN)
  add_definitions(-DNOMINMAX)
  set(ADDITIONAL_CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS})
  set(EXTRA_WINDOWS_LIBS ${EXTRA_WINDOWS_LIBS} ws2_32 ${JPEG_LIBRARY})
else()
  add_definitions(-Dlinux=1)
endif()

set(CMAKE_CXX_FLAGS ${ADDITIONAL_CMAKE_CXX_FLAGS} "-msse2 -msse3 -O3 -mfpmath=sse -Wall -std=c++11")#-march=native -mavx 
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -Wall -std=c++11")

add_executable(ElasticFusion 
               ${srcs}
               ${tools_srcs}
               ${hdrs}
               ${tools_hdrs}
)

target_link_libraries(ElasticFusion
                      ${ORBSLAM3_LIBRARIES}
                      ${OpenCV_LIBS}
					            ${EXTRA_WINDOWS_LIBS}
                      ${ZLIB_LIBRARY}
                      ${Pangolin_LIBRARIES}
                      ${CUDA_LIBRARIES}
                      ${EXTRA_LIBS}
                      ${EFUSION_LIBRARY}
                      ${OPENNI2_LIBRARY}
                      ${SUITESPARSE_LIBRARIES}
                      ${BLAS_LIBRARIES}
                      ${LAPACK_LIBRARIES}
                      ${LCM_LIBRARIES}                      
                      ${realsense2_LIBRARY}
                      ${Boost_LIBRARIES}
                      boost_system
                      boost_thread
                      ${onnxruntime_LIBRARIES}
                      ${EIGEN3_LIBS}
)


INSTALL(TARGETS ElasticFusion
        RUNTIME DESTINATION bin
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib
)
