cmake_minimum_required(VERSION 2.6.0)

project(FreiburgLcm)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_SOURCE_DIR}")


find_package(PkgConfig REQUIRED)
find_package(IlmBase)
find_package(OpenEXR REQUIRED)
find_package(OpenCV REQUIRED)
find_package(ZLIB REQUIRED)
find_package(JPEG REQUIRED)
find_package(Boost COMPONENTS thread REQUIRED)
find_package(Boost COMPONENTS system REQUIRED)


include_directories(${EIGEN_INCLUDE_DIRS})
include_directories(${OpenEXR_INCLUDE_DIRS})

set(Boost_USE_STATIC_LIBS OFF)
set(Boost_USE_MULTITHREADED ON)
set(Boost_USE_STATIC_RUNTIME OFF)
set(BOOST_ALL_DYN_LINK ON) 


pkg_check_modules(LCM REQUIRED lcm)

file(GLOB lcmtypes lcmtypes/*.lcm)
execute_process(COMMAND lcm-gen -x ${lcmtypes} --cpp-hpath ${CMAKE_CURRENT_SOURCE_DIR}/lcmtypes/)
execute_process(COMMAND lcm-gen -p ${lcmtypes} --ppath ${CMAKE_CURRENT_SOURCE_DIR}/lcmtypes/)

file(GLOB frei_srcs FreiburgLcm.cpp)
file(GLOB klg_srcs KlgToLcm.cpp RawLogReader.cpp)
file(GLOB canyon_srcs CanyonToLcm.cpp)
file(GLOB canyon_exr_srcs CanyonExrToLcm.cpp)

set(CMAKE_CXX_FLAGS ${ADDITIONAL_CMAKE_CXX_FLAGS} "-g -msse2 -msse3 -Wall -std=c++11")

add_executable(FreiburgLcm ${frei_srcs})

target_link_libraries(FreiburgLcm
                      ${ZLIB_LIBRARY}
                      ${LCM_LIBRARIES}
                      ${Boost_LIBRARIES}
                      ${OpenCV_LIBS} 
                      boost_system
                      boost_filesystem
                      boost_thread
                      ${LCM_LIBRARIES}
)

add_executable(KlgToLcm ${klg_srcs})

target_link_libraries(KlgToLcm
                      ${ZLIB_LIBRARY}
                      ${LCM_LIBRARIES}
                      ${Boost_LIBRARIES}
                      ${OpenCV_LIBS} 
                      boost_system
                      boost_filesystem
                      boost_thread
                      ${LCM_LIBRARIES}
                      ${JPEG_LIBRARIES}
)


add_executable(CanyonToLcm ${canyon_srcs})

target_link_libraries(CanyonToLcm
                      ${ZLIB_LIBRARY}
                      ${LCM_LIBRARIES}
                      ${Boost_LIBRARIES}
                      ${OpenCV_LIBS} 
                      boost_system
                      boost_filesystem
                      boost_thread
                      ${LCM_LIBRARIES}
                      ${JPEG_LIBRARIES}
                      ${Eigen_LIBRARIES}
)

add_executable(CanyonExrToLcm ${canyon_exr_srcs})

target_link_libraries(CanyonExrToLcm
                      ${ZLIB_LIBRARY}
                      ${LCM_LIBRARIES}
                      ${Boost_LIBRARIES}
                      ${OpenCV_LIBS} 
                      boost_system
                      boost_filesystem
                      boost_thread
                      ${LCM_LIBRARIES}
                      ${JPEG_LIBRARIES}
                      ${Eigen_LIBRARIES}
                      OpenEXR::IlmImf
)