###############################################################################
# Find OpenNI2
#
# This sets the following variables:
# ORBSLAM3_FOUND - True if ORBSLAM3 was found.
# ORBSLAM3_INCLUDE_DIRS - Directories containing the ORBSLAM3 include files.
# ORBSLAM3_LIBRARIES - Libraries needed to use ORBSLAM3.

# find_package(PkgConfig)
# if(${CMAKE_VERSION} VERSION_LESS 2.8.2)
#   pkg_check_modules(PC_OPENNI openni2-dev)
# else()
#   pkg_check_modules(PC_OPENNI QUIET openni2-dev)
# endif()

# set(OPENNI2_DEFINITIONS ${PC_OPENNI_CFLAGS_OTHER})

#add a hint so that it can find it without the pkg-config

find_path(ORBSLAM3_INCLUDE_DIR System.h
          PATHS
            "${CMAKE_SOURCE_DIR}/../libs/orb_slam3/include"
            "${CMAKE_SOURCE_DIR}/../../libs/orb_slam3/include"
            "${CMAKE_SOURCE_DIR}/../../../libs/orb_slam3/include"
            "${CMAKE_SOURCE_DIR}/../../../../libs/orb_slam3/include"
            "${CMAKE_SOURCE_DIR}/../../../libs/orb_slam3/include"
            "${CMAKE_SOURCE_DIR}/../../../../libs/orb_slam3/include"
)

find_path(DBOW2_INCLUDE_DIR FeatureVector.h BowVector.h 
          PATHS
            "${CMAKE_SOURCE_DIR}/../libs/orb_slam3/Thirdparty/DBoW2/DBoW2"
            "${CMAKE_SOURCE_DIR}/../../libs/orb_slam3/Thirdparty/DBoW2/DBoW2"
            "${CMAKE_SOURCE_DIR}/../../../libs/orb_slam3/Thirdparty/DBoW2/DBoW2"
            "${CMAKE_SOURCE_DIR}/../../../../libs/orb_slam3/Thirdparty/DBoW2/DBoW2"
            "${CMAKE_SOURCE_DIR}/../../../libs/orb_slam3/Thirdparty/DBoW2/DBoW2"
            "${CMAKE_SOURCE_DIR}/../../../../libs/orb_slam3/Thirdparty/DBoW2/DBoW2"
)

# if(${CMAKE_CL_64})
#     set(ORBSLAM3_PATH_SUFFIXES lib64)
# else()
#     set(ORBSLAM3_PATH_SUFFIXES lib)
# endif()

#add a hint so that it can find it without the pkg-config
find_library(ORBSLAM3_LIBRARY
             NAMES libORB_SLAM3.so
             PATHS
               "${CMAKE_SOURCE_DIR}/../libs/orb_slam3/lib"
               "${CMAKE_SOURCE_DIR}/../../libs/orb_slam3/lib"
               "${CMAKE_SOURCE_DIR}/../../../libs/orb_slam3/lib"
               "${CMAKE_SOURCE_DIR}/../../../../libs/orb_slam3/lib"
               "${CMAKE_SOURCE_DIR}/../../../libs/orb_slam3/lib"
               "${CMAKE_SOURCE_DIR}/../../../../libs/orb_slam3/lib"
)

set(ORBSLAM3_INCLUDE_DIRS ${ORBSLAM3_INCLUDE_DIR} ${DBOW2_INCLUDE_DIR} "${CMAKE_SOURCE_DIR}/../../../../libs/orb_slam3/" "${CMAKE_SOURCE_DIR}/../../../../libs/orb_slam3/include/CameraModels/")
set(ORBSLAM3_LIBRARIES ${ORBSLAM3_LIBRARY})

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(ORB_SLAM3 DEFAULT_MSG
    ORBSLAM3_LIBRARY ORBSLAM3_INCLUDE_DIR)

mark_as_advanced(ORBSLAM3_LIBRARY ORBSLAM3_INCLUDE_DIR)


IF (ORBSLAM3_INCLUDE_DIRS AND ORBSLAM3_LIBRARIES)
   SET(ORBSLAM3_FOUND TRUE)
   MESSAGE(STATUS "Found ORB_SLAM3")
ENDIF (ORBSLAM3_INCLUDE_DIRS AND ORBSLAM3_LIBRARIES)