# Init CMake (require at least version 3.12.0)
CMAKE_MINIMUM_REQUIRED(VERSION 3.12.0)

# General build settings
IF (NOT CMAKE_BUILD_TYPE)
	SET(CMAKE_BUILD_TYPE "Release" CACHE STRING "Possible build types: None Debug Release RelWithDebInfo MinSizeRel")
ENDIF()

IF (NOT CMAKE_INSTALL_PREFIX)
	IF (UNIX AND NOT APPLE)
		SET(CMAKE_INSTALL_PREFIX "/usr" CACHE STRING "CMake default install prefix, set to /usr on Unix/Linux")
	ELSE()
		MESSAGE(FATAL_ERROR "CMAKE_INSTALL_PREFIX is not set")
	ENDIF()
ENDIF()

# Project name and version
PROJECT(feast-dv-module
	VERSION 1.0.0
	LANGUAGES C CXX)

# Define installation paths
INCLUDE(GNUInstallDirs)

# Custom CMake base module
SET(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH}
	${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_DATAROOTDIR}/dv /usr/${CMAKE_INSTALL_DATAROOTDIR}/dv
	${CMAKE_INSTALL_PREFIX}/share/dv /usr/share/dv
	/usr/local/${CMAKE_INSTALL_DATAROOTDIR}/dv /usr/local/share/dv)

# Basic setup
INCLUDE(dv-modules)
DV_MODULE_SETUP()

# Set full RPATH, modules are libraries for DV
SET(CMAKE_INSTALL_RPATH ${DV_MODULES_DIR})

find_package(Eigen3 REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIR})
#find_package( OpenCV REQUIRED )
#include_directories( ${OpenCV_INCLUDE_DIRS} )

# Compile this module (standard C++ file, no additional dependencies)
ADD_LIBRARY(feast_dv_module SHARED FeastDv.cpp)

#SET_TARGET_PROPERTIES(mantis_shrimp_dv
#	PROPERTIES
#	PREFIX "user_"
#)
TARGET_LINK_LIBRARIES(feast_dv_module PRIVATE ${DV_LIBRARIES})


#TARGET_LINK_LIBRARIES(feast_dv_module PRIVATE ${DV_LIBRARIES} ${OpenCV_LIBS})

INSTALL(TARGETS feast_dv_module DESTINATION ${DV_MODULES_DIR})
