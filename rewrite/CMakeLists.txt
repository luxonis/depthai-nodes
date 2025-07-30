cmake_minimum_required(VERSION 3.31)

set(CMAKE_CXX_STANDARD 20)

set(PROJECT_NAME depthai_nodes)
include(cmake/${PROJECT_NAME}Options.cmake)

if(DEPTHAI_BOOTSTRAP_VCPKG)
    message(STATUS "Including vcpkg.cmake")
    include(cmake/vcpkg.cmake)
#    include(cmake/depthaiVcpkgFeatures.cmake)
else()
    message(STATUS "DEPTHAI_BOOTSTRAP_VCPKG is OFF")
endif()



### Constants
set(TARGET_ALIAS nodes::host_extensions)
set(TARGET_NAME depthai-nodes-host_extensions)
set(PROJECT_EXPORT_GROUP "${PROJECT_NAME}Targets")

set(DEPTHAI_XTENSOR_SUPPORT ON)

project(${PROJECT_NAME})

### Deps
find_package(depthai "3" CONFIG REQUIRED)
find_package(pybind11 CONFIG)
# spdlog for library and device logging
find_package(spdlog CONFIG REQUIRED)
find_package(fmt REQUIRED)

set(SOURCES_FILENAME_LIST
        host_extensions/messages/Keypoints
        host_extensions/parsers/BaseParser
        host_extensions/parsers/KeypointParser
        host_extensions/parsers/SimCCKeypointParser
        host_extensions/ParsingNeuralNetwork
        ParserGenerator
)

set(SRC_DIR "src/")
set(INC_DIR "include/")

list(TRANSFORM SOURCES_FILENAME_LIST PREPEND ${SRC_DIR} OUTPUT_VARIABLE SRC_FILES)
list(TRANSFORM SOURCES_FILENAME_LIST PREPEND ${INC_DIR} OUTPUT_VARIABLE HDR_FILES)
list(TRANSFORM HDR_FILES APPEND ".hpp")
list(TRANSFORM SRC_FILES APPEND ".cpp")

add_library(${TARGET_NAME}
        ${HDR_FILES}
        ${SRC_FILES}
        src/utility/ErrorMacros.hpp
        src/utility/spdlog-fmt.hpp
)
set_target_properties(${TARGET_NAME} PROPERTIES EXPORT_NAME ${TARGET_ALIAS})

target_include_directories(${TARGET_NAME}
        PUBLIC
        "$<INSTALL_INTERFACE:${INC_DIR}>"
        "$<INSTALL_INTERFACE:${INC_DIR}host_extensions/>"
        "$<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/${INC_DIR}>"
        "$<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/${INC_DIR}host_extensions/>"
        PRIVATE
        "$<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/${SRC_DIR}utility/>"
)

target_link_libraries(${TARGET_NAME}
        PRIVATE
        spdlog::spdlog
        xtensor
        PUBLIC
        depthai::core
)

if (DEPTHAI_NODES_INSTALL)
    # Install targets
    install(
            TARGETS ${TARGET_NAME}
            EXPORT ${PROJECT_EXPORT_GROUP}
            RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}"
            LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}"
            ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}"
    )
    # Install public headers
    install(DIRECTORY include/depthai_nodes DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}")

    install(EXPORT ${PROJECT_EXPORT_GROUP}
            NAMESPACE ${PROJECT_NAME}::
            DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/${PROJECT_NAME}"
    )

    # Install CMake specific files
    install(FILES
            "${CMAKE_CURRENT_BINARY_DIR}/_install/${PROJECT_NAME}Config.cmake"
            "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}ConfigVersion.cmake"
            DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/${PROJECT_NAME}"
    )
endif ()