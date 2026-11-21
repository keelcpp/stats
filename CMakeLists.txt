cmake_minimum_required(VERSION 3.20)
project(keel_stats VERSION 0.1.0 LANGUAGES CXX)

option(KEEL_STATS_NO_EXCEPTIONS_RTTI "Compile keel_stats without C++ exceptions and RTTI" OFF)
option(KEEL_STATS_BUILD_TESTS "Build keel_stats unit tests and header purity checks" ${PROJECT_IS_TOP_LEVEL})

if(NOT TARGET keel::keel)
    find_package(keel QUIET CONFIG)
endif()
if(NOT TARGET keel::keel)
    include(FetchContent)
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/../keel/CMakeLists.txt" AND NOT DEFINED FETCHCONTENT_SOURCE_DIR_KEEL)
        set(FETCHCONTENT_SOURCE_DIR_KEEL "${CMAKE_CURRENT_SOURCE_DIR}/../keel" CACHE PATH "Local keel checkout")
    endif()
    if(KEEL_STATS_NO_EXCEPTIONS_RTTI)
        set(KEEL_NO_EXCEPTIONS_RTTI ON CACHE BOOL "" FORCE)
    endif()
    FetchContent_Declare(
        keel
        GIT_REPOSITORY https://github.com/keelcpp/keel.git
        GIT_TAG        869a5c07a5d77341e5f6acbac5f8e024b274ea08
    )
    FetchContent_MakeAvailable(keel)
endif()

if(NOT TARGET keel::specfun)
    find_package(keel_specfun QUIET CONFIG)
    if(TARGET keel::keel_specfun AND NOT TARGET keel::specfun)
        add_library(keel::specfun ALIAS keel::keel_specfun)
    endif()
endif()
if(NOT TARGET keel::specfun)
    include(FetchContent)
    set(KEEL_SPECFUN_BUILD_TESTS OFF CACHE BOOL "" FORCE)
    if(KEEL_STATS_NO_EXCEPTIONS_RTTI)
        set(KEEL_SPECFUN_NO_EXCEPTIONS_RTTI ON CACHE BOOL "" FORCE)
    endif()
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/../specfun/CMakeLists.txt" AND NOT DEFINED FETCHCONTENT_SOURCE_DIR_KEEL_SPECFUN)
        set(FETCHCONTENT_SOURCE_DIR_KEEL_SPECFUN "${CMAKE_CURRENT_SOURCE_DIR}/../specfun" CACHE PATH "Local keel_specfun checkout")
    endif()
    FetchContent_Declare(
        keel_specfun
        GIT_REPOSITORY https://github.com/keelcpp/specfun.git
        GIT_TAG        04f170a4554f2e1c85504222ee56c7b53152d035
    )
    FetchContent_MakeAvailable(keel_specfun)
endif()

add_library(keel_stats STATIC
    src/distributions.cpp
    src/descriptive.cpp
    src/distance.cpp
    src/linkage.cpp
    src/lasso.cpp
    src/anova.cpp
)
add_library(keel::stats ALIAS keel_stats)

target_link_libraries(keel_stats PUBLIC keel::keel keel::specfun)

target_compile_features(keel_stats PUBLIC cxx_std_20)
set_target_properties(keel_stats PROPERTIES
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED ON
    CXX_EXTENSIONS OFF
    POSITION_INDEPENDENT_CODE ON
)

target_include_directories(keel_stats PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

if(MSVC)
    target_compile_options(keel_stats PRIVATE /W4 /utf-8)
    if(KEEL_STATS_NO_EXCEPTIONS_RTTI)
        string(REGEX REPLACE "/EH[a-z]+" "" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
        string(REGEX REPLACE "/GR" "" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
        target_compile_options(keel_stats PUBLIC /EHs-c- /GR-)
        target_compile_definitions(keel_stats PUBLIC _HAS_EXCEPTIONS=0)
    endif()
else()
    target_compile_options(keel_stats PRIVATE -Wall -Wextra -Wpedantic)
    if(KEEL_STATS_NO_EXCEPTIONS_RTTI)
        target_compile_options(keel_stats PUBLIC -fno-exceptions -fno-rtti)
    endif()
endif()

if(COMMAND keel_enforce_guidelines)
    keel_enforce_guidelines(keel_stats
        SCOPE ALL
        BAN_RAW_INTS ALL
        BAN_RAW_FLOATS ALL
        HEADERS_DIR "${CMAKE_CURRENT_SOURCE_DIR}/include"
        STANDALONE_HEADERS ${KEEL_STATS_BUILD_TESTS}
        TEST_NAME test_guidelines_headers
        WARNINGS OFF
    )
endif()

include(CMakePackageConfigHelpers)
install(TARGETS keel_stats EXPORT keel_statsTargets
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
    RUNTIME DESTINATION bin
)
install(DIRECTORY include/keel DESTINATION include)
install(EXPORT keel_statsTargets
    FILE keel_statsTargets.cmake
    NAMESPACE keel::
    DESTINATION lib/cmake/keel_stats
)
configure_package_config_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/keel_statsConfig.cmake.in"
    "${CMAKE_CURRENT_BINARY_DIR}/keel_statsConfig.cmake"
    INSTALL_DESTINATION lib/cmake/keel_stats
)
write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/keel_statsConfigVersion.cmake"
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMinorVersion
)
install(FILES
    "${CMAKE_CURRENT_BINARY_DIR}/keel_statsConfig.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/keel_statsConfigVersion.cmake"
    DESTINATION lib/cmake/keel_stats
)

if(KEEL_STATS_BUILD_TESTS)
    enable_testing()

    add_executable(test_stats tests/test_stats.cpp)
    target_link_libraries(test_stats PRIVATE keel::stats)
    if(MSVC)
        target_compile_options(test_stats PRIVATE /W4 /utf-8)
    else()
        target_compile_options(test_stats PRIVATE -Wall -Wextra -Wpedantic)
    endif()
    add_test(NAME test_stats COMMAND test_stats)

    set(KEEL_DEP_INC_CLEAN "")
    foreach(DEP_TGT keel::keel keel::specfun)
        get_target_property(DEP_INC_DIRS ${DEP_TGT} INTERFACE_INCLUDE_DIRECTORIES)
        foreach(IDIR IN LISTS DEP_INC_DIRS)
            string(REGEX REPLACE "\\$<BUILD_INTERFACE:([^>]+)>" "\\1" IDIR_STRIPPED "${IDIR}")
            if(NOT IDIR_STRIPPED MATCHES "\\$<INSTALL_INTERFACE:" AND EXISTS "${IDIR_STRIPPED}")
                list(APPEND KEEL_DEP_INC_CLEAN "${IDIR_STRIPPED}")
            endif()
        endforeach()
    endforeach()
    list(REMOVE_DUPLICATES KEEL_DEP_INC_CLEAN)
    string(REPLACE ";" "|" KEEL_DEP_INC_JOINED "${KEEL_DEP_INC_CLEAN}")
    string(REPLACE ";" "|" KEEL_STD_INC_JOINED "${CMAKE_CXX_STANDARD_INCLUDE_DIRECTORIES}")

    add_test(
        NAME test_header_purity
        COMMAND ${CMAKE_COMMAND}
            "-DKEEL_INCLUDE_DIR=${CMAKE_CURRENT_SOURCE_DIR}/include"
            "-DKEEL_DEP_INC_DIRS=${KEEL_DEP_INC_JOINED}"
            "-DCOMPILER_PATH=${CMAKE_CXX_COMPILER}"
            "-DCOMPILER_ID=${CMAKE_CXX_COMPILER_ID}"
            "-DSTD_INC_DIRS=${KEEL_STD_INC_JOINED}"
            "-DCMAKE_BINARY_DIR=${CMAKE_CURRENT_BINARY_DIR}"
            -P "${CMAKE_CURRENT_SOURCE_DIR}/cmake/CheckHeaderPurity.cmake"
    )
endif()
