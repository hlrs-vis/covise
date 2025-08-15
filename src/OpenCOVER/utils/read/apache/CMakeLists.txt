option(BUILD_ARROW "Build Apache Arrow" OFF)

if (NOT BUILD_ARROW)
    message(STATUS "Apache Arrow build is disabled. Set BUILD_ARROW to ON to enable.")
    return()
endif()

find_package(Arrow QUIET CONFIG)

if (NOT Arrow_FOUND)
    message(STATUS "Arrow not found. Fetching and building from source.")
    # The COVISE project defines BYTESWAP globally, which conflicts with Arrow's internal definition.
    remove_definitions(-DBYTESWAP)
    include(FetchContent)
    
    FetchContent_Declare(
        Arrow
        GIT_REPOSITORY "https://github.com/apache/arrow.git"
        GIT_TAG "apache-arrow-21.0.0"
        GIT_SHALLOW TRUE #copy version and not git history
    	SOURCE_SUBDIR "cpp"
        CMAKE_ARGS
            -DARROW_PARQUET=OFF
            -DARROW_WITH_SNAPPY=OFF
            -DARROW_WITH_ZSTD=OFF
            -DARROW_WITH_LZ4=OFF
            -DARROW_BUILD_TESTS=OFF
            -DARROW_BUILD_BENCHMARKS=OFF
            -DARROW_BUILD_UTILITIES=OFF
            -DARROW_USE_XSIMD=ON
            -DARROW_BUILD_STATIC=ON
            -DARROW_PACKAGE_KIND=""
            -DARROW_POSITION_INDEPENDENT_CODE=ON
            -DARROW_USE_CCACHE=ON
            -DARROW_USE_SCCACHE=ON
            -DARROW_USE_LD_GOLD=OFF
            -DARROW_USE_LLD=OFF
            -DARROW_USE_MOLD=OFF
            -DARROW_SIMD_LEVEL=SSE4_2
            -DARROW_RUNTIME_SIMD_LEVEL=MAX
            -DARROW_ALTIVEC=ON
            -DARROW_RPATH_ORIGIN=OFF
            -DARROW_INSTALL_NAME_RPATH=ON
            -DARROW_GGDB_DEBUG=ON
            -DARROW_WITH_MUSL=OFF
            -DARROW_ENABLE_THREADING=ON
            -DARROW_DEFINE_OPTIONS=ON
            -DARROW_BUILD_EXAMPLES=OFF
            -DARROW_BUILD_TESTS=OFF
            -DARROW_ENABLE_TIMING_TESTS=OFF
            -DARROW_BUILD_INTEGRATION=OFF
            -DARROW_BUILD_BENCHMARKS=OFF
            -DARROW_BUILD_BENCHMARKS_REFERENCE=OFF
            -DARROW_BUILD_DETAILED_BENCHMARKS=OFF
            -DARROW_TEST_LINKAGE=shared
            -DARROW_FUZZING=OFF
            -DARROW_LARGE_MEMORY_TESTS=OFF
            -DARROW_GENERATE_COVERAGE=OFF
            -DARROW_TEST_MEMCHECK=OFF
            -DARROW_USE_ASAN=OFF
            -DARROW_USE_TSAN=OFF
            -DARROW_USE_UBSAN=OFF
            -DARROW_ACERO=OFF
            -DARROW_AZURE=OFF
            -DARROW_BUILD_UTILITIES=OFF
            -DARROW_COMPUTE=OFF
            -DARROW_CSV=OFF
            -DARROW_CUDA=OFF
            -DARROW_DATASET=OFF
            -DARROW_FILESYSTEM=OFF
            -DARROW_FLIGHT=OFF
            -DARROW_FLIGHT_SQL=OFF
            -DARROW_FLIGHT_SQL_ODBC=OFF
            -DARROW_GANDIVA=OFF
            -DARROW_GCS=OFF
            -DARROW_HDFS=OFF
            -DARROW_IPC=ON
            -DARROW_JEMALLOC=OFF
            -DARROW_JSON=OFF
            -DARROW_MIMALLOC=ON
            -DARROW_PARQUET=OFF
            -DARROW_ORC=OFF
            -DARROW_PYTHON=OFF
            -DARROW_S3=OFF
            -DARROW_S3_MODULE=OFF
            -DARROW_SKYHOOK=OFF
            -DARROW_SUBSTRAIT=OFF
            -DARROW_TENSORFLOW=OFF
            -DARROW_TESTING=OFF
            -DARROW_DEPENDENCY_SOURCE=AUTO
            -DARROW_VERBOSE_THIRDPARTY_BUILD=OFF
            -DARROW_DEPENDENCY_USE_SHARED=ON
            -DARROW_BOOST_USE_SHARED=ON
            -DARROW_BROTLI_USE_SHARED=ON
            -DARROW_BZ2_USE_SHARED=ON
            -DARROW_GFLAGS_USE_SHARED=ON
            -DARROW_GRPC_USE_SHARED=ON
            -DARROW_JEMALLOC_USE_SHARED=ON
            -DARROW_LLVM_USE_SHARED=ON
            -DARROW_LZ4_USE_SHARED=ON
            -DARROW_OPENSSL_USE_SHARED=ON
            -DARROW_PROTOBUF_USE_SHARED=ON
            -DARROW_SNAPPY_USE_SHARED=ON
            -DARROW_THRIFT_USE_SHARED=ON
            -DARROW_UTF8PROC_USE_SHARED=ON
            -DARROW_ZSTD_USE_SHARED=ON
            -DARROW_USE_GLOG=OFF
            -DARROW_WITH_BACKTRACE=ON
            -DARROW_WITH_OPENTELEMETRY=OFF
            -DARROW_WITH_BROTLI=OFF
            -DARROW_WITH_BZ2=OFF
            -DARROW_WITH_LZ4=OFF
            -DARROW_WITH_SNAPPY=OFF
            -DARROW_WITH_ZLIB=OFF
            -DARROW_WITH_ZSTD=OFF
            -DARROW_WITH_UTF8PROC=OFF
            -DARROW_WITH_RE2=OFF
            -DPARQUET_BUILD_EXECUTABLES=OFF
            -DPARQUET_BUILD_EXAMPLES=OFF
            -DPARQUET_REQUIRE_ENCRYPTION=OFF
            -DARROW_GANDIVA_STATIC_LIBSTDCPP=OFF
            -DARROW_GANDIVA_PC_CXX_FLAGS=""
            -DARROW_GRPC_CPP_PLUGIN=""
            -DARROW_EXTRA_ERROR_CONTEXT=OFF
            -DARROW_OPTIONAL_INSTALL=OFF
            -DARROW_GDB_INSTALL_DIR=""
    )
else()
    message(STATUS "Found pre-installed Arrow. Using imported targets.")
    set(ARROW_TARGET_NAME arrow_shared)
    set(ARROW_BUILD FALSE)
endif()

set(SOURCES
	arrow.cpp
)

set(HEADERS
	arrow.h
	enums.h
    export.h
)

set(Name "coApacheArrow")

ADD_COVISE_LIBRARY(${Name} ${SOURCES} ${HEADERS})
target_link_libraries(${Name} ${ARROW_TARGET_NAME})
if (ARROW_BUILD)
    target_include_directories(${Name}
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        $<BUILD_INTERFACE:${ARROW_SOURCE_DIR}>
        $<BUILD_INTERFACE:${ARROW_BINARY_DIR}>
    )
endif()
COVISE_INSTALL_TARGET(${Name})
COVISE_INSTALL_HEADERS(${Name} ${HEADERS})
COVISE_WNOERROR(${Name} BASEARCH zebu)
set_target_properties(${Name} PROPERTIES LINKER_LANGUAGE CXX)
