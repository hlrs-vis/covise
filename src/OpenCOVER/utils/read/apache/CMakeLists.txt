# set(ARROW_CXXFLAGS "" CACHE STRING "Compiler flags to append when compiling Arrow" FORCE)
option(BUILD_ARROW "Build Apache Arrow" OFF)

if (NOT BUILD_ARROW)
    message(STATUS "Apache Arrow build is disabled. Set BUILD_ARROW to ON to enable.")
    return()
endif()

set(ARROW_BUILD_STATIC OFF CACHE BOOL "Build static libraries" FORCE)
set(ARROW_BUILD_SHARED ON CACHE BOOL "Build shared libraries" FORCE)
set(ARROW_PACKAGE_KIND "" CACHE STRING "Arbitrary string that identifies the kind of package" FORCE)
set(ARROW_POSITION_INDEPENDENT_CODE ON CACHE BOOL "Whether to create position-independent target" FORCE)
set(ARROW_USE_CCACHE ON CACHE BOOL "Use ccache when compiling (if available)" FORCE)
set(ARROW_USE_SCCACHE ON CACHE BOOL "Use sccache when compiling (if available)," FORCE)
set(ARROW_USE_LD_GOLD OFF CACHE BOOL "Use ld.gold for linking on Linux (if available)" FORCE)
set(ARROW_USE_LLD OFF CACHE BOOL "Use the LLVM lld for linking (if available)" FORCE)
set(ARROW_USE_MOLD OFF CACHE BOOL "Use mold for linking on Linux (if available)" FORCE)
set(ARROW_SIMD_LEVEL SSE4_2 CACHE STRING "Compile-time SIMD optimization level" FORCE)
set(ARROW_RUNTIME_SIMD_LEVEL MAX CACHE STRING "Max runtime SIMD optimization level" FORCE)
set(ARROW_ALTIVEC ON CACHE BOOL "Build with Altivec if compiler has support" FORCE)
set(ARROW_RPATH_ORIGIN OFF CACHE BOOL "Build Arrow libraries with RPATH set to $ORIGIN" FORCE)
set(ARROW_INSTALL_NAME_RPATH ON CACHE BOOL "Build Arrow libraries with install_name set to @rpath" FORCE)
set(ARROW_GGDB_DEBUG ON CACHE BOOL "Pass -ggdb flag to debug builds" FORCE)
set(ARROW_WITH_MUSL OFF CACHE BOOL "Whether the system libc is musl or not" FORCE)
set(ARROW_ENABLE_THREADING ON CACHE BOOL "Enable threading in Arrow core" FORCE)

set(ARROW_BUILD_EXAMPLES OFF CACHE BOOL "Build the Arrow examples" FORCE)
set(ARROW_BUILD_TESTS OFF CACHE BOOL "Build the Arrow googletest unit tests" FORCE)
set(ARROW_ENABLE_TIMING_TESTS OFF CACHE BOOL "Enable timing-sensitive tests" FORCE)
set(ARROW_BUILD_INTEGRATION OFF CACHE BOOL "Build the Arrow integration test executables" FORCE)
set(ARROW_BUILD_BENCHMARKS OFF CACHE BOOL "Build the Arrow micro benchmarks" FORCE)
set(ARROW_BUILD_BENCHMARKS_REFERENCE OFF CACHE BOOL "Build the Arrow micro reference benchmarks" FORCE)
set(ARROW_BUILD_DETAILED_BENCHMARKS OFF CACHE BOOL "Build benchmarks that do a longer exploration of performance" FORCE)
set(ARROW_TEST_LINKAGE shared CACHE STRING "Linkage of Arrow libraries with unit tests executables." FORCE)
set(ARROW_FUZZING OFF CACHE BOOL "Build Arrow Fuzzing executables" FORCE)
set(ARROW_LARGE_MEMORY_TESTS OFF CACHE BOOL "Enable unit tests which use large memory" FORCE)

# Coverage options:
set(ARROW_GENERATE_COVERAGE OFF CACHE BOOL "Build with C++ code coverage enabled" FORCE)

# Checks options:
set(ARROW_TEST_MEMCHECK OFF CACHE BOOL "Run the test suite using valgrind --tool=memcheck" FORCE)
set(ARROW_USE_ASAN OFF CACHE BOOL "Enable Address Sanitizer checks" FORCE)
set(ARROW_USE_TSAN OFF CACHE BOOL "Enable Thread Sanitizer checks" FORCE)
set(ARROW_USE_UBSAN OFF CACHE BOOL "Enable Undefined Behavior sanitizer checks" FORCE)

# Project component options:
set(ARROW_ACERO OFF CACHE BOOL "Build the Arrow Acero Engine Module" FORCE)
set(ARROW_AZURE OFF CACHE BOOL "Build Arrow with Azure support (requires the Azure SDK for C++)" FORCE)
set(ARROW_BUILD_UTILITIES OFF CACHE BOOL "Build Arrow commandline utilities" FORCE)
set(ARROW_COMPUTE OFF CACHE BOOL "Build all Arrow Compute kernels" FORCE)
set(ARROW_CSV OFF CACHE BOOL "Build the Arrow CSV Parser Module" FORCE)
set(ARROW_CUDA OFF CACHE BOOL "Build the Arrow CUDA extensions (requires CUDA toolkit)" FORCE)
set(ARROW_DATASET OFF CACHE BOOL "Build the Arrow Dataset Modules" FORCE)
set(ARROW_FILESYSTEM OFF CACHE BOOL "Build the Arrow Filesystem Layer" FORCE)
set(ARROW_FLIGHT OFF CACHE BOOL "Build the Arrow Flight RPC System (requires GRPC, Protocol Buffers)" FORCE)
set(ARROW_FLIGHT_SQL OFF CACHE BOOL "Build the Arrow Flight SQL extension" FORCE)
set(ARROW_FLIGHT_SQL_ODBC OFF CACHE BOOL "Build the Arrow Flight SQL ODBC extension" FORCE)
set(ARROW_GANDIVA OFF CACHE BOOL "Build the Gandiva libraries" FORCE)
set(ARROW_GCS OFF CACHE BOOL "Build Arrow with GCS support (requires the Google Cloud Platform C++ Client Libraries)" FORCE)
set(ARROW_HDFS OFF CACHE BOOL "Build the Arrow HDFS bridge" FORCE)
set(ARROW_IPC ON CACHE BOOL "Build the Arrow IPC extensions" FORCE)
set(ARROW_JEMALLOC OFF CACHE BOOL "Build the Arrow jemalloc-based allocator" FORCE)
set(ARROW_JSON OFF CACHE BOOL "Build Arrow with JSON support (requires RapidJSON)" FORCE)
set(ARROW_MIMALLOC ON CACHE BOOL "Build the Arrow mimalloc-based allocator" FORCE)
set(ARROW_PARQUET OFF CACHE BOOL "Build the Parquet libraries" FORCE)
set(ARROW_ORC OFF CACHE BOOL "Build the Arrow ORC adapter" FORCE)
set(ARROW_PYTHON OFF CACHE BOOL "Build some components needed by PyArrow (deprecated)" FORCE)
set(ARROW_S3 OFF CACHE BOOL "Build Arrow with S3 support (requires the AWS SDK for C++)" FORCE)
set(ARROW_S3_MODULE OFF CACHE BOOL "Build the Arrow S3 filesystem as a dynamic module" FORCE)
set(ARROW_SKYHOOK OFF CACHE BOOL "Build the Skyhook libraries" FORCE)
set(ARROW_SUBSTRAIT OFF CACHE BOOL "Build the Arrow Substrait Consumer Module" FORCE)
set(ARROW_TENSORFLOW OFF CACHE BOOL "Build Arrow with TensorFlow support enabled" FORCE)
set(ARROW_TESTING OFF CACHE BOOL "Build the Arrow testing libraries" FORCE)

# Thirdparty toolchain options:
set(ARROW_DEPENDENCY_SOURCE AUTO CACHE STRING "Method to use for acquiring arrow's build dependencies" FORCE)
set(ARROW_VERBOSE_THIRDPARTY_BUILD OFF CACHE BOOL "Show output from ExternalProjects rather than just logging to files" FORCE)
set(ARROW_DEPENDENCY_USE_SHARED ON CACHE BOOL "Link to shared libraries" FORCE)
set(ARROW_BOOST_USE_SHARED ON CACHE BOOL "Rely on Boost shared libraries where relevant" FORCE)
set(ARROW_BROTLI_USE_SHARED ON CACHE BOOL "Rely on Brotli shared libraries where relevant" FORCE)
set(ARROW_BZ2_USE_SHARED ON CACHE BOOL "Rely on Bz2 shared libraries where relevant" FORCE)
set(ARROW_GFLAGS_USE_SHARED ON CACHE BOOL "Rely on GFlags shared libraries where relevant" FORCE)
set(ARROW_GRPC_USE_SHARED ON CACHE BOOL "Rely on gRPC shared libraries where relevant" FORCE)
set(ARROW_JEMALLOC_USE_SHARED ON CACHE BOOL "Rely on jemalloc shared libraries where relevant" FORCE)
set(ARROW_LLVM_USE_SHARED ON CACHE BOOL "Rely on LLVM shared libraries where relevant" FORCE)
set(ARROW_LZ4_USE_SHARED ON CACHE BOOL "Rely on lz4 shared libraries where relevant" FORCE)
set(ARROW_OPENSSL_USE_SHARED ON CACHE BOOL "Rely on OpenSSL shared libraries where relevant" FORCE)
set(ARROW_PROTOBUF_USE_SHARED ON CACHE BOOL "Rely on Protocol Buffers shared libraries where relevant" FORCE)
set(ARROW_SNAPPY_USE_SHARED ON CACHE BOOL "Rely on snappy shared libraries where relevant" FORCE)
set(ARROW_THRIFT_USE_SHARED ON CACHE BOOL "Rely on thrift shared libraries where relevant" FORCE)
set(ARROW_UTF8PROC_USE_SHARED ON CACHE BOOL "Rely on utf8proc shared libraries where relevant" FORCE)
set(ARROW_ZSTD_USE_SHARED ON CACHE BOOL "Rely on zstd shared libraries where relevant" FORCE)
set(ARROW_USE_GLOG OFF CACHE BOOL "Build libraries with glog support for pluggable logging" FORCE)
set(ARROW_WITH_BACKTRACE ON CACHE BOOL "Build with backtrace support" FORCE)
set(ARROW_WITH_OPENTELEMETRY OFF CACHE BOOL "Build libraries with OpenTelemetry support for distributed tracing" FORCE)
set(ARROW_WITH_BROTLI OFF CACHE BOOL "Build with Brotli compression" FORCE)
set(ARROW_WITH_BZ2 OFF CACHE BOOL "Build with BZ2 compression" FORCE)
set(ARROW_WITH_LZ4 OFF CACHE BOOL "Build with lz4 compression" FORCE)
set(ARROW_WITH_SNAPPY OFF CACHE BOOL "Build with Snappy compression" FORCE)
set(ARROW_WITH_ZLIB OFF CACHE BOOL "Build with zlib compression" FORCE)
set(ARROW_WITH_ZSTD OFF CACHE BOOL "Build with zstd compression" FORCE)
set(ARROW_WITH_UTF8PROC OFF CACHE BOOL "Build with support for Unicode properties using the utf8proc library" FORCE)
set(ARROW_WITH_RE2 OFF CACHE BOOL "Build with support for regular expressions using the re2 library" FORCE)

# Parquet options:
set(PARQUET_BUILD_EXECUTABLES OFF CACHE BOOL "Build the Parquet executable CLI tools. Requires static libraries to be built." FORCE)
set(PARQUET_BUILD_EXAMPLES OFF CACHE BOOL "Build the Parquet examples. Requires static libraries to be built." FORCE)
set(PARQUET_REQUIRE_ENCRYPTION OFF CACHE BOOL "Build support for encryption. Fail if OpenSSL is not found" FORCE)

# Gandiva options:
set(ARROW_GANDIVA_STATIC_LIBSTDCPP OFF CACHE BOOL "Include -static-libstdc++ -static-libgcc when linking with Gandiva static libraries" FORCE)
set(ARROW_GANDIVA_PC_CXX_FLAGS "" CACHE STRING "Compiler flags to append when pre-compiling Gandiva operations" FORCE)

# Cross compiling options:
set(ARROW_GRPC_CPP_PLUGIN "" CACHE STRING "grpc_cpp_plugin path to be used" FORCE)

# Advanced developer options:
set(ARROW_EXTRA_ERROR_CONTEXT OFF CACHE BOOL "Compile with extra error context (line numbers, code)" FORCE)
set(ARROW_OPTIONAL_INSTALL OFF CACHE BOOL "If enabled install ONLY targets that have already been built" FORCE)
set(ARROW_GDB_INSTALL_DIR "" CACHE STRING "Use a custom install directory for GDB plugin" FORCE)

# The COVISE project defines BYTESWAP globally, which conflicts with Arrow's internal definition.
remove_definitions(-DBYTESWAP)
include(FetchContent)

FetchContent_Declare(
    Arrow
    GIT_REPOSITORY "https://github.com/apache/arrow.git"
    GIT_TAG "apache-arrow-21.0.0"
    GIT_SHALLOW TRUE #copy version and not git history
	SOURCE_SUBDIR "cpp"
)
FetchContent_MakeAvailable(Arrow)

# --- Diagnostic Code ---
if(TARGET arrow::arrow_shared)
    message(STATUS "Found namespaced target 'arrow::arrow_shared'")
    set(ARROW_TARGET_NAME arrow::arrow_shared)
elseif(TARGET arrow_shared)
    message(STATUS "Found target 'arrow_shared'")
    set(ARROW_TARGET_NAME arrow_shared)
else()
    message(FATAL_ERROR "Could not find either 'arrow::arrow_shared' or 'arrow_shared'. Check Arrow's CMake configuration.")
    return()
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
target_include_directories(${Name}
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        $<BUILD_INTERFACE:${arrow_SOURCE_DIR}/cpp/src>
        $<BUILD_INTERFACE:${arrow_BINARY_DIR}/src>
)
COVISE_INSTALL_TARGET(${Name})
COVISE_INSTALL_HEADERS(${Name} ${HEADERS})
COVISE_WNOERROR(${Name} BASEARCH zebu)
set_target_properties(${Name} PROPERTIES LINKER_LANGUAGE CXX)