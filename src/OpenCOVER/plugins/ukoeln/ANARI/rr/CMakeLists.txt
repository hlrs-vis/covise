project(MiniRR LANGUAGES CXX)
cmake_minimum_required(VERSION 3.22)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Threads REQUIRED)
find_package(libjpeg-turbo REQUIRED)
#find_package(Snappy REQUIRED)
find_package(Boost COMPONENTS iostreams system REQUIRED)

add_library(minirr SHARED)

set(SOURCES
    async/connection.cpp
    async/connection_manager.cpp
    async/message.cpp
    MiniRR.cpp
    Buffer.cpp
    Compression.cpp)
target_sources(minirr PUBLIC ${SOURCES})
target_link_libraries(minirr PUBLIC
    Boost::iostreams
    Boost::system
    libjpeg-turbo::turbojpeg
    ${CMAKE_THREAD_LIBS_INIT}
)
target_compile_definitions(minirr PUBLIC -DHAVE_TURBOJPEG=1)
target_include_directories(minirr PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
  #$<INSTALL_INTERFACE:include>
)
covise_export_target(minirr)

option(MINIRR_BUILD_TEST_APP "mini test app for debugging" OFF)
if (MINIRR_BUILD_TEST_APP)
  add_executable(minirrTest)
  target_sources(minirrTest PRIVATE ${SOURCES})
  target_link_libraries(minirrTest PUBLIC
      Boost::iostreams
      Boost::system
      libjpeg-turbo::turbojpeg
      ${CMAKE_THREAD_LIBS_INIT}
  )
  target_compile_definitions(minirrTest PUBLIC -DHAVE_TURBOJPEG=1)
endif()
