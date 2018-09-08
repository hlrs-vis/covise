find_package(Boost COMPONENTS filesystem serialization system REQUIRED)
deskvox_use_package(Boost)

deskvox_link_libraries(virvo_fileio)

deskvox_add_tool(vconv
  vvconv.cpp
  vvconv.h
)
