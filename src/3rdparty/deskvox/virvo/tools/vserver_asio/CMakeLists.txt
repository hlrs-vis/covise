find_package(Boost COMPONENTS chrono system thread REQUIRED)
find_package(Pthreads REQUIRED)

deskvox_use_package(Boost)
deskvox_use_package(Pthreads)

add_definitions(-DHAVE_CONFIG_H)

deskvox_link_libraries(virvo)
deskvox_link_libraries(virvo_fileio)

deskvox_add_tool(vserver_asio
  main.cpp
  resource_manager.cpp
  resource_manager.h
  server.cpp
  server.h
  server_manager.cpp
  server_manager.h
  simple_server.cpp
  simple_server.h
)
