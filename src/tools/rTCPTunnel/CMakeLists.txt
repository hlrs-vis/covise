# PROJECT(rTcpTunnel)

set(HEADERS
   tcpsock.h
)

set(SOURCES
   tcpsock.cpp
)

add_covise_executable(rTcpClient rTcpClient.cpp rTcpClient.h tcpsock.cpp tcpsock.h)
add_covise_executable(rTcpServer rTcpServer.cpp rTcpServer.h tcpsock.cpp tcpsock.h)

covise_install_target(rTcpClient)
covise_install_target(rTcpServer)
