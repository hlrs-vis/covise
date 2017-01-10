set(HEADERS
   tcpsock.h
)

set(SOURCES
   tcpsock.cpp
)

if(COVISE_BUILDING_COVISE)
    add_covise_executable(rTcpClient rTcpClient.cpp rTcpClient.h tcpsock.cpp tcpsock.h)
    add_covise_executable(rTcpServer rTcpServer.cpp rTcpServer.h tcpsock.cpp tcpsock.h)

    covise_install_target(rTcpClient)
    covise_install_target(rTcpServer)
else()
    project(rTcpTunnel)
    cmake_minimum_required(VERSION 2.8.11)

    add_executable(rTcpClient rTcpClient.cpp rTcpClient.h tcpsock.cpp tcpsock.h)
    add_executable(rTcpServer rTcpServer.cpp rTcpServer.h tcpsock.cpp tcpsock.h)

    install(TARGETS rTcpClient rTcpServer
        RUNTIME DESTINATION bin
        )
endif()
