

#include <thread>
#include <chrono>
#include <iostream>

#include "connection.h"
#include <net/covise_connect.h>
#include <net/covise_socket.h>

using namespace covise;
namespace test
{
    void testConnectionShutdown()
    {
        int port = 0;
        auto conn = createListeningConn<ServerConnection>(&port, 1001, 1);
        std::thread thread{
            [&conn]() {
                int timeout = 0; // do not timeout
                std::cerr << "waiting for accept" << std::endl;
                if (conn->acceptOne(timeout) < 0)
                {
                    std::cerr << "accept failed (as planned)" << std::endl;
                }
            }};

        std::this_thread::sleep_for(std::chrono::microseconds(200));
        std::cerr << "trying to shutdown connection" << std::endl;
        ::shutdown(conn->getSocket()->get_id(), 2);
        thread.join();
        std::cerr << "connection shutdown successful" << std::endl;
    }
}