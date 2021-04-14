

#include <thread>
#include <chrono>
#include <iostream>
#include <mutex>

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

void testSetupServerConnection()
{
    std::mutex m;
    m.lock();
    int port = 0;
    std::thread t{[&m, &port]() {
        m.lock();
        covise::ClientConnection c{nullptr, port, 1000, 2};
        if (!c.is_connected())
        {
            throw;
        }
        
    }};
    auto sconn = setupServerConnection(1000, 2, 0, [&m, &port](const covise::ServerConnection &c) {
        port = c.get_port();
        m.unlock();
        return true;
    });
    if (!sconn)
    {
        throw;
    }
    t.join();
}
}