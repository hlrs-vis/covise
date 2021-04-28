#include "messageExchange.h"
#include <cassert>
#include <thread>
#include <atomic>
#include <mutex>
#include <net/covise_connect.h>
#include <net/covise_socket.h>
#include <net/covise_host.h>
#include <net/message_types.h>

using namespace covise;

void test::test_socket_write_receive()
{
#ifdef _WIN32
#define testSize 16000
#else
    constexpr size_t testSize = 16000;
#endif
    int port{-1};
    std::mutex m;
    m.lock();
    std::thread t1{[&port, &m]() {
        covise::ServerConnection conn{&port, 1000, 1};
        conn.listen();
        m.unlock();
        conn.acceptOne();
        int buf[testSize];
        conn.receive(buf, testSize * sizeof(int));
        for (size_t i = 0; i < testSize; i++)
        {
            assert(buf[i] == i);
        }
        
    }};

    std::thread t2{[&port, &m]() {
        m.unlock();
        Host h;
        covise::ClientConnection conn{&h, port, 1000, 1};
        assert(conn.is_connected());
        int buf[testSize];
        for (size_t i = 0; i < testSize; i++)
            buf[i] = i;
        conn.send(buf, testSize * sizeof(int));

    }};
    t1.join();
    t2.join();
}

void test::test_message_send_receive()
{
#ifdef _WIN32
#define testSize 1000000
#else
    constexpr size_t testSize = 1000000;
#endif
    int port{-1};
    std::mutex m;
    m.lock();
    std::thread t1{[&port, &m]() {
        covise::ServerConnection conn{&port, 1000, 1};
        conn.listen();
        m.unlock();
        conn.acceptOne();
        Message msg;
        conn.recv_msg(&msg);
        for (size_t i = 0; i < testSize; i++)
        {
            assert(((int*)msg.data.data())[i] == i);
        }
        
    }};

    std::thread t2{[&port, &m]() {
        m.unlock();
        Host h;
        covise::ClientConnection conn{&h, port, 1000, 1};
        assert(conn.is_connected());

        DataHandle dh{testSize * sizeof(int)};
        for (size_t i = 0; i < testSize; i++)
            ((int*)dh.accessData())[i] = i;
        Message msg{COVISE_MESSAGE_RENDER, dh};
        conn.sendMessage(&msg);

    }};
    t1.join();
    t2.join();
}

