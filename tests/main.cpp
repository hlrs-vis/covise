
#include "message_macros.h"
#include "tryPrint.h"
#include "CRB_EXEC.h"
#include "connection.h"
#include "messageExchange.h"
#include "syncVar.h"
#include "asyncVrb.h"
using namespace test;

#define TEST(t)                   \
    std::cerr << #t << std::endl; \
    t();

int main(int argc, char const *argv[])
{
    TEST(test_message_macros)
    TEST(test_tryPrint)
    TEST(test_crbExec)
    TEST(testConnectionShutdown)
    TEST(testSetupServerConnection)
    TEST(test_socket_write_receive)
    TEST(test_message_send_receive)
    TEST(testSyncVar)
    TEST(testAsyncVrb);

    std::cerr << "all tests succseeded!" << std::endl;
    return 0;

}
