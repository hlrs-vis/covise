
#include "message_macros.h"
#include "tryPrint.h"
#include "CRB_EXEC.h"
using namespace test;

#define TEST(t)                   \
    std::cerr << #t << std::endl; \
    t();

int main(int argc, char const *argv[])
{
    TEST(test_message_macros)
    TEST(test_tryPrint)
    TEST(test_crbExec)
    std::cerr << "all tests succseeded!" << std::endl;
    return 0;

}
