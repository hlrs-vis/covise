
#include "message_macros.h"
#include "tryPrint.h"

using namespace test;

int main(int argc, char const *argv[])
{
    test_message_macros();
    test_tryPrint();
    std::cerr << "all tests succseeded!" << std::endl;
    return 0;

}
