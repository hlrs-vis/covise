
#include "message_macros.h"
#include "tryPrint.h"
#include "CRB_EXEC.h"

using namespace test;

int main(int argc, char const *argv[])
{
    test_message_macros();
    test_tryPrint();
    test_crbExec();
    std::cerr << "all tests succseeded!" << std::endl;
    return 0;

}
