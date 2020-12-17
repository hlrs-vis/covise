#include "coSpawnProgram.h"
#include <util/coSpawnProgram.h>
#include <cassert>

void test::test_coSpawnProgram(){

    auto args = covise::parseCmdArgString("test1 -test2 :test3;");
    assert(args[0] == "test1");
    assert(args[1] == "-test2");
    assert(args[2] == ":test3;");
}
