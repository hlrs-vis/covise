#include "asyncVrb.h"
#include <vrb/client/AsyncClient.h>
#include <net/program_type.h>
#include <messages/PROXY.h>
#include <util/asyncWait.h>
#include <iostream>
using namespace covise;

void test::testAsyncVrb()
{
    std::cerr << "testing async client" << std::endl;

    auto &test1 = AsyncWait<bool>(
        []()
        { return true; },
        [](bool b)
        {
            return true;
        });

    auto &test2 = AsyncWait<bool>(
        []()
        { return true; },
        [](bool b)
        {
            return true;
        });
    test1 >> test2;
    test1.wait();

    while (!asyncWaits.empty())
    {
        static size_t size = 0;
        if (asyncWaits.size() != size)
        {
            size = asyncWaits.size();
            std::cerr << "asyncWaits.size() = " << size << std::endl;
        }

        handleAsyncWaits();
    }
}
