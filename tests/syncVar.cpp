
#include "syncVar.h"
#include "../sys/controller/syncVar.h"

#include <thread>
void test::testSyncVar()
{
    covise::controller::SyncVar<bool> test1, test2;
    std::thread t{[&test1, &test2]() {
        test1.waitForValue();
        test2.setValue(true);

    }};
    test1.setValue(true);
    test2.waitForValue();
    t.join();
}