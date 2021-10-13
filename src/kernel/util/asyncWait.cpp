#include "asyncWait.h"
#include <cassert>
#include <algorithm>

using namespace covise;

AsyncWaits covise::asyncWaits = AsyncWaits{};

void covise::handleAsyncWaits()
{
    for (auto i = asyncWaits.begin(); i != asyncWaits.end();)
    {
        auto ptr = i->get();
        if (ptr->operator()()) //this can change connectionAtempts therefore we have to search for i again and start from scratch
        {
           asyncWaits.erase(std::remove_if(asyncWaits.begin(), asyncWaits.end(), [ptr](const AsyncWaits::value_type &ca)
                             { return ca.get() == ptr; }));
           i = asyncWaits.begin();
        }
        else
            ++i;
    }
}