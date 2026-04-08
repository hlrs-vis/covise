#include "AlutContext.h"
#include <AL/alut.h>

using namespace opencover::audio;

int AlutContext::_refcount = 0;
bool AlutContext::is_initialized = false;
bool AlutContext::has_context = false;

AlutContext::AlutContext()
{
    _refcount++;
    if (_refcount == 1)
    {
        if (alutInit(nullptr, nullptr))
        {
            is_initialized = true;
            has_context = true;
        }
        else if (alutInitWithoutContext(nullptr, nullptr))
        {
            is_initialized = true;
            has_context = false;
        }
        else
        {
            is_initialized = false;
            has_context = false;
        }
    }
}

AlutContext::~AlutContext()
{
    _refcount--;

    if (_refcount == 0 && is_initialized)
    {
        alutExit();
        is_initialized = false;
        has_context = false;
    }
}
