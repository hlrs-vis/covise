#include "AlutContext.h"

#ifdef HAVE_AUDIO
#include <AL/alut.h>
#endif

using namespace opencover::audio;

int AlutContext::_refcount = 0;
bool AlutContext::is_initialized = false;
bool AlutContext::has_context = false;

AlutContext::AlutContext()
{
    _refcount++;

    if (_refcount == 1)
    {
#ifdef HAVE_AUDIO
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
#else
        is_initialized = false;
        has_context = false;
#endif
    }
}

AlutContext::~AlutContext()
{
    _refcount--;

    if (_refcount == 0 && is_initialized)
    {
#ifdef HAVE_AUDIO
        alutExit();
#endif
        is_initialized = false;
        has_context = false;
    }
}
