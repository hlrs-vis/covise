/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "IgnitionLock.h"
#include <sys/time.h>

//Seat////////////////////////////////////////////////////////////////

// set protected static pointer for singleton to NULL
IgnitionLock *IgnitionLock::p_ignitionlock = NULL;

// constructor, destructor, instance ---------------------------------
IgnitionLock::IgnitionLock()
{
    p_beckhoff = Beckhoff::instance();
    unlock = false;
}

IgnitionLock::~IgnitionLock()
{
    p_ignitionlock = NULL;
}

// singleton
IgnitionLock *IgnitionLock::instance()
{
    if (p_ignitionlock == NULL)
    {
        p_ignitionlock = new IgnitionLock();
    }
    return p_ignitionlock;
}
//--------------------------------------------------------------------

// public methods ----------------------------------------------------
int IgnitionLock::getLockState()
{
    if (p_beckhoff->getDigitalIn(0, 3) && p_beckhoff->getDigitalIn(0, 4) && p_beckhoff->getDigitalIn(0, 5))
    {
        return KEYOUT;
    }
    if (p_beckhoff->getDigitalIn(0, 3) && p_beckhoff->getDigitalIn(0, 4))
    {
        return ENGINESTOP;
    }
    if (p_beckhoff->getDigitalIn(0, 3) && p_beckhoff->getDigitalIn(0, 5))
    {
        return IGNITION;
    }
    else if (p_beckhoff->getDigitalIn(0, 4))
    {
        return ENGINESTART;
    }
    return KEYIN;
}

void IgnitionLock::releaseKey()
{
    p_beckhoff->setDigitalOut(0, 2, true);

    timeval currentTime;
    gettimeofday(&currentTime, NULL);
    startTime = (currentTime.tv_sec + (double)currentTime.tv_usec / 1000000.0);
    unlock = true;
    return;
}

bool IgnitionLock::update()
{
    timeval currentTime;
    gettimeofday(&currentTime, NULL);
    double cTime = (currentTime.tv_sec + (double)currentTime.tv_usec / 1000000.0);
    if (unlock && cTime > startTime + 0.1)
    {
        p_beckhoff->setDigitalOut(0, 2, false);
        unlock = false;
    }
    return true;
}

//--------------------------------------------------------------------
