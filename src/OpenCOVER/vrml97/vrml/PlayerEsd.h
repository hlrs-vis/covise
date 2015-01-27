/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _PLAYER_ESD_
#define _PLAYER_ESD_

#include "PlayerMix.h"

#ifdef HAVE_ESD
#include <esd.h>
#endif

namespace vrml
{

class VRMLEXPORT PlayerEsd : public PlayerMix
{
public:
#ifndef HAVE_ESD
    PlayerEsd(const Listener *listener, bool threaded, const std::string host)
        : PlayerMix(listener, threaded, 2)
    {
        (void)host;
    }
#else
    PlayerEsd(const Listener *listener, bool threaded, const std::string host);
    virtual ~PlayerEsd();

    virtual int writeFrames(const char *frames, int numFrames) const;
    virtual int getQueued() const;
    virtual int getWritable() const;
    virtual double getDelay() const;

protected:
    int esdFd;
#endif
};
}
#endif
