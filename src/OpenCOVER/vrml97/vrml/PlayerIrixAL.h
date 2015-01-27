/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _PLAYER_IRIXAL_
#define _PLAYER_IRIXAL_

#include "PlayerMix.h"

#ifdef HAVE_IRIXAL
#include <audio.h>
#endif

namespace vrml
{

class VRMLEXPORT PlayerIrixAL : public PlayerMix
{
public:
#ifndef HAVE_IRIXAL
    PlayerIrixAL(const Listener *listener, bool threaded = false, int channels = 2, int rate = 44100, int bps = 16)
        : PlayerMix(listener, threaded, channels, rate, bps){};
#else
    PlayerIrixAL(const Listener *listener, bool threaded = false, int channels = 2, int rate = 44100, int bps = 16);
    virtual ~PlayerIrixAL();
    virtual int writeFrames(const char *frame, int numFrames) const;
    virtual int getQueued() const;
    virtual int getWritable() const;
    virtual double getDelay() const;

protected:
    ALport irixPort;
#endif
};
}
#endif
