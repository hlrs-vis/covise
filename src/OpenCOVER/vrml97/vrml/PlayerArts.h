/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _PLAYER_ARTS_
#define _PLAYER_ARTS_

#include "PlayerMix.h"

#ifdef HAVE_ARTS
#include <artsc/artsc.h>
#endif

namespace vrml
{

class VRMLEXPORT PlayerArts : public PlayerMix
{
public:
#ifndef HAVE_ARTS
    PlayerArts(const Listener *listener, bool threaded, int channels = 2, int rate = 44100, int bps = 16)
        : PlayerMix(listener, threaded, 2)
    {
        (void)channels;
        (void)rate;
        (void)bps;
    }
#else
    PlayerArts(const Listener *listener, bool threaded, int channels = 2, int rate = 44100, int bps = 16);
    virtual ~PlayerArts();

    virtual int writeFrames(const char *frames, int numFrames) const;
    virtual int getQueued() const;
    virtual int getWritable() const;
    virtual double getDelay() const;
    virtual int getPacketSize() const;

protected:
    arts_stream_t arts_stream;
#endif
};
}
#endif
