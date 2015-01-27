/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef PLAYER_OSS_H
#define PLAYER_OSS_H

#include "PlayerMix.h"

namespace vrml
{

class VRMLEXPORT PlayerOSS : public PlayerMix
{
public:
#ifndef HAVE_OSS
    PlayerOSS(const Listener *listener, bool threaded = true, int
                                                                  channels = 2,
              int rate = 44100, int bps = 16, std::string device = "")
        : PlayerMix(listener, threaded, channels, rate, bps)
    {
        (void)device;
    };
#else
    PlayerOSS(const Listener *listener, bool threaded = true, int channels = 2, int rate = 44100, int bps = 16, std::string device = "");
    virtual ~PlayerOSS();

    virtual int writeFrames(const char *buf, int numFrames) const;
    virtual int getQueued() const;
    virtual int getWritable() const;
    virtual double getDelay() const;

protected:
    int dsp_fd;
#endif
};
}
#endif
