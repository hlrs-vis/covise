/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef PLAYER_ALSA_H
#define PLAYER_ALSA_H

#include "PlayerMix.h"

#ifdef HAVE_ALSA
#include <alsa/asoundlib.h>
#endif

namespace vrml
{

class VRMLEXPORT PlayerAlsa : public PlayerMix
{
public:
#ifndef HAVE_ALSA
    PlayerAlsa(const Listener *listener, bool threaded = true, int
                                                                   channels = 2,
               int rate = 44100, int bps = 16, const std::string device = "")
        : PlayerMix(listener, threaded, channels, rate, bps)
    {
        (void)device;
    }
#else
    PlayerAlsa(const Listener *listener, bool threaded = true, int channels = 2, int rate = 44100, int bps = 16, std::string = "");
    virtual ~PlayerAlsa();

    virtual int writeFrames(const char *buf, int numFrames) const;
    virtual int getQueued() const;
    virtual int getWritable() const;
    virtual double getDelay() const;

protected:
    snd_pcm_t *alsaHandle;
    snd_pcm_status_t *alsaStatus;
#endif
};
}
#endif
