/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "Listener.h"
#include "PlayerEsd.h"

#ifdef HAVE_ESD
#include <esd.h>
#include <sys/fcntl.h>
#include <unistd.h>
using namespace vrml;

PlayerEsd::PlayerEsd(const Listener *listener, bool threaded, const std::string host)
    : PlayerMix(listener, threaded, 2)
    , esdFd(-1)
{
    int esd_flags = ESD_STREAM | ESD_PLAY;

    CERR << "channels=" << this->channels << ", bps=" << bps << endl;

    switch (bps)
    {
    case 8:
        esd_flags |= ESD_BITS8;
        break;
    case 16:
        esd_flags |= ESD_BITS16;
        break;
    default:
        CERR << "unhandled number of bits per sample" << endl;
        return;
    }

    switch (channels)
    {
    case 1:
        esd_flags |= ESD_MONO;
        break;
    case 2:
        esd_flags |= ESD_STEREO;
        break;
    default:
        CERR << "unhandled number of channels" << endl;
        return;
    }

    if (host.empty())
    {
        host = "localhost";
    }
    esdFd = esd_play_stream(esd_flags, rate, host.c_str(), NULL);

    if (!threaded && esdFd != -1)
        fcntl(esdFd, F_SETFL, O_NONBLOCK);

    startThread();
}

PlayerEsd::~PlayerEsd()
{
    stopThread();

    if (esdFd != -1)
        esd_close(esdFd);
}

int
PlayerEsd::writeFrames(const char *frames, int numFrames) const
{
    if (-1 == esdFd)
        return -1;

    int n = write(esdFd, frames, numFrames * bytesPerFrame);
    if (n >= 0)
        return n / bytesPerFrame;
    else
        return n;
}

int
PlayerEsd::getQueued() const
{
    // can we know how many frames are already queued to esd?
    return -1;
}

int
PlayerEsd::getWritable() const
{
    // can we know how many frames are writable to esd?
    return 0;
}

double
PlayerEsd::getDelay() const
{
    if (esdFd != -1)
        return (double)esd_get_latency(esdFd) / (double)rate;
    else
        return 0.0;
}
#endif
