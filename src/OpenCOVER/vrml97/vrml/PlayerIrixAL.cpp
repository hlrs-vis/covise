/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "Listener.h"
#include "PlayerIrixAL.h"

#ifdef HAVE_IRIXAL
#include <audio.h>
#include <errno.h>

using namespace vrml;

PlayerIrixAL::PlayerIrixAL(const Listener *listener, bool threaded, int channels, int rate, int bps)
    : PlayerMix(listener, threaded, channels, rate, bps)
    , irixPort(0)
{
    int dev = AL_DEFAULT_OUTPUT;
    int wsize = AL_SAMPLE_16;
    int sampfmt = AL_SAMPFMT_TWOSCOMP;
    int bufframes;

    ALconfig c;
    ALport p;
    ALpv pv;
    ALpv rpv[2];
    double old_rate;

    if (8 == this->bps)
    {
        wsize = AL_SAMPLE_8;
    }
    else if (16 == this->bps)
    {
        wsize = AL_SAMPLE_16;
    }
    else if (32 == this->bps || 64 == this->bps)
    {
        wsize = AL_SAMPLE_24;
        CERR << "unsupported sample width" << endl;
    }
    else
    {
        CERR << "unsupported sample width" << endl;
        return;
    }

    if (channels != 1 && channels != 2)
    {
        CERR << "unsupported number of channels" << endl;
        return;
    }

    pv.param = AL_RATE;
    if (alGetParams(dev, &pv, 1) < 0)
    {
        CERR << "alGetParams failed: " << alGetErrorString(oserror());
        return;
    }
    old_rate = alFixedToDouble(pv.value.ll);

    rpv[0].param = AL_PORT_COUNT;
    if (alGetParams(dev, rpv, 1) < 0)
    {
        CERR << "alGetParams failed: " << alGetErrorString(oserror());
        return;
    }

    if (rpv[0].value.i && old_rate != this->rate)
    {
        CERR << "device already in use, using other " << old_rate << " HZ sample rate" << endl;
        this->rate = old_rate;
    }
    else
    {
        rpv[0].param = AL_RATE;
        rpv[0].value.ll = alDoubleToFixed(this->rate);
        rpv[1].param = AL_MASTER_CLOCK;
        rpv[1].value.i = AL_CRYSTAL_MCLK_TYPE;

        if (alSetParams(dev, rpv, 2) < 0)
        {
            CERR << "alSetParams failed: " << alGetErrorString(oserror()) << endl;
            return;
        }
    }

    pv.param = AL_RATE;
    if (alGetParams(dev, &pv, 1) < 0)
    {
        CERR << "alGetParams failed: " << alGetErrorString(oserror()) << endl;
        return;
    }
    this->rate = alFixedToDouble(pv.value.ll);

    if (this->rate <= 0 || pv.sizeOut < 0)
    {
        CERR << "failed to get sample rate, assuming 48 kHZ" << endl;
        this->rate = 48000;
    }

    c = alNewConfig();
    if (!c)
    {
        CERR << "alNewConfig failed: " << alGetErrorString(oserror()) << endl;
        return;
    }

    bufframes = (int)this->rate;
    if (alSetQueueSize(c, bufframes * this->channels) < 0)
    {
        CERR << "alSetQueueSize failed: " << alGetErrorString(oserror()) << endl;

        if (alSetQueueSize(c, bufframes * this->channels / 2) < 0)
        {
            CERR << "alSetQueueSize failed: " << alGetErrorString(oserror()) << endl;

            if (alSetQueueSize(c, 32000) < 0)
            {
                CERR << "alSetQueueSize failed: " << alGetErrorString(oserror()) << endl;
                return;
            }
        }
    }

    if (alSetChannels(c, 2) < 0)
    {
        CERR << "alSetChannels failed: " << alGetErrorString(oserror()) << endl;
        return;
    }

    if (alSetSampFmt(c, sampfmt) < 0)
    {
        CERR << "alSetSampFmt failed: " << alGetErrorString(oserror()) << endl;
        return;
    }

    if (alSetWidth(c, wsize) < 0)
    {
        CERR << "alSetWidth failed: " << alGetErrorString(oserror()) << endl;
        return;
    }

    if (alSetDevice(c, dev) < 0)
    {
        CERR << "alSetDevice failed: " << alGetErrorString(oserror()) << endl;
        return;
    }

    p = alOpenPort("audout", "w", c);
    if (!p)
    {
        CERR << "alOpenPort failed: " << alGetErrorString(oserror());
        return;
    }

    irixPort = p;

    startThread();
}

PlayerIrixAL::~PlayerIrixAL()
{
    stopThread();

    if (irixPort > 0)
    {
        if (alClosePort(irixPort) < 0)
        {
            CERR << "alClosePort failed: " << alGetErrorString(oserror());
        }
    }
}

int
PlayerIrixAL::writeFrames(const char *frames, int numFrames) const
{
    int n = alGetFillable(irixPort);
    if (n <= 0)
        return 0;

    if (n > numFrames)
        n = numFrames;
    alWriteFrames(irixPort, (void *)frames, n);

    return n;
}

int
PlayerIrixAL::getQueued() const
{
    return alGetFilled(irixPort);
}

int
PlayerIrixAL::getWritable() const
{
    return alGetFillable(irixPort);
}

double
PlayerIrixAL::getDelay() const
{
    return (double)alGetFilled(irixPort) / (double)rate;
}
#endif
