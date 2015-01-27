/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "Listener.h"
#include "PlayerOSS.h"

#ifdef HAVE_OSS
#include <sys/soundcard.h>
#include <sys/ioctl.h>
#include <unistd.h>
#include <sys/fcntl.h>

using std::endl;
using namespace vrml;

PlayerOSS::PlayerOSS(const Listener *listener, bool threaded, int channels, int rate, int bps, std::string device)
    : PlayerMix(listener, threaded, channels, rate, bps)
    , dsp_fd(-1)
{
    if (device.empty())
        device = "/dev/dsp";

    dsp_fd = open(device.c_str(), O_WRONLY, O_NONBLOCK);
    if (-1 == dsp_fd)
    {
        CERR << "failed to open " << device << endl;
        return;
    }

    if (fcntl(dsp_fd, F_SETFL, 0) < 0)
    {
        CERR << "failed to make dsp filedes blocking" << endl;
    }

    int srate = rate;
    if (ioctl(dsp_fd, SNDCTL_DSP_SPEED, &srate) == -1
        || srate != rate)
    {
        CERR << "failed to set sample rate" << endl;
        close(dsp_fd);
        dsp_fd = -1;
        return;
    }

    // sample format
    int format = AFMT_S16_LE;
    if (ioctl(dsp_fd, SNDCTL_DSP_SETFMT, &format) == -1
        || format != AFMT_S16_LE)
    {
        CERR << "failed to set sample format" << endl;
        close(dsp_fd);
        dsp_fd = -1;
        return;
    }

    // channels
    CERR << "channels=" << channels << endl;
    if (channels > 2)
    {
        int chan = channels;
        if (ioctl(dsp_fd, SNDCTL_DSP_CHANNELS, &chan) == -1)
        {
            CERR << "failed to set number of channels" << endl;
            close(dsp_fd);
            dsp_fd = -1;
            return;
        }
        if (chan != channels)
        {
            CERR << "failed to set number of channels" << endl;
            close(dsp_fd);
            dsp_fd = -1;
            return;
        }
    }
    else
    {
        int stereo_flag = channels == 2 ? 1 : 0;
        if (ioctl(dsp_fd, SNDCTL_DSP_STEREO, &stereo_flag) == -1)
        {
            CERR << "failed to set mono/stereo mode" << endl;
            close(dsp_fd);
            dsp_fd = -1;
            return;
        }
        if (stereo_flag != channels - 1)
        {
            CERR << "failed to set mono/stereo mode" << endl;
            close(dsp_fd);
            dsp_fd = -1;
            return;
        }
    }

    startThread();
    return;
}

PlayerOSS::~PlayerOSS()
{
    stopThread();

    if (-1 == dsp_fd)
        return;

    ioctl(dsp_fd, SNDCTL_DSP_RESET, NULL);
    close(dsp_fd);
    dsp_fd = -1;
}

int
PlayerOSS::writeFrames(const char *frames, int numFrames) const
{
    if (-1 == dsp_fd)
        return -1;

    int n = write(dsp_fd, frames, numFrames * bytesPerFrame);
    if (n >= 0)
        return n / bytesPerFrame;

    return n;
}

int
PlayerOSS::getQueued() const
{
    int delay = 0;
    if (ioctl(dsp_fd, SNDCTL_DSP_GETODELAY, &delay) != -1)
        return delay / bytesPerFrame;

    return 0;
}

int
PlayerOSS::getWritable() const
{
    return 0;
}

double
PlayerOSS::getDelay() const
{
    return (double)getQueued() / (double)rate;
}
#endif
