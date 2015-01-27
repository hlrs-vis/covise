/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "Listener.h"
#include "PlayerAlsa.h"

#ifdef HAVE_ALSA
#include <alsa/asoundlib.h>

using std::endl;
using namespace vrml;

PlayerAlsa::PlayerAlsa(const Listener *listener, bool threaded, int channels, int rate, int bps, std::string device)
    : PlayerMix(listener, threaded, channels, rate, bps)
{
    if (device.empty())
    {
        switch (this->channels)
        {
        case 1:
        case 2:
            device = "plughw:0,0";
            break;
        case 4:
            device = "surround40";
            break;
        case 6:
            device = "surround51";
            break;
        default:
            CERR << "unsupported number of channels" << endl;
            output_failed = true;
            return;
        }
    }

    if (snd_pcm_status_malloc(&alsaStatus) < 0)
    {
        CERR << "failed to allocate status structure" << endl;
        output_failed = true;
        return;
    }

    //if (snd_pcm_open(&alsaHandle, dev, SND_PCM_STREAM_PLAYBACK, SND_PCM_NONBLOCK) < 0) {
    if (snd_pcm_open(&alsaHandle, device.c_str(), SND_PCM_STREAM_PLAYBACK, 0) < 0)
    {
        CERR << "failed to open " << device << endl;
        output_failed = true;
        return;
    }

    snd_pcm_hw_params_t *hw_params;
    snd_pcm_hw_params_malloc(&hw_params);
    if (snd_pcm_hw_params_any(alsaHandle, hw_params) < 0)
    {
        CERR << "failed to configure " << device << endl;
        output_failed = true;
        return;
    }

    if (snd_pcm_hw_params_set_access(alsaHandle, hw_params, SND_PCM_ACCESS_RW_INTERLEAVED) < 0)
    {
        CERR << "failed to set interleaved access" << endl;
        output_failed = true;
        return;
    }

    if (snd_pcm_hw_params_set_format(alsaHandle, hw_params, SND_PCM_FORMAT_S16_LE) < 0)
    {
        CERR << "failed to set format" << endl;
        output_failed = true;
        return;
    }

    int dir;
#if SND_LIB_MAJOR < 1
    int exact_rate = snd_pcm_hw_params_set_rate_near(alsaHandle, hw_params, this->rate, &dir);
#else
    unsigned r = this->rate;
    int exact_rate = snd_pcm_hw_params_set_rate_near(alsaHandle, hw_params, &r, &dir);
#endif
    if (0 != dir)
    {
        CERR << "using rate " << exact_rate << " HZ instead of " << this->rate << " HZ" << endl;

        if (exact_rate > 0)
        {
            this->rate = exact_rate;
        }
        else
        {
            output_failed = true;
        }
    }

    if (snd_pcm_hw_params_set_channels(alsaHandle, hw_params, this->channels) < 0)
    {
        CERR << "failed to set " << this->channels << " channels" << endl;
        output_failed = true;
        return;
    }

    if (snd_pcm_hw_params_set_periods(alsaHandle, hw_params, 4, 0) < 0)
    {
        CERR << "failed to set periods" << endl;
        output_failed = true;
        return;
    }

    if (snd_pcm_hw_params_set_buffer_size(alsaHandle, hw_params, 32768) < 0)
    {
        CERR << "failed to set buffer size" << endl;
        output_failed = true;
        return;
    }

    int err = snd_pcm_hw_params(alsaHandle, hw_params);
    if (err < 0)
    {
        CERR << "failed to apply hwparams: " << snd_strerror(err) << endl;
        output_failed = true;
        return;
    }
    snd_pcm_hw_params_free(hw_params);

    snd_pcm_sw_params_t *sw_params;
    if (snd_pcm_sw_params_malloc(&sw_params) < 0)
    {
        CERR << "failed to allocate sw_param structure" << endl;
        output_failed = true;
        return;
    }

    if (snd_pcm_sw_params_current(alsaHandle, sw_params) < 0)
    {
        CERR << "failed to get current sw_params" << endl;
        output_failed = true;
        return;
    }

    if (snd_pcm_sw_params_set_avail_min(alsaHandle, sw_params, 4096) < 0)
    {
        CERR << "failed to set min avail" << endl;
        output_failed = true;
        return;
    }

    if (snd_pcm_sw_params(alsaHandle, sw_params) < 0)
    {
        CERR << "failed to set sw_params" << endl;
        output_failed = true;
        return;
    }
    snd_pcm_sw_params_free(sw_params);

    startThread();
}

PlayerAlsa::~PlayerAlsa()
{
    stopThread();

    snd_pcm_drop(alsaHandle);
    snd_pcm_status_free(alsaStatus);
}

int
PlayerAlsa::writeFrames(const char *frames, int numFrames) const
{
    int n = snd_pcm_writei(alsaHandle, frames, numFrames);
    if (n < 0)
    {
        if (-EPIPE == n)
        {
            CERR << "Buffer underrun" << endl;
            if (snd_pcm_prepare(alsaHandle) >= 0)
            {
                // try it again
                n = snd_pcm_writei(alsaHandle, frames, numFrames);
                if (n >= 0)
                    return n;
                if (-EAGAIN == n)
                    return 0;
                return -1;
            }
            else
            {
                CERR << "snd_pcm_prepare failed" << endl;
                return -1;
            }
        }
        else if (-ESTRPIPE == n)
        {
            CERR << "Suspend" << endl;
            while (snd_pcm_resume(alsaHandle) == -EAGAIN)
                usleep(10000);
            if (snd_pcm_prepare(alsaHandle) >= 0)
                return 0;
        }
        else if (-EAGAIN == n)
        {
            return 0;
        }
        CERR << "FATAL Alsa ERROR" << endl;
        return -1;
    }
    return n;
}

int
PlayerAlsa::getQueued() const
{
    int err = snd_pcm_status(alsaHandle, alsaStatus);
    if (err < 0)
    {
        CERR << "Stream status error: " << snd_strerror(err) << endl;
        return -1;
    }
    int queued = snd_pcm_status_get_avail_max(alsaStatus) - snd_pcm_status_get_avail(alsaStatus);
    if (queued < 0)
        return 0;

    return queued;
}

int
PlayerAlsa::getWritable() const
{
    int err = snd_pcm_status(alsaHandle, alsaStatus);
    int ret = 0;
    if (err < 0)
    {
        CERR << "Stream status error: " << snd_strerror(err) << endl;
        ret = -1;
    }
    else
    {
        ret = snd_pcm_status_get_avail(alsaStatus);
    }

    CERR << "ret=" << ret << endl;
    return ret;
}

double
PlayerAlsa::getDelay() const
{
    int err = snd_pcm_status(alsaHandle, alsaStatus);
    if (err < 0)
    {
        CERR << "Stream status error: " << snd_strerror(err) << endl;
        return -1;
    }
    int delay = snd_pcm_status_get_delay(alsaStatus);

    return (double)delay / (double)rate;
}
#endif
