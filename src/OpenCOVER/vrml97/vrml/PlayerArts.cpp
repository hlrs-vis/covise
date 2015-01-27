/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "Listener.h"
#include "PlayerArts.h"

#ifdef HAVE_ARTS
#include <artsc/artsc.h>

using std::endl;
using namespace vrml;

PlayerArts::PlayerArts(const Listener *listener, bool threaded, int channels, int rate, int bps)
    : PlayerMix(listener, threaded, 2, rate, bps)
{
    CERR << "channels=" << this->channels << ", bps=" << bps << endl;

    int err = arts_init();
    if (err)
    {
        CERR << "failed to connect to Arts server: " << arts_error_text(err) << endl;
        return;
    }

    arts_stream = arts_play_stream(rate, bps, channels, "COVER");
    if (arts_stream == NULL)
    {
        CERR << "failed to play stream" << endl;
        output_failed = true;
        return;
    }
    if (!threaded)
        arts_stream_set(arts_stream, ARTS_P_BLOCKING, 0);

    startThread();
}

PlayerArts::~PlayerArts()
{
    stopThread();

    arts_close_stream(arts_stream);
    arts_free();
}

int
PlayerArts::writeFrames(const char *frames, int numFrames) const
{
    int n = arts_write(arts_stream, frames, numFrames * bytesPerFrame);
    //fprintf(stderr, "arts:writeFrames: numFrames=%d, n=%d\n", numFrames, n);
    if (n >= 0)
        return n / bytesPerFrame;
    else
        return n;
}

int
PlayerArts::getQueued() const
{
    int bufsiz = arts_stream_get(arts_stream, ARTS_P_BUFFER_SIZE);
    int avail = arts_stream_get(arts_stream, ARTS_P_BUFFER_SPACE);

    //fprintf(stderr, "arts:getQueued: bufsiz=%d, avail=%d\n", bufsiz, avail);
    return -1;
    return (bufsiz - avail) / bytesPerFrame;
}

int
PlayerArts::getWritable() const
{
    int space = arts_stream_get(arts_stream, ARTS_P_BUFFER_SPACE);

    //fprintf(stderr, "arts:getWritable: space=%d\n", space);
    return space / bytesPerFrame;
}

double
PlayerArts::getDelay() const
{
    //int latency = arts_stream_get(arts_stream, ARTS_P_TOTAL_LATENCY);
    //fprintf(stderr, "arts:getDelay: latency=%d\n", latency);

    return (double)(arts_stream_get(arts_stream, ARTS_P_BUFFER_SIZE) - arts_stream_get(arts_stream, ARTS_P_BUFFER_SPACE)) / (double)(bytesPerFrame * rate);
}

int
PlayerArts::getPacketSize() const
{
    int size = arts_stream_get(arts_stream, ARTS_P_PACKET_SIZE);

    //fprintf(stderr, "arts:getPacketSize: size=%d\n", size);
    return size / bytesPerFrame;
}
#endif
