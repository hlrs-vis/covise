/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "Listener.h"
#include "PlayerMix.h"
#include <util/unixcompat.h>
#include <errno.h>

using std::endl;
using namespace vrml;

bool PlayerMix::output_failed = false;
bool PlayerMix::threadStarted = false;

PlayerMix::PlayerMix(const Listener *listener, bool threaded, int channels, int rate, int bps)
    : Player(listener)
    , threaded(threaded)
    , channels(channels)
    , rate(rate)
    , bps(bps)
    , bytesPerFrame(channels * bps / 8)
    , headphones(false)
    , surround(false)
    , buf(0)
    , bufsiz(rate)
    , // buffer for one second
    startValid(0)
    , endValid(0)
    , numValid(0)
    , lastTime(-1.0)
    , speakers()
    , sleeptime(1000)
{
    for (int i = 0; i < channels; i++)
        speakers.push_back(new Speaker());

    if (channels == 2)
    {
        speakers[0]->setPosition(-1000.0, 1000.0, 0.0);
        speakers[1]->setPosition(1000.0, 1000.0, 0.0);
    }
    else if (channels == 4)
    {
        speakers[0]->setPosition(-1000.0, 1000.0, 0.0);
        speakers[1]->setPosition(1000.0, 1000.0, 0.0);
        speakers[2]->setPosition(-1000.0, -1000.0, 0.0);
        speakers[3]->setPosition(1000.0, -1000.0, 0.0);
    }

#ifndef HAVE_PTHREAD
    if (threaded)
    {
        CERR << "no pthread support compiled in, no separate audio thread" << endl;
        this->threaded = false;
    }
#endif

    // make sure that alignment constraints remain fulfilled
    bufsiz = (bufsiz + 15) / 16 * 16;
    buf = new char[bufsiz * bytesPerFrame];
}

PlayerMix::~PlayerMix()
{
#ifdef HAVE_PTHREAD
    if (threaded && threadStarted)
    {
        stopThread();
    }
#endif

    for (int i = 0; i < channels; i++)
        delete speakers[i];

    if (buf)
        delete[] buf;
}

void
PlayerMix::lockMutex() const
{
#ifdef HAVE_PTHREAD
    /*fprintf(stderr, "L%d", (int)pthread_self());
   if(threadStarted) fprintf(stderr, "t");
   fflush(stderr);*/
    if (threaded && threadStarted)
        pthread_mutex_lock(&mixMutex);
/*fprintf(stderr, "L-");
   fflush(stderr);*/
#endif
}

void
PlayerMix::unlockMutex() const
{
#ifdef HAVE_PTHREAD
    /*fprintf(stderr, "u%d", (int)pthread_self());
   if(threadStarted) fprintf(stderr, "t");
   fflush(stderr);*/
    if (threaded && threadStarted)
        pthread_mutex_unlock(&mixMutex);
/*fprintf(stderr, "u-");
   fflush(stderr);*/
#endif
}

void
PlayerMix::startThread()
{
#ifdef HAVE_PTHREAD
    if (threaded)
    {
        CERR << "Master Audio Thread: " << (int)pthread_self() << endl;
        pthread_mutexattr_t attr;
        pthread_mutexattr_init(&attr);
        pthread_mutexattr_settype(&attr, PTHREAD_MUTEX_RECURSIVE);
        pthread_mutex_init(&mixMutex, &attr);
        pthread_mutexattr_destroy(&attr);
        if (pthread_create(&mixThread, NULL, audioThread, this))
        {
            CERR << "failed to create audioThread: " << strerror(errno) << endl;
            ;
        }
        else
        {
            threadStarted = true;
        }
    }
#endif
}

void
PlayerMix::stopThread()
{
#ifdef HAVE_PTHREAD
    if (threaded)
    {
        pthread_cancel(mixThread);
        pthread_join(mixThread, 0);
        pthread_mutex_destroy(&mixMutex);
    }
#endif
}

#ifdef HAVE_PTHREAD
void *
PlayerMix::audioThread(void *data)
{
    PlayerMix *player = reinterpret_cast<PlayerMix *>(data);

    for (;;)
    {
        player->realUpdate(player->lastTime);
        if (output_failed)
        {
            CERR << "output failed, disabling audio" << endl;
            return NULL;
        }
    }

    return NULL;
}
#endif

void
PlayerMix::update()
{
    if (output_failed)
        return;

    double time = listener->getTime();

    if (threaded)
    {
        lockMutex();
        Player::update();
        lastTime = time;
        unlockMutex();
    }
    else
    {
        Player::update();
        realUpdate(time);
        if (output_failed)
        {
            CERR << "output failed, disabling audio" << endl;
        }
    }
}

void
PlayerMix::realUpdate(double time)
{
    float framerate = 20.0;
    int framesToProduce = rate / 10;
    if (!threaded)
    {
        if (numValid > 0)
        {
            int w = numValid;
            if (startValid + w > bufsiz)
                w = bufsiz - startValid;
            int n = writeFrames(buf + startValid * bytesPerFrame, w);
            if (n > 0)
            {
                numValid -= n;
                startValid += n;
                startValid %= bufsiz;
            }
            else
            {
                output_failed = true;
                return;
            }
        }

        if (lastTime != -1.0)
        {
            framerate = (float)(1.0 / (time - lastTime));
        }
        lastTime = time;

        if (framerate < 1.0)
            framerate = 1.0;
        framesToProduce = (int)(2.0 / framerate * rate);

        if (framesToProduce < rate / 15)
            framesToProduce = rate / 15;
    }
    //fprintf(stderr, "f=%d, q=%d, v=%d -- ", framesToProduce, getQueued(), numValid);
    if (threaded)
    {
        int queued = getQueued();
        while (queued > 0 && (queued > rate / 15 || framesToProduce - queued <= 0))
        {
            if (queued > rate / 15 || framesToProduce - queued <= 0)
            {
                unsigned sleeptime = (unsigned)(0.3 * (float)queued / (float)rate * 1e6);
                usleep(sleeptime);
            }
            queued = getQueued();
        }
    }

    //fprintf(stderr, "f=%d, q=%d, v=%d -- ", framesToProduce, getQueued(), numValid);
    if (framesToProduce > bufsiz)
        CERR << "buf too small, expect buffer underruns" << endl;

    if (numValid + framesToProduce > bufsiz)
    {
        framesToProduce = bufsiz - numValid;
    }

    if (framesToProduce > rate / 5)
        framesToProduce = rate / 5;

    int packetsize = getPacketSize();
    if (packetsize > 0)
    {
        framesToProduce = (framesToProduce + packetsize - 1) / packetsize * packetsize;
    }

    //CERR << "framesToProduce=" << framesToProduce << endl;

    if (framesToProduce > 0)
    {
        int f = framesToProduce;
        if (endValid + framesToProduce > bufsiz)
        {
            f = bufsiz - endValid;
        }
        memset(buf + endValid * bytesPerFrame, '\0', f * bytesPerFrame);
        if (f < framesToProduce)
        {
            memset(buf, '\0', (framesToProduce - f) * bytesPerFrame);
        }
        lockMutex();
        for (unsigned i = 0; i < sources.size(); i++)
        {
            if (sources[i])
            {
                sources[i]->update(this, buf + endValid * bytesPerFrame, f);
            }
        }
        if (f < framesToProduce)
        {
            for (unsigned i = 0; i < sources.size(); i++)
            {
                if (sources[i])
                {
                    sources[i]->update(this, buf, framesToProduce - f);
                }
            }
        }
        unlockMutex();
        numValid += framesToProduce;
        endValid += framesToProduce;
        endValid %= bufsiz;
    }

    //fprintf(stderr, "f=%d, q=%d, v=%d\n", framesToProduce, getQueued(), numValid);
    int w = numValid;
    if (w > bufsiz - startValid)
        w = bufsiz - startValid;
    int n = writeFrames(buf + startValid * bytesPerFrame, w);
    if (n > 0)
    {
        startValid += n;
        startValid %= bufsiz;
        numValid -= n;
    }
    else
    {
        output_failed = true;
        return;
    }
    if (n == w && w < numValid)
    {
        n = writeFrames(buf, numValid - w);
        if (n > 0)
        {
            startValid += n;
            startValid %= bufsiz;
            numValid -= n;
        }
        else
        {
            output_failed = true;
            return;
        }
    }
}

Player::Source *
PlayerMix::newSource(const Audio *audio)
{
    lockMutex();
    Source *src = new Source(audio);
    int handle = addSource(src);
    if (-1 == handle)
    {
        delete src;
        src = 0;
    }
    unlockMutex();
    return src;
}

void
PlayerMix::setSeparation(float sep)
{
    separation = sep;
}

PlayerMix::Source::Source(const Audio *audio)
    : Player::Source(audio)
    , pos(0.0)
    , off()
{
}

PlayerMix::Source::~Source()
{
}

void
PlayerMix::Source::play(double start)
{
    //CERR << "PlayerMix::Source::play(double)" << endl;
    // dynamic_cast causes problems on gcc2 systems
    // and why the hell do you use dynamic_cast if you don't verify the result?
    //pos = (start + dynamic_cast<PlayerMix *>(player)->getDelay()) * audio->samplesPerSec();
    pos = (start + ((PlayerMix *)(player))->getDelay()) * audio->samplesPerSec();

    Player::Source::play(start);
}

void
PlayerMix::Source::play()
{
    Player::Source::play();
}

int
PlayerMix::Source::update(const Player *genericPlayer, char *buf, int numFrames)
{
    Player::Source::update(genericPlayer, buf, numFrames);

    if (!playing)
    {
        return 0;
    }

    // dynamic_cast causes problems on gcc2 systems
    //const PlayerMix *player = dynamic_cast<const PlayerMix *>(genericPlayer);
    const PlayerMix *player = (const PlayerMix *)(genericPlayer);
    if (!player)
    {
        CERR << "no player" << endl;
        return -1;
    }

    unsigned chan = player->channels;
    if (chan != off.size())
    {
        off.resize(chan);
        for (unsigned c = 0; c < chan; c++)
            off[c] = 0.0;
    }

    int rate = player->rate;
    int bps = player->bps;
    if (16 != bps)
    {
        CERR << "bps != 16" << endl;
        return -1;
    }

    int16_t *samp16 = (int16_t *)audio->samples();
    int8_t *samp8 = (int8_t *)audio->samples();
    int16_t *buf16 = (int16_t *)buf;
    //int8_t *buf8 = (int8_t *)buf;

    vec l = player->getListenerPositionWC();
    vec rel_source = x.sub(l);

    std::vector<float> cangle(chan), dist(chan), intens(chan), newoff(chan);
    std::vector<vec> rel_speaker(chan);
    for (unsigned c = 0; c < chan; c++)
    {
        cangle[c] = 0.0;
        dist[c] = 0.0;
        intens[c] = 0.0;
        newoff[c] = (float)off[c];
        Speaker *s = player->speakers[c];
        rel_speaker[c] = s->x.sub(l);
    }

    float globalPitch = 1.0;
    if (rel_source.length() > 0.0)
    {
        // calculate Doppler effect
        float vl = player->getListenerVelocity().dot(rel_source.normalize());
        float vs = v.dot(rel_source.normalize());

        // include Doppler effect in pitch
        globalPitch = pitch * (player->speedOfSound - vl) / (player->speedOfSound + vs);
    }
    else
    {
        CERR << "r <= 0.0 !!! " << endl;
    }

    for (unsigned c = 0; c < chan; c++)
    {
        dist[c] = rel_speaker[c].length();
        newoff[c] = -dist[c] / player->speedOfSound * audio->samplesPerSec();
    }

    if (1e-6 >= intensity)
    {
        for (unsigned c = 0; c < chan; c++)
            off[c] = newoff[c];
        pos += audio->samplesPerSec() * globalPitch / rate * numFrames;
        pos = fmod(pos, audio->numSamples());
        return 0;
    }

    // parameters for Dolby Surround mixing
    float angle = 0.0;
    float fh = 0.0, fl = 0.0, fr = 0.0, fv = 0.0;
    if (spatialize && chan > 1)
    {
        if (2 == chan && player->surround)
        {
            angle = asin(rel_source.normalize().x);
            if (rel_source.normalize().y < 0.0)
            {
                angle = (float)M_PI - angle;
            }

            if (angle < 0.0)
            {
                fl = (float)(-angle / M_PI * 2.0);
                fv = 1.0f - fl;
            }
            else if (angle < 0.5f * M_PI)
            {
                fr = (float)(angle / M_PI * 2.0);
                fv = 1.0f - fr;
            }
            else if (angle < M_PI)
            {
                fh = (float)(angle / M_PI * 2.0 - 1.0);
                fr = 1.0f - fh;
            }
            else if (angle < 1.5f * M_PI)
            {
                fl = (float)(2.0f * angle / M_PI - 2.0f);
                fh = 1.0f - fl;
            }
            else
            {
                fv = (float)(2.0f * angle / M_PI - 3.0f);
                fl = 1.0f - fv;
            }
        }
        else if (2 == chan && player->headphones)
        {
            //vec orient;
            vec up, at;
            player->getListenerOrientation(&at, &up);
            vec pos = player->getListenerPositionWC();
            vec diff = up.cross(at).normalize().mult(player->separation / 2.0f);
            vec left_ear = pos.add(diff);
            vec right_ear = pos.sub(diff);

            vec rel_l = x.sub(left_ear).normalize();
            vec rel_r = x.sub(right_ear).normalize();
            diff = diff.normalize();

            float vol_l = (-diff.dot(rel_l) + 3.0f) / 4.0f;
            float vol_r = (+diff.dot(rel_r) + 3.0f) / 4.0f;

            float distl = left_ear.sub(x).length();
            float distr = right_ear.sub(x).length();

            intens[0] = vol_l * intensity;
            intens[1] = vol_r * intensity;

            float timediff = (distl - distr) / player->speedOfSound;

            newoff[0] = -timediff / 2.0f * audio->samplesPerSec();
            newoff[1] = +timediff / 2.0f * audio->samplesPerSec();
        }
        else if (2 == chan)
        {
            vec spk = player->speakers[1]->x.sub(player->speakers[0]->x);
            vec src = x.sub(player->speakers[0]->x);
            float c = src.normalize().dot(spk.normalize());
            if (c < 0.0)
            {
                intens[0] = 1.0f * intensity;
                intens[1] = 0.0;
            }
            else if (c > 1.0)
            {
                intens[0] = 1.0f * intensity;
                intens[1] = 0.0;
            }
            else
            {
                intens[0] = (1.0f - c) * intensity;
                intens[1] = c * intensity;
            }
        }
        else
        {
            // find the 2 speakers nearest to source direction
            // (we assume they are arranged in one plane)
            int index_first = -1, index_second = -1;
            float cangle_first = -1.0, cangle_second = -1.0;
            for (unsigned c = 0; c < chan; c++)
            {
                if (!player->speakers[c]->spatialize)
                {
                    // e.g. a subwoofer
                    intens[c] = intensity;
                    continue;
                }
                cangle[c] = rel_speaker[c].normalize().dot(rel_source.normalize());
                if (cangle[c] >= cangle_second)
                {
                    if (cangle[c] >= cangle_first)
                    {
                        cangle_second = cangle_first;
                        index_second = index_first;
                        cangle_first = cangle[c];
                        index_first = c;
                    }
                    else
                    {
                        index_second = c;
                        cangle_second = cangle[c];
                    }
                }
            }
            if (index_first < 0 || index_second < 0)
            {
                CERR << "FIRST: " << index_first << ", SECOND: " << index_second << endl;
            }

            vec n1 = rel_speaker[index_first].cross(rel_speaker[index_second]);
            // normal on plane perpendicular to the speakers-listener plane containing the source
            vec n = rel_source.cross(n1);
            //float nom = rel_speaker[index_first].dot(n);
            float denom = rel_speaker[index_first].sub(rel_speaker[index_second]).dot(n);
            float u = rel_speaker[index_first].dot(n) / rel_speaker[index_first].sub(rel_speaker[index_second]).dot(n);
            if (denom == 0.0)
            {
                u = 0.5;
                CERR << "denom==0.0 !!! " << endl;
            }

            if (u < 0.0 || u > 1.0)
            {
                // the source is outside the convex hull of the speakers and the listener
                intens[index_first] = intensity;
                intens[index_second] = 0.0;
            }
            else
            {
                intens[index_first] = (1.0f - u) * intensity;
                intens[index_second] = u * intensity;
            }
        }
    }
    else
    {
        /* non-spatialized audio */
        for (unsigned c = 0; c < chan; c++)
        {
            intens[c] = intensity;
        }
    }

    // volume correction according to distance of speakers
    for (unsigned c = 0; c < chan; c++)
    {
        //intens[c] *= dist[c]*dist[c]/1e6;
        //intens[c] /= player->_numSources;
    }

#define DEFW                                   \
    int s1 = (int)floor(s), s2 = (int)ceil(s); \
    float w = (float)(s - s1)
#define INCS                                  \
    s += d;                                   \
    if (s >= audio->numSamples())             \
    {                                         \
        if (loop)                             \
            s = fmod(s, audio->numSamples()); \
        else                                  \
            break;                            \
    }
    if (spatialize && 2 == chan && player->surround)
    {
        double d = audio->samplesPerSec() / rate * globalPitch;
        double s = pos;
        s = fmod(s, audio->numSamples());
        if (s < 0.0)
            s += audio->numSamples();
        for (int i = 0; i < numFrames; i++)
        {
            DEFW;
            float lr = 0.0;
            if (audio->bitsPerSample() == 16)
            {
                if (audio->channels() == 1)
                    lr = intensity * ((1.0f - w) * samp16[s1] + w * samp16[s2]);
                else
                    lr = intensity * 0.5f * ((1.0f - w) * (samp16[s1 * 2] + (float)samp16[s1 * 2 + 1])
                                             + w * (samp16[s2 * 2] + (float)samp16[s2 * 2 + 1]));
            }
            else if (audio->bitsPerSample() == 8)
            {
                if (audio->channels() == 1)
                    lr = 256.0f * intensity * ((1.0f - w) * samp8[s1] + w * samp8[s2]);
                else
                    lr = 256.0f * intensity * 0.5f * ((1.0f - w) * (samp8[s1 * 2] + (float)samp8[s1 * 2 + 1])
                                                      + w * (samp8[s2 * 2] + (float)samp8[s2 * 2 + 1]));
            }
            if (fh > 0.0)
            {
                buf16[i * 2] += (int16_t)(lr * fh);
                buf16[i * 2 + 1] += (int16_t)(-lr * fh);
            }
            if (fr > 0.0)
            {
                buf16[i * 2 + 1] += (int16_t)(lr * fr);
            }
            if (fl > 0.0)
            {
                buf16[i * 2] += (int16_t)(lr * fl);
            }
            if (fv > 0.0)
            {
                buf16[i * 2] += (int16_t)(lr * fv);
                buf16[i * 2 + 1] += (int16_t)(lr * fv);
            }
            INCS;
        }
        pos = s;
    }
    else
    {
        for (unsigned c = 0; c < chan; c++)
        {
            if (0.0 == intens[c])
            {
                off[c] = newoff[c];
                continue;
            }
            double d = (audio->samplesPerSec() * numFrames / (float)rate + newoff[c] - off[c]) / numFrames * globalPitch;
            double s = pos + off[c];
            s = fmod(s, audio->numSamples());
            if (s < 0.0)
                s += audio->numSamples();
            if (audio->channels() == 1 && audio->bitsPerSample() == 16)
            {
                for (int i = 0; i < numFrames; i++)
                {
                    DEFW;
                    buf16[i * chan + c] += (int16_t)(intens[c] * ((1.0 - w) * samp16[s1] + w * samp16[s2]));
                    INCS;
                }
            }
            else if (audio->channels() == 1 && audio->bitsPerSample() == 8)
            {
                for (int i = 0; i < numFrames; i++)
                {
                    DEFW;
                    buf16[i * chan + c] += (int16_t)(256 * intens[c] * ((1.0 - w) * samp8[s1] + w * samp8[s2]));
                    INCS;
                }
            }
            else if (audio->channels() == 2 && audio->bitsPerSample() == 16)
            {
                for (int i = 0; i < numFrames; i++)
                {
                    DEFW;
                    buf16[i * chan + c] += (int16_t)(intens[c] * 0.5 * ((1.0 - w) * (samp16[s1 * 2] + samp16[s1 * 2 + 1])
                                                                        + w * (samp16[s2 * 2] + samp16[s2 * 2 + 1])));
                    INCS;
                }
            }
            else if (audio->channels() == 2 && audio->bitsPerSample() == 8)
            {
                for (int i = 0; i < numFrames; i++)
                {
                    DEFW;
                    buf16[i * chan + c] += (int16_t)(256 * intens[c] * 0.5 * ((1.0 - w) * (samp8[s1 * 2] + samp16[s1 * 2 + 1])
                                                                              + w * (samp8[s2 * 2] + samp16[s2 * 2 + 1])));
                    INCS;
                }
            }
            off[c] = newoff[c];
        }
        pos += audio->samplesPerSec() * globalPitch / rate * numFrames;
        pos = fmod(pos, audio->numSamples());
    }

    return numFrames;
}

void
PlayerMix::setSpeakerPosition(int speaker, float x, float y, float z)
{
    if (0 <= speaker && (unsigned)speaker < speakers.size())
        speakers[speaker]->setPosition(x, y, z);
}

void
PlayerMix::setSpeakerSpatialize(int speaker, bool spatialize)
{
    if (0 <= speaker && (unsigned)speaker < speakers.size())
        speakers[speaker]->setSpatialize(spatialize);
}

void
PlayerMix::setHeadphones(bool headphones)
{
    this->headphones = headphones;
}

void
PlayerMix::setDolbySurround(bool surround)
{
    this->surround = surround;
}

void
PlayerMix::Speaker::setPosition(float x, float y, float z)
{
    this->x.x = x;
    this->x.y = y;
    this->x.z = z;
}

void
PlayerMix::Speaker::setSpatialize(bool spatialize)
{
    this->spatialize = spatialize;
}

PlayerMix::Speaker::Speaker()
    : x(0.0, 0.0, 0.0)
    , spatialize(true)
{
}

PlayerMix::Speaker::~Speaker()
{
}

#define PROTECT_1INT(func)          \
    void PlayerMix::func(int param) \
    {                               \
        lockMutex();                \
        Player::func(param);        \
        unlockMutex();              \
    }

PROTECT_1INT(removeSource)

#if 0
void
PlayerMix::setSourceAudio(int handle, const Audio *audio) const
{
   lockMutex();
   Player::setSourceAudio(handle, audio);
   unlockMutex();
}
#endif
