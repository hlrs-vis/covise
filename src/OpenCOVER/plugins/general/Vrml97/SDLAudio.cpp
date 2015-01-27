/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

//
//  SDLAudio for ffmpeg movies
//  from osgmovie example
//
//  %W% %G%
//  SDLAudio.h
//  Class for display of audio in VRML models using OpenSceneGraph.
//  OpenSceneGraph has to be compiled with SDL (e.g. by building the examples)
//

#include "SDLAudio.h"
#include <SDL.h>

static void soundReadCallback(void *user_data, uint8_t *data, int datalen)
{
    SDLAudioSink *sink = reinterpret_cast<SDLAudioSink *>(user_data);
    osg::ref_ptr<osg::AudioStream> as = sink->_audioStream.get();
    if (as.valid())
    {
        as->consumeAudioBuffer(data, datalen);
    }
}

SDLAudioSink::~SDLAudioSink()
{
    stop();
}

void SDLAudioSink::play()
{
    if (_started)
    {
        if (_paused)
        {
            SDL_PauseAudio(0);
            _paused = false;
        }
        return;
    }

    _started = true;
    _paused = false;

    fprintf(stderr, "  audioFrequency()=%d", _audioStream->audioFrequency());
    fprintf(stderr, "  audioNbChannels()=%d", _audioStream->audioNbChannels());
    fprintf(stderr, "  audioSampleFormat()=%d", _audioStream->audioSampleFormat());

    SDL_AudioSpec specs = { 0, 0, 0, 0, 0, 0, 0, 0, 0 };
    SDL_AudioSpec wanted_specs = { 0, 0, 0, 0, 0, 0, 0, 0, 0 };

    wanted_specs.freq = _audioStream->audioFrequency();
    wanted_specs.format = AUDIO_S16SYS;
    wanted_specs.channels = _audioStream->audioNbChannels();
    wanted_specs.silence = 0;
    wanted_specs.samples = 1024;
    wanted_specs.callback = soundReadCallback;
    wanted_specs.userdata = this;

    if (SDL_OpenAudio(&wanted_specs, &specs) < 0)
        throw "SDL_OpenAudio() failed (" + std::string(SDL_GetError()) + ")";

    SDL_PauseAudio(0);
}

void SDLAudioSink::pause()
{
    if (_started)
    {
        SDL_PauseAudio(1);
        _paused = true;
    }
}

void SDLAudioSink::stop()
{
    if (_started)
    {
        if (!_paused)
            SDL_PauseAudio(1);
        SDL_CloseAudio();

        fprintf(stderr, "~SDLAudioSink() destructor, but still playing");
    }
}
