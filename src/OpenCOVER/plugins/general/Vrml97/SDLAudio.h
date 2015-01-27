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
#include <osg/ImageStream>
#include <osg/observer_ptr>

class SDLAudioSink : public osg::AudioSink
{
public:
    SDLAudioSink(osg::AudioStream *audioStream)
        : _started(false)
        , _paused(false)
        , _audioStream(audioStream)
    {
    }

    ~SDLAudioSink();

    virtual void play();
    virtual void pause();
    virtual void stop();

    virtual bool playing() const
    {
        return _started && !_paused;
    }

    bool _started;
    bool _paused;
    osg::observer_ptr<osg::AudioStream> _audioStream;
};
