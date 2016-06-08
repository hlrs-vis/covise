/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */


#ifdef _WIN32
#if (_MSC_VER >= 1300) && !(defined(MIDL_PASS) || defined(RC_INVOKED))
#define POINTER_64 __ptr64
#else
#define POINTER_64
#endif
#include <winsock2.h>
#include <windows.h>
#endif

#include <cover/coVRPluginSupport.h>
#include <vrml97/vrml/VrmlNodeSound.h>
#include <vrml97/vrml/Player.h>
#include <vrml97/vrml/PlayerMix.h>
#include <vrml97/vrml/PlayerFactory.h>

#include "ViewerOsg.h"
#include "ListenerCover.h"
#include "Vrml97Plugin.h"

#include <config/CoviseConfig.h>

using covise::coCoviseConfig;

ListenerCover::ListenerCover(coVRPluginSupport *cover)
    : cover(cover)
    , lastTime(0.0)
    , lastPos(0.0, 0.0, 0.0)
    , velocity(0.0, 0.0, 0.0)
{
    if (!this->cover)
        CERR << "cover == NULL -- this won't work !!!" << endl;
    matIdent = new osg::Matrix;
    matIdent->makeIdentity();
}

ListenerCover::~ListenerCover()
{
    delete matIdent;
}

osg::Matrix
ListenerCover::getVrmlBaseMat() const
{
    if (Vrml97Plugin::plugin && Vrml97Plugin::plugin->viewer)
        return Vrml97Plugin::plugin->viewer->vrmlBaseMat;
    else
        return *matIdent;
}

osg::Matrix
ListenerCover::getCurrentTransform() const
{
    if (Vrml97Plugin::plugin && Vrml97Plugin::plugin->viewer)
        return Vrml97Plugin::plugin->viewer->currentTransform;
    else
        return *matIdent;
}

vec
ListenerCover::getPositionWC() const
{
    osg::Matrix m = cover->getViewerMat();
    //fprintf(stderr, "listener: pos=(%f %f %f)\n", m(3,0), m(3,1), m(3,2));
    return vec(m(3, 0), m(3, 1), m(3, 2));
}

vec
ListenerCover::getPositionVC() const
{
    //vec w = getPositionWC();
    //vec v = WCtoVC(getPositionWC());
    //fprintf(stderr, "wc=(%f %f %f), getPosVC=(%f %f %f)\n", w.x, w.y, w.z, v.x, v.y, v.z);
    return WCtoVC(getPositionWC());
}

vec
ListenerCover::getPositionOC() const
{
    //fprintf(stderr, "getPosOC\n");
    return WCtoOC(getPositionWC());
}

vec
ListenerCover::getVelocity() const
{
    double now = cover->frameTime();
    if (lastTime < now)
    {
        if (0.0 == lastTime)
            velocity = vec(0.0, 0.0, 0.0);
        else
            velocity = getPositionVC().sub(lastPos).div(now - lastTime);
        lastPos = getPositionVC();
        lastTime = now;
    }

    return velocity;
}

void
ListenerCover::getOrientation(vec *at, vec *up) const
{
    osg::Matrix m = osg::Matrix::inverse(cover->getViewerMat());
    osg::Vec3 v1(0.0, 1.0, 0.);

    //v2.xformVec(v1, m);
    osg::Vec3 v2 = osg::Matrix::transform3x3(m, v1);
    v2.normalize();
    *at = vec(v2[0], v2[1], v2[2]);

    v1.set(0.0, 0.0, 1.0);
    //v2.xformVec(v1, m);
    v2 = osg::Matrix::transform3x3(m, v1);
    v2.normalize();
    *up = vec(v2[0], v2[1], v2[2]);
}

// from object to world coordinates
vec
ListenerCover::OCtoWC(vec p) const
{
    osg::Vec4 pos(p.x, p.y, p.z, 1.0);
    osg::Matrix tr = getCurrentTransform();
    tr.postMult(getVrmlBaseMat());
    osg::Vec4 v = pos * tr;

    return vec(v[0] / v[3], v[1] / v[3], v[2] / v[3]);
}

// from object to VRML coordinates
vec
ListenerCover::OCtoVC(vec p) const
{
    osg::Vec4 pos(p.x, p.y, p.z, 1.0);
    osg::Matrix tr = getCurrentTransform();
    osg::Vec4 v = pos * tr;

    return vec(v[0] / v[3], v[1] / v[3], v[2] / v[3]);
}

// from world to VRML coordinates
vec
ListenerCover::WCtoVC(vec p) const
{
    osg::Matrix tr = getVrmlBaseMat();
    osg::Matrix inv = osg::Matrix::inverse(tr);
    osg::Vec4 pos(p.x, p.y, p.z, 1.0);
    osg::Vec4 v = pos * inv;

    return vec(v[0] / v[3], v[1] / v[3], v[2] / v[3]);
}

// from world to object coordinates
vec
ListenerCover::WCtoOC(vec p) const
{
    osg::Vec4 pos(p.x, p.y, p.z, 1.0);
    osg::Matrix tr = getCurrentTransform();
    tr.postMult(getVrmlBaseMat());
    osg::Matrix inv = osg::Matrix::inverse(tr);
    osg::Vec4 v = pos * inv;

    return vec(v[0] / v[3], v[1] / v[3], v[2] / v[3]);
}

// from VRML to object coordinates
vec
ListenerCover::VCtoOC(vec p) const
{
    osg::Vec4 pos(p.x, p.y, p.z, 1.0);
    osg::Matrix tr = getCurrentTransform();
    osg::Matrix inv = osg::Matrix::inverse(tr);
    osg::Vec4 v = pos * inv;

    return vec(v[0] / v[3], v[1] / v[3], v[2] / v[3]);
}

// from VRML to world coordinates
vec
ListenerCover::VCtoWC(vec p) const
{
    osg::Vec4 pos(p.x, p.y, p.z, 1.0);
    osg::Matrix tr = getVrmlBaseMat();
    osg::Vec4 v = pos * tr;

    return vec(v[0] / v[3], v[1] / v[3], v[2] / v[3]);
}

double
ListenerCover::getTime() const
{
    return cover->frameTime();
}

Player *
ListenerCover::createPlayer()
{
    Player *player = 0;
    PlayerFactory *factory = new PlayerFactory(this);

    factory->type = coCoviseConfig::getEntry("value", "COVER.Plugin.Vrml97.Audio", "None");

    factory->host = coCoviseConfig::getEntry("value", "COVER.Plugin.Vrml97.Audio.Host", "localhost");
    factory->port = coCoviseConfig::getInt("port", "COVER.Plugin.Vrml97.Audio.Host", 31231);
    factory->channels = coCoviseConfig::getInt("COVER.Plugin.Vrml97.AudioChannels", 2);
    bool headphones = coCoviseConfig::isOn("headphones", std::string("COVER.Plugin.Vrml97.Audio"), false);
    if (headphones)
        factory->channels = 2;
    factory->rate = coCoviseConfig::getInt("rate", "COVER.Plugin.Vrml97.AudioDevice", 44100);
    factory->bps = coCoviseConfig::getInt("bitsPerSample", "COVER.Plugin.Vrml97.Audio.Device", 16);
    factory->device = coCoviseConfig::getEntry("value", "COVER.Plugin.Vrml97.Audio.Device", "");
    factory->threaded = coCoviseConfig::isOn("COVER.Plugin.Vrml97.Audio.Threaded", true);

    player = factory->createPlayer();

    if (player && player->isPlayerMix())
    {
        PlayerMix *playerMix = (PlayerMix *)player;
        if (coCoviseConfig::isOn("COVER.Plugin.Vrml97.Audio.Surround", false))
        {
            playerMix->setDolbySurround(true);
        }
        else if (coCoviseConfig::isOn("headphones", std::string("COVER.Plugin.Vrml97.Audio"), false))
        {
            float separation = coCoviseConfig::getFloat("COVER.Plugin.Vrml97.Audio.Separation", 200.0);
            playerMix->setHeadphones(true);
            playerMix->setSeparation(separation);
        }
        else
        {
            for (int c = 0; c < factory->channels; c++)
            {
                char buf[1024];
                snprintf(buf, sizeof(buf), "COVER.Plugin.Vrml97.Audio.Speaker:%d", c);
                std::string speaker = coCoviseConfig::getEntry(buf);
                if (!speaker.empty())
                {
                    if (speaker == "SUBWOOFER")
                    {
                        playerMix->setSpeakerSpatialize(c, false);
                    }
                    else
                    {
                        float x, y, z;
                        if (sscanf(speaker.c_str(), "%f %f %f", &x, &y, &z) != 3)
                        {
                            cerr << "ListenerCover::createPlayer: sscanf failed" << endl;
                        }
                        fprintf(stderr, "Speaker %d at (%f,%f,%f)\n", c, x, y, z);
                        playerMix->setSpeakerPosition(c, x, y, z);
                    }
                }
            }
        }
    }

    return player;
}

void
ListenerCover::destroyPlayer(Player *player)
{
    if (player)
        delete player;
}
