/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include <util/coErr.h>
#include "PlayerFactory.h"
#include "Listener.h"
#include "PlayerAlsa.h"
#include "PlayerArts.h"
#include "PlayerAServer.h"
#include "PlayerEsd.h"
#include "PlayerIrixAL.h"
#include "PlayerOpenAL.h"
#include "PlayerOSS.h"
#include <cstring>

using std::endl;
using namespace vrml;

PlayerFactory::PlayerFactory(Listener *listener)
    : listener(listener)
    , threaded(false)
    , channels(2)
    , rate(44100)
    , bps(16)
    , port(0)
{
}

#ifdef _WIN32
#define strcasecmp stricmp
#endif

Player *PlayerFactory::createPlayer()
{
    if (type.empty())
    {
        //CERR <<  "have to set type" << endl;
        return NULL;
    }

    if (!strcasecmp(type.c_str(), "Alsa"))
    {
        //CERR << "PlayerFactory::createPlayer() - going to use Alsa!" << endl;
        return new PlayerAlsa(listener, threaded, channels, rate, bps, device);
    }
    else if (!strcasecmp(type.c_str(), "Arts"))
    {
        //CERR << "PlayerFactory::createPlayer() - going to use Arts!" << endl;
        return new PlayerArts(listener, threaded, channels, rate, bps);
    }
    else if (!strcasecmp(type.c_str(), "AServer"))
    {
        //CERR << "PlayerFactory::createPlayer() - going to use AServer!" << endl;
        return new PlayerAServer(listener, host, port);
    }
    else if (!strcasecmp(type.c_str(), "Esd"))
    {
        //CERR << "PlayerFactory::createPlayer() - going to use Esd!" << endl;
        return new PlayerEsd(listener, threaded, host);
    }
    else if (!strcasecmp(type.c_str(), "IrixAL"))
    {
        //CERR << "PlayerFactory::createPlayer() - going to use IrixAL!" << endl;
        return new PlayerIrixAL(listener, threaded, channels, rate, bps);
    }
    else if (!strcasecmp(type.c_str(), "OpenAL"))
    {
        //CERR << "PlayerFactory::createPlayer() - going to use OpenAL!" << endl;
        return new PlayerOpenAL(listener);
    }
    else if (!strcasecmp(type.c_str(), "OSS"))
    {
        //CERR << "PlayerFactory::createPlayer() - going to use OSS!" << endl;
        return new PlayerOSS(listener, threaded, channels, rate, bps, device);
    }
    else if (!strcasecmp(type.c_str(), "None"))
    {
        //CERR << "PlayerFactory::createPlayer() - no audio!" << endl;
        return NULL;
    }

    CERR << "unknown player type: " << type << endl;
    return NULL;
}
