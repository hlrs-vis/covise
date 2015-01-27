/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _VRML_PLAYER_FACTORY_
#define _VRML_PLAYER_FACTORY_

#include "vrmlexport.h"

namespace vrml
{

class Player;
class Listener;

class VRMLEXPORT PlayerFactory
{
public:
    const Listener *listener;
    std::string type;
    bool threaded;
    int channels;
    int rate;
    int bps;
    std::string device;
    std::string host;
    int port;

    PlayerFactory(Listener *listener);
    Player *createPlayer();
};
}
#endif
