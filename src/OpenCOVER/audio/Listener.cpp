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
#include <util/unixcompat.h>

#include "Player.h"
#include "Listener.h"

#include <config/CoviseConfig.h>

using covise::coCoviseConfig;
using namespace opencover::audio;

Listener::Listener()
    : velocity(0.0, 0.0, 0.0)
{
}

Listener::~Listener()
{
}

void Listener::update(float deltaTime, const glm::mat4 &transform)
{
    if (deltaTime == 0)
    {
        velocity = glm::vec3(0, 0, 0);
    }
    else
    {
        glm::vec4 origin(0.0, 0.0, 0.0, 1.0);
        glm::vec3 last_position = (origin * this->transform).xyz();
        glm::vec3 current_position = (origin * transform).xyz();
        velocity = (current_position - last_position) / deltaTime;
    }

    this->transform = transform;
}

glm::vec3 Listener::getPosition() const
{
    const glm::mat4 &m = transform;
    return glm::vec3(m[3][0], m[3][1], m[3][2]); // TODO: check if we need to flip columns/rows
}

void Listener::getOrientation(glm::vec3 *at, glm::vec3 *up) const
{
    glm::mat4 m = glm::inverse(transform);
    *at = glm::normalize((glm::vec4(0.0, 1.0, 0.0, 1.0) * m).xyz());
    *up = glm::normalize((glm::vec4(0.0, 0.0, 1.0, 1.0) * m).xyz());
}

// from object to world coordinates
glm::vec3 Listener::OCtoWC(const glm::vec3 &pos) const
{
    glm::vec4 v = glm::vec4(pos, 1.0) * transform;
    return v.xyz() / v.w;
}

// from world to object coordinates
glm::vec3 Listener::WCtoOC(const glm::vec3 &pos) const
{
    glm::vec4 v = glm::vec4(pos, 1.0) * glm::inverse(transform);
    return v.xyz() / v.w;
}

Player *
Listener::createPlayer()
{
    std::string type = coCoviseConfig::getEntry("value", "COVER.Plugin.Vrml97.Audio", "none");
    return Player::createPlayer(this, type);
}

