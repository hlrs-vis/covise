/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _LISTENER_
#define _LISTENER_

#include <util/coExport.h>

#include <glm/vec3.hpp>
#include <glm/vec4.hpp>
#include <glm/mat4x4.hpp>

namespace opencover::audio
{

class Player;

class COVRAUDIOEXPORT Listener
{
public:
    Listener();
    ~Listener();

    Player *createPlayer();

    glm::mat4 getCurrentTransform() const;

    glm::vec3 WCtoOC(const glm::vec3 &pos) const;
    glm::vec3 OCtoWC(const glm::vec3 &pos) const;

    glm::vec3 getPosition() const;

    const glm::vec3 &getVelocity() const { return velocity; };
    void getOrientation(glm::vec3 *at, glm::vec3 *up) const;

    void update(float deltaTime, const glm::mat4 &transform);

private:
    glm::mat4 transform;
    glm::vec3 velocity;
};

}
#endif
