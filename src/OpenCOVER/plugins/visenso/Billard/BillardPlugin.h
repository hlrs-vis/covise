/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _BILLARD_PLUGIN_H
#define _BILLARD_PLUGIN_H

#include <cover/coVRPlugin.h>
class BillardBall;

namespace opencover
{
class coVR3DTransRotInteractor;
}
using namespace opencover;

class MessageReceiver;

#define NUM_BALLS 16
#define BALL_SIZE 57.2
#define BALL_RADIUS 28.6
#define TABLE_HEIGHT 800
#define TABLE_WIDTH 1270
#define TABLE_LENGTH 2540

class BillardPlugin : public coVRPlugin
{
public:
    static BillardPlugin *plugin;
    BillardPlugin();
    virtual ~BillardPlugin();
    virtual bool init();
    void preFrame();

private:
    //coVR3DTransRotInteractor *balls[NUM_BALLS];
    BillardBall *balls[NUM_BALLS];
    osg::Matrix startMatrices[NUM_BALLS];
    osg::Vec3 handPosLastFrame_, queuePosLastFrame_;
    MessageReceiver *messageReceiver_;
    int port_;
    int timeout_;
    void handleTokens(const std::vector<std::string> &tokens);
};

#endif
