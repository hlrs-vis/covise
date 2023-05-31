/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "BillardPlugin.h"
#include "MessageReceiver.h"
#include "BillardBall.h"

using namespace covise;
using namespace opencover;

#include <cover/coVRPluginSupport.h>
#include <PluginUtil/coVR3DTransRotInteractor.h>
#include <cover/coVRMSController.h>
#include <cover/VRSceneGraph.h>
#include <cover/coVRFileManager.h>
#include <config/CoviseConfig.h>

#include <osg/Matrix>
#include <osg/Vec3>
#include <osg/ShapeDrawable>
#include <osg/Geometry>

#include <stdio.h>
template <class T>
void operator>>(const std::string &s, T &converted)
{
    std::istringstream iss(s);
    iss >> converted;
    if ((iss.rdstate() & std::istringstream::failbit) != 0)
    {
        std::cerr << "Error in conversion from string \""
                  << s
                  << "\" to type "
                  << typeid(T).name()
                  << std::endl;
    }
}
BillardPlugin *BillardPlugin::plugin = NULL;

BillardPlugin::BillardPlugin()
: coVRPlugin(COVER_PLUGIN_NAME)
{
    if (cover->debugLevel(0))
        fprintf(stderr, "\nBillardPlugin::BillardPlugin\n");

    // read configuration file
    port_ = (int)coCoviseConfig::getInt("value", "COVER.Plugin.Billard.Port", 5555);
    timeout_ = (int)coCoviseConfig::getInt("value", "COVER.Plugin.Billard.Timeout", 3600);
}

BillardPlugin::~BillardPlugin()
{
    if (cover->debugLevel(0))
        fprintf(stderr, "\nBillardPlugin::~BillardPlugin\n");

    if (coVRMSController::instance()->isMaster())
    {
        delete messageReceiver_;
    }
}

bool BillardPlugin::init()
{
    if (plugin)
        return false;

    if (cover->debugLevel(0))
        fprintf(stderr, "\nBillardPlugin::init\n");

    // set plugin
    BillardPlugin::plugin = this;

    // info to coord system:
    // cover world coord system is typically the midlle of the VR screen
    // cover floor is defined in config.xml FloorHeight for example -1300
    // the typical height of a billard table is between -750 and 850 mm
    // the table area  of the model is 800 mm above the table foot
    // -> the table plate is at -500

    // floor & table
    float floorHeight = VRSceneGraph::instance()->floorHeight();

    osg::Node *modelNode = coVRFileManager::instance()->loadIcon("Billard/Billardtisch");
    if (modelNode)
    {
        osg::MatrixTransform *trans = new osg::MatrixTransform();
        osg::Matrix m;
        m.makeTranslate(0, 0, floorHeight + TABLE_HEIGHT);
        trans->setMatrix(m);
        cover->getObjectsRoot()->addChild(trans);
        trans->addChild(modelNode);
    }
    else
    {
        osg::Box *floorBox = new osg::Box(osg::Vec3(0, 0, floorHeight), 3000, 3000, 1);
        osg::TessellationHints *hint = new osg::TessellationHints();
        hint->setDetailRatio(0.5);
        osg::ShapeDrawable *floorDrawable = new osg::ShapeDrawable(floorBox, hint);
        floorDrawable->setColor(osg::Vec4(0, 0.8, 0, 1));
        osg::Geode *floorGeode = new osg::Geode();
        floorGeode->addDrawable(floorDrawable);
        cover->getObjectsRoot()->addChild(floorGeode);
        osg::Box *box = new osg::Box(osg::Vec3(0, 0, floorHeight + TABLE_HEIGHT), TABLE_WIDTH, TABLE_LENGTH, 1);
        osg::ShapeDrawable *boxDrawable = new osg::ShapeDrawable(box, hint);
        boxDrawable->setColor(osg::Vec4(0, 0.8, 0, 1));
        osg::Geode *boxGeode = new osg::Geode();
        boxGeode->addDrawable(boxDrawable);
        cover->getObjectsRoot()->addChild(boxGeode);
    }

    // define startPositions
    float h = floorHeight + TABLE_HEIGHT + BALL_RADIUS;
    startMatrices[0].makeTranslate(osg::Vec3(0., -TABLE_LENGTH / 4., h)); // white ball
    startMatrices[1].makeTranslate(osg::Vec3(0., 500., h)); // top ball
    startMatrices[2].makeTranslate(osg::Vec3(-BALL_RADIUS, 500.0 + BALL_SIZE, h));
    startMatrices[3].makeTranslate(osg::Vec3(BALL_RADIUS, 500.0 + BALL_SIZE, h));
    startMatrices[4].makeTranslate(osg::Vec3(0.0, 500.0 + 2.0 * BALL_SIZE, h));
    startMatrices[5].makeTranslate(osg::Vec3(-2.0 * BALL_RADIUS, 500.0 + 2.0 * BALL_SIZE, h));
    startMatrices[6].makeTranslate(osg::Vec3(2.0 * BALL_RADIUS, 500.0 + 2.0 * BALL_SIZE, h));
    startMatrices[7].makeTranslate(osg::Vec3(-BALL_RADIUS, 500.0 + 3.0 * BALL_SIZE, h));
    startMatrices[8].makeTranslate(osg::Vec3(BALL_RADIUS, 500.0 + 3.0 * BALL_SIZE, h));
    startMatrices[9].makeTranslate(osg::Vec3(-3.0 * BALL_RADIUS, 500.0 + 3.0 * BALL_SIZE, h));
    startMatrices[10].makeTranslate(osg::Vec3(3.0 * BALL_RADIUS, 500.0 + 3.0 * BALL_SIZE, h));
    startMatrices[11].makeTranslate(osg::Vec3(0.0, 500.0 + 4.0 * BALL_SIZE, h));
    startMatrices[12].makeTranslate(osg::Vec3(-2 * BALL_RADIUS, 500.0 + 4.0 * BALL_SIZE, h));
    startMatrices[13].makeTranslate(osg::Vec3(2 * BALL_RADIUS, 500.0 + 4.0 * BALL_SIZE, h));
    startMatrices[14].makeTranslate(osg::Vec3(-4 * BALL_RADIUS, 500.0 + 4.0 * BALL_SIZE, h));
    startMatrices[15].makeTranslate(osg::Vec3(4 * BALL_RADIUS, 500.0 + 4.0 * BALL_SIZE, h));

    // create balls
    balls[0] = new BillardBall(startMatrices[0], BALL_SIZE, "Billard/weiss");
    balls[1] = new BillardBall(startMatrices[1], BALL_SIZE, "Billard/gelb_1");
    balls[2] = new BillardBall(startMatrices[2], BALL_SIZE, "Billard/blau_2");
    balls[3] = new BillardBall(startMatrices[3], BALL_SIZE, "Billard/rot_3");
    balls[4] = new BillardBall(startMatrices[4], BALL_SIZE, "Billard/lila_4");
    balls[5] = new BillardBall(startMatrices[5], BALL_SIZE, "Billard/orange_5");
    balls[6] = new BillardBall(startMatrices[6], BALL_SIZE, "Billard/gruen_6");
    balls[7] = new BillardBall(startMatrices[7], BALL_SIZE, "Billard/braun_7");
    balls[8] = new BillardBall(startMatrices[8], BALL_SIZE, "Billard/schwarz_8");
    balls[9] = new BillardBall(startMatrices[9], BALL_SIZE, "Billard/gelb_9");
    balls[10] = new BillardBall(startMatrices[10], BALL_SIZE, "Billard/blau_10");
    balls[11] = new BillardBall(startMatrices[11], BALL_SIZE, "Billard/rot_11");
    balls[12] = new BillardBall(startMatrices[12], BALL_SIZE, "Billard/lila_12");
    balls[13] = new BillardBall(startMatrices[13], BALL_SIZE, "Billard/orange_13");
    balls[14] = new BillardBall(startMatrices[14], BALL_SIZE, "Billard/gruen_14");
    balls[15] = new BillardBall(startMatrices[15], BALL_SIZE, "Billard/braun_15");

    for (int i = 0; i < NUM_BALLS; i++)
    {
        //balls[i] = new BillardBall(startMatrices[i], BALL_SIZE, "Billard/gelb_9");
        //osg::BoundingBox bb;
        //bb = cover->getBBox(balls[i]->getNode());
        //fprintf(stderr, " size of Billard/gelb9=%f\n", bb._max.x()-bb._min.x());

        //balls[i] = new coVR3DTransRotInteractor(startMatrices[i], BALL_SIZE,coInteraction::ButtonA, "Menu", "ball", coInteraction::Medium );
        balls[i]->enableIntersection();
    }

    // create socket to simulation
    if (coVRMSController::instance()->isMaster())
    {
        std::cerr << "Creating MessageReceiver..." << std::endl;
        messageReceiver_ = new MessageReceiver(port_, timeout_);
    }

    osg::Matrix m = cover->getPointerMat();
    handPosLastFrame_ = m.getTrans();
    queuePosLastFrame_ = (osg::Vec3(0, 1000, 0)) * m;

    return true;
}

void BillardPlugin::preFrame()
{
    //fprintf(stderr,"BillardPlugin::preFrame\n");
    // delta t of 1 frame
    float t = cover->frameDuration();

    // hand matrix
    osg::Matrix m = cover->getPointerMat();

    // hand pos
    osg::Vec3 hp = m.getTrans();
    //fprintf(stderr,"---- hp=[%f %f %f]\n", hp[0], hp[1], hp[2]);

    // calc velocity magnitude of intersection ray at ray endpoint
    osg::Vec3 qp(0, 1000, 0);
    qp = qp * m;
    //fprintf(stderr,"---- qp=[%f %f %f]\n", qp[0], qp[1], qp[2]);

    // distance of ray endpoint between this frame and last frame
    float d = (qp - queuePosLastFrame_).length();
    // distance of hand
    //float d = (hp-handPosLastFrame_).length();

    //fprintf(stderr,"hp=[%f %f %f] oldhp=[%f %f %f]\n", hp[0], hp[1], hp[2], handPosLastFrame_[0], handPosLastFrame_[1], handPosLastFrame_[2]);
    //fprintf(stderr,"frameDuration=%f distance %f\n", t, d);
    // velocity = distance/time
    float vmag = d / t;
    handPosLastFrame_ = hp;
    queuePosLastFrame_ = qp;
    // direction of intersection ray
    osg::Vec4 dir;
    dir = (osg::Vec4(0.0, 1.0, 0.0, 0.0)) * m;
    dir.normalize();
    //fprintf(stderr,"dir=[%f %f %f]\n", dir[0], dir[1], dir[2]);
    dir = dir * vmag;

    // check all balls for intersection
    for (int i = 0; i < NUM_BALLS; i++)
    {
        balls[i]->preFrame();
        if (balls[i]->wasHit())
        {
            // pointer ray intersects ball
            osg::Vec3 pw = balls[i]->getHitPos(); //world coord
            osg::Matrix w_to_o = cover->getInvBaseMat();
            osg::Vec3 po = pw * w_to_o; // object coord
            //fprintf(stderr,"ball %d was hit word pos=[%f %f %f] obj pos=[%f %f %f] velocitymag=%f velDir=[%f %f %f]\n", i, pw[0], pw[1], pw[2], po[0], po[1], po[2], vmag, dir[0], dir[1], dir[2]);

            //send index and intersection point of hit ball to simulation
            std::stringstream buf;
            //char buf[2048];
            //sprintf(buf,"TOUCH_MSG %d %f %f %f %f %f %f", i, pw[0], pw[1], pw[2], dir[0], dir[1], dir[2]);
            buf << i;
            buf << po[0];
            buf << po[1];
            buf << po[2];
            buf << dir[0];
            buf << dir[1];
            buf << dir[2];
            std::string sbuf = buf.str();
            messageReceiver_->send(sbuf);
            fprintf(stderr, "sending buffer [%s]\n", sbuf.c_str());
        }
    }

    if (coVRMSController::instance()->isMaster())
    {
        std::vector<std::string> queue = messageReceiver_->popMessageQueue();
        // MOVE_MSG    px0 py0 pz0 rx0 ry0 rz0     px1 py1 pz1 rx1 ry1 rz1 ....px15

        for (size_t i = 0; i < queue.size(); i++)
        {
            // Ray: Debug output
            std::cout << "queue[" << i << "] = " << queue[i] << std::endl;

            // Ray: It certainly would be NICER to interpret binary data
            // break up message into tokens
            std::vector<std::string> tokens;
            std::string token;
            std::istringstream iss(queue[i]);
            while (std::getline(iss, token, ' '))
            {
                tokens.push_back(token);
            }
            if (tokens[0] == "MOVE_MSG")
                handleTokens(tokens);
        }
    }
}

void BillardPlugin::handleTokens(const std::vector<std::string> &tokens)
{
    int ti = 0;
    if (tokens[0] == "MOVE_MSG")
    {

        ti++;
        for (int i = 0; i < NUM_BALLS; i++)
        {
            float px, py, pz, rx, ry, rz;
            tokens[ti] >> px;
            ti++;
            tokens[ti] >> py;
            ti++;
            tokens[ti] >> pz;
            ti++;
            tokens[ti] >> rx;
            ti++;
            tokens[ti] >> ry;
            ti++;
            tokens[ti] >> rz;
            ti++;
        }
        osg::Matrix m;
    }
    // RAY: Don't we need to pop the tokens?
}

COVERPLUGIN(BillardPlugin)
