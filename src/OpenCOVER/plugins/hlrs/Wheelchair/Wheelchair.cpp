/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */


#include "Wheelchair.h"
#include <config/CoviseConfig.h>
#include <OpenVRUI/osg/mathUtils.h>
#include <util/byteswap.h>
#include <util/unixcompat.h>
#include <cover/coVRMSController.h>
#include <cover/coVRCollaboration.h>
#include <cover/VRSceneGraph.h>
#include <osg/LineSegment>
#include <osgUtil/IntersectionVisitor>
#include <osgUtil/LineSegmentIntersector>
#include "cover/coIntersection.h"
#include <cover/input/dev/Joystick/Joystick.h>
#include <cover/input/input.h>
#include <cover/input/deviceDiscovery.h>


#include <cover/input/input.h>

using namespace opencover;

static float zeroAngle = 1152.;


Wheelchair::Wheelchair()
: coVRPlugin(COVER_PLUGIN_NAME)
, coVRNavigationProvider("Wheelchair",this)
{
    wcData.countLeft = 0;
    wcData.countRight = 0;
    wcData.state = 0;
    float u = M_PI * 0.1; // 100mm Durchmesser
    // 30000 counts /revolution ;10000 counts on motor ;72/24 = 3 gear ratio
    mPerCount = u / ((-10000)*3);


        stepSizeUp=200;
        stepSizeDown=2000;
        coVRNavigationManager::instance()->registerNavigationProvider(this);
}
Wheelchair::~Wheelchair()
{
    coVRNavigationManager::instance()->unregisterNavigationProvider(this);
    running = false;
    delete udp;
}

bool Wheelchair::init()
{
    float floorHeight = VRSceneGraph::instance()->floorHeight();

	//const std::string host = covise::coCoviseConfig::getEntry("value", "Wheelchair.serverHost", "192.168.178.36");
	unsigned short serverPort = covise::coCoviseConfig::getInt("Wheelchair.serverPort", 31319);
	unsigned short localPort = covise::coCoviseConfig::getInt("Wheelchair.localPort", 31321);
    float x = covise::coCoviseConfig::getFloat("x", "COVER.Plugin.Wheelchair.Position", 0);
    float y = covise::coCoviseConfig::getFloat("y", "COVER.Plugin.Wheelchair.Position", 0);
    float z = covise::coCoviseConfig::getFloat("z", "COVER.Plugin.Wheelchair.Position", floorHeight);
    float h = covise::coCoviseConfig::getFloat("h", "COVER.Plugin.Wheelchair.Position", 0);
    float p = covise::coCoviseConfig::getFloat("p", "COVER.Plugin.Wheelchair.Position", 0);
    float r = covise::coCoviseConfig::getFloat("r", "COVER.Plugin.Wheelchair.Position", 0);

    MAKE_EULER_MAT(WheelchairPos, h, p, r);
    WheelchairPos.postMultTranslate(osg::Vec3(x, y, z));

    joystickNumber = covise::coCoviseConfig::getInt("Wheelchair.joystickNumber", 0);
    yIndex = covise::coCoviseConfig::getInt("Wheelchair.yIndex", 5);
    yScale = covise::coCoviseConfig::getFloat("Wheelchair.yScale", 10);
    xIndex = covise::coCoviseConfig::getInt("Wheelchair.xIndex", 2);
    xScale = covise::coCoviseConfig::getFloat("Wheelchair.xScale", 0.02);
    debugPrint = covise::coCoviseConfig::isOn("Wheelchair.debugPrint", false);
    udp = nullptr;
    dev = nullptr;
    if (coVRMSController::instance()->isMaster())
    {
        std::string host = "";
        for (const auto &i: opencover::Input::instance()->discovery()->getDevices())
        {
            if (i->pluginName == "Wheelchair")
            {
                host = i->address;
                udp = new UDPComm(host.c_str(), serverPort, localPort);
                if (!udp->isBad())
                {
                    ret = true;
                    start();
                }
                else
                {
                    std::cerr << "Wheelchair: falided to open local UDP port" << localPort << std::endl;
                    ret = false;
                }
                break;
            }
        }
           
        coVRMSController::instance()->sendSlaves(&ret, sizeof(ret));
    }
    else
    {
        coVRMSController::instance()->readMaster(&ret, sizeof(ret));
    }


    dev = (Joystick*)(Input::instance()->getDevice("joystick"));
        return true;
        
}

void
Wheelchair::run()
{
	running = true;
	doStop = false;
	if(coVRMSController::instance()->isMaster())
	{
		while (running && !doStop)
		{
			usleep(5000);
			this->updateThread();

		}
	}
}

void
Wheelchair::stop()
{
	doStop = true;
}


void Wheelchair::syncData()
{
        if (coVRMSController::instance()->isMaster())
        {
            coVRMSController::instance()->sendSlaves(&wcData, sizeof(wcData));
        }
        else
        {
            coVRMSController::instance()->readMaster(&wcData, sizeof(wcData));
        }
}

bool Wheelchair::update()
{
    if (isEnabled())
    {
        fprintf(stderr, "wc %ld %ld\n", (long)wcData.countLeft, (long)wcData.countRight);
        if (oldCountLeft == 0)
        {
            oldCountLeft = wcData.countLeft;
            oldCountRight = wcData.countRight;
	}

        float ml = (wcData.countLeft - oldCountLeft)* mPerCount;
        float mr = (wcData.countRight - oldCountRight)*mPerCount;
        oldCountLeft = wcData.countLeft;
        oldCountRight = wcData.countRight;
	
        double dT = cover->frameDuration();
        float wheelBase = 0.595;;
        float v = 0;
        float x = 0;
        /*if (dev && dev->number_axes[joystickNumber] >= 10)
        {
            if (debugPrint)
            {
                for (int i = 0; i < dev->number_axes[joystickNumber]; i++)
                    fprintf(stderr, "%d %f    ", i, dev->axes[joystickNumber][i]);
            }

            v = dev->axes[joystickNumber][yIndex] * yScale * coVRNavigationManager::instance()->getDriveSpeed();
            x = dev->axes[joystickNumber][xIndex];
            if (x < 0.002 && x > -0.002)
                x = 0;
        }*/
        float s = v * dT;
        if (ml != 0 || mr != 0)
        {
            s = (ml / 2.0 + mr / 2.0);
        }
        if (s > 10 || s < -10)
        {
            s = 0;
        }
        osg::Vec3 V(0, -s, 0);
        float rotAngle = tan((mr-ml)/wheelBase);
        //fprintf(stderr, "v: %f \n", v);
        if ((s < 0.0001 && s > -0.0001)) // straight
        {
        }
        fprintf(stderr, "s %f r %f\n", s, rotAngle);

        osg::Matrix relTrans;
        osg::Matrix relRot;
        relRot.makeRotate(-rotAngle, 0, 0, 1);
        relTrans.makeTranslate(V * 1000); // m to mm

        //fprintf(stderr,"bikeTrans: %f %f %f\n",bikeTrans(3, 0), bikeTrans(3, 1), bikeTrans(3, 2) );
        TransformMat = VRSceneGraph::instance()->getTransform()->getMatrix();
        TransformMat = TransformMat * relRot * relTrans;

        MoveToFloor();  

        if (coVRMSController::instance()->isMaster())
        {
            coVRMSController::instance()->sendSlaves((char*)TransformMat.ptr(), sizeof(TransformMat));
        }
        else
        {
            coVRMSController::instance()->readMaster((char*)TransformMat.ptr(), sizeof(TransformMat));
        }
        VRSceneGraph::instance()->getTransform()->setMatrix(TransformMat);
        coVRCollaboration::instance()->SyncXform();
    }
       
    return false;
}

void Wheelchair::MoveToFloor()
{
    float floorHeight = VRSceneGraph::instance()->floorHeight();

    //  just adjust height here


    //osg::Vec3 pos = WheelchairPos.getTrans();
    osg::Vec3 pos(-wheelWidth/2.0, 0, 0);
    pos = pos * WheelchairPos;
    pos[2]-=floorHeight;
    osg::Vec3 pos2(-wheelWidth / 2.0, wheelBase, 0);
    pos2 = pos2 * WheelchairPos;
    pos2[2]-=floorHeight;
    osg::Vec3 pos3(wheelWidth / 2.0, wheelBase, 0);
    pos3 = pos3 * WheelchairPos;
    pos3[2]-=floorHeight;

    // down segment
    osg::Vec3 p0, q0;
    p0.set(pos[0], pos[1], pos[2] + floorHeight + stepSizeUp);
    q0.set(pos[0], pos[1], pos[2] + floorHeight - stepSizeDown);

    osg::ref_ptr<osg::LineSegment> ray[3];
    ray[0] = new osg::LineSegment(p0, q0);

    // down segment 2
    p0.set(pos2[0], pos2[1], pos2[2] + floorHeight + stepSizeUp);
    q0.set(pos2[0], pos2[1], pos2[2] + floorHeight - stepSizeDown);
    ray[1] = new osg::LineSegment(p0, q0);

    // down segment 3
    p0.set(pos3[0], pos3[1], pos3[2] + floorHeight + stepSizeUp);
    q0.set(pos3[0], pos3[1], pos3[2] + floorHeight - stepSizeDown);
    ray[2] = new osg::LineSegment(p0, q0);

    osg::ref_ptr<osgUtil::IntersectorGroup> igroup = new osgUtil::IntersectorGroup;
    osg::ref_ptr<osgUtil::LineSegmentIntersector> intersectors[3];
    for (int i=0; i<3; ++i)
    {
        intersectors[i] = coIntersection::instance()->newIntersector(ray[i]->start(), ray[i]->end());
        igroup->addIntersector(intersectors[i]);
    }

    osgUtil::IntersectionVisitor visitor(igroup);
    visitor.setTraversalMask(Isect::Walk);
    VRSceneGraph::instance()->getTransform()->accept(visitor);

    bool haveIsect[3];
    for (int i=0; i<3; ++i)
    {
        haveIsect[i] = intersectors[i]->containsIntersections();
       fprintf(stderr,"haveIsect %d %d\n",i,haveIsect[i]);
    }
    if (!haveIsect[0] && !haveIsect[1])
    {
        oldFloorNode = NULL;
        return;

    }

    osg::Node *floorNode = NULL;

    float dist = FLT_MAX;
    osgUtil::LineSegmentIntersector::Intersection isect;
    if (haveIsect[0])
    {
        isect = intersectors[0]->getFirstIntersection();
        dist = isect.getWorldIntersectPoint()[2] - floorHeight;
        floorNode = isect.nodePath.back();
    }
    if (haveIsect[1] && fabs(intersectors[1]->getFirstIntersection().getWorldIntersectPoint()[2] - floorHeight) < fabs(dist))
    {
        isect = intersectors[1]->getFirstIntersection();
        dist = isect.getWorldIntersectPoint()[2] - floorHeight;
        floorNode = isect.nodePath.back();
    }
    if(!haveIsect[0]&&!haveIsect[1])
    {
        dist = 0;
    }

    //  get xform matrix
    //TransformMat = VRSceneGraph::instance()->getTransform()->getMatrix();

    if (floorNode && floorNode == oldFloorNode)
    {
        // we are walking on the same object as last time so move with the object if it is moving
        osg::Matrix modelTransform;
        modelTransform.makeIdentity();
        int on = oldNodePath.size() - 1;
        bool notSamePath = false;
        for (int i = isect.nodePath.size() - 1; i >= 0; i--)
        {
            osg::Node*n = isect.nodePath[i];
            if (n == cover->getObjectsRoot())
                break;
            osg::MatrixTransform *t = dynamic_cast<osg::MatrixTransform *>(n);
            if (t != NULL)
            {
                modelTransform = modelTransform * t->getMatrix();
            }
            // check if this is really the same object as it could be a reused object thus compare the whole NodePath
            // instead of just the last node
            if (on < 0 || n != oldNodePath[on])
            {
                //oops, not same path
                notSamePath = true;
            }
            on--;
        }
        if (notSamePath)
        {
            oldFloorMatrix = modelTransform;
            oldFloorNode = floorNode;
            oldNodePath = isect.nodePath;
        }
        else if (modelTransform != oldFloorMatrix)
        {

            //osg::Matrix iT;
            osg::Matrix iS;
            osg::Matrix S;
            osg::Matrix imT;
            //iT.invert_4x4(dcs_mat);
            float sf = cover->getScale();
            S.makeScale(sf, sf, sf);
            sf = 1.0 / sf;
            iS.makeScale(sf, sf, sf);
            imT.invert_4x4(modelTransform);
	    
	fprintf(stderr,"oops %f \n",dist);
           // TransformMat = iS *imT*oldFloorMatrix * S * TransformMat;
            oldFloorMatrix = modelTransform;
            // set new xform matrix
            //VRSceneGraph::instance()->getTransform()->setMatrix(TransformMat);
            // now we have a new base matrix and we have to compute the floor height again, otherwise we will jump up and down
            //
            VRSceneGraph::instance()->getTransform()->accept(visitor);

            osgUtil::IntersectionVisitor visitor(igroup);
            igroup->reset();
            visitor.setTraversalMask(Isect::Walk);
            VRSceneGraph::instance()->getTransform()->accept(visitor);

            for (int i=0; i<3; ++i)
                haveIsect[i] = intersectors[i]->containsIntersections();
            dist = FLT_MAX;
            if (haveIsect[0])
            {
                isect = intersectors[0]->getFirstIntersection();
                dist = isect.getWorldIntersectPoint()[2] - floorHeight;
                floorNode = isect.nodePath.back();
            }
            if (haveIsect[1] && fabs(intersectors[1]->getFirstIntersection().getWorldIntersectPoint()[2] - floorHeight) < fabs(dist))
            {
                isect = intersectors[1]->getFirstIntersection();
                dist = isect.getWorldIntersectPoint()[2] - floorHeight;
                floorNode = isect.nodePath.back();
            }
    if(!haveIsect[0]&&!haveIsect[1])
    {
        dist = 0;
    }
        }
    }

    if (haveIsect[0] && haveIsect[1] && haveIsect[2])
    {
        isect = intersectors[0]->getFirstIntersection();
        osg::Vec3 p0 = isect.getWorldIntersectPoint();
        isect = intersectors[1]->getFirstIntersection();
        osg::Vec3 p1 = isect.getWorldIntersectPoint();
        isect = intersectors[2]->getFirstIntersection();
        osg::Vec3 p2 = isect.getWorldIntersectPoint();
        osg::Vec3 v1 = p1 - p0;
        osg::Vec3 v2 = p2 - p1;
        v1.normalize();
        v2.normalize();
        wcNormal = v2 ^ v1;
        wcNormal.normalize();
        wcDataOut.normal[0] = wcNormal[0];
        wcDataOut.normal[1] = wcNormal[1];
        wcDataOut.normal[2] = wcNormal[2];
        wcDataOut.direction[0] = WheelchairPos(0, 1);
        wcDataOut.direction[1] = WheelchairPos(1, 1);
        wcDataOut.direction[2] = WheelchairPos(2, 1);
        //wcDataOut.downhillForce = 100.0;
        wcDataOut.downhillForce = calculateDownhillForce(wcNormal,osg::Vec3(wcDataOut.direction[0],wcDataOut.direction[1],wcDataOut.direction[2]));
        //fprintf(stderr, "test1");
        fprintf(stderr, "normal\nx: %f y: %f\n z:%f\n direction\nx: %f y: %f\n z:%f\ndownhillForce: %f\n",
                wcNormal[0], wcNormal[1], wcNormal[2], wcDataOut.direction[0], wcDataOut.direction[1],
                wcDataOut.direction[2], wcDataOut.downhillForce);
    }

	fprintf(stderr,"dist %f \n",dist);

    //  apply translation , so that isectPt is at floorLevel
    osg::Matrix tmp;
    tmp.makeTranslate(0, 0, -dist);
    TransformMat.postMult(tmp);

    // set new xform matrix
    //VRSceneGraph::instance()->getTransform()->setMatrix(TransformMat);

    if ((floorNode != oldFloorNode) && !isect.nodePath.empty())
    {
        osg::Matrix modelTransform;
        modelTransform.makeIdentity();
        for (int i = isect.nodePath.size() - 1; i >= 0; i--)
        {
            osg::Node*n = isect.nodePath[i];
            if (n == cover->getObjectsRoot())
                break;
            osg::MatrixTransform *t = dynamic_cast<osg::MatrixTransform *>(n);
            if (t != NULL)
            {
                modelTransform = modelTransform * t->getMatrix();
            }
        }
        oldFloorMatrix = modelTransform;
        oldNodePath = isect.nodePath;
    }

    oldFloorNode = floorNode;

    // do not sync with remote, they will do the same
    // on their side SyncXform();
	
    // coVRCollaboration::instance()->SyncXform();
}
void Wheelchair::setEnabled(bool flag)
{
    coVRNavigationProvider::setEnabled(flag);
    //WakeUp Wheelchair
    if (udp)
    {
        if (flag)
        {
            udp->send("start");
        }
        else
        {
            udp->send("stop");
        }
    }
    Initialize();

}
void Wheelchair::updateThread()
{
    if (udp)
    {
        char tmpBuf[10000];
        int status = udp->receive(&tmpBuf, 10000);


        if (status > 0 && status >= sizeof(WCData))
        {

            {
                OpenThreads::ScopedLock<OpenThreads::Mutex> lock(mutex);
                memcpy(&wcData, tmpBuf, sizeof(WCData));
                udp->send(&wcDataOut, sizeof(wcDataOut));
            }

        }
        else if (status == -1)
        {
            if (isEnabled()) // otherwise we are not supposed to receive anything
            {
                std::cerr << "Wheelchair::update: error while reading data" << std::endl;
                if (udp) // try to wake up the Wheelchair (if the first start UDP message was lost)
                    udp->send("start");
            }
            return;
        }
        else
        {
            std::cerr << "Wheelchair::update: received invalid no. of bytes: recv=" << status << ", got=" << status << std::endl;
            return;
        }

    }
}
void Wheelchair::Initialize()
{
        if (coVRMSController::instance()->isMaster())
        {
        }
}

float Wheelchair::calculateDownhillForce(const osg::Vec3 &n,const osg::Vec3 &direction)
{

    osg::Vec3 g(0.0, 0.0, -1.0);
    osg::Vec3 b = n*((g*n)/(n*n));
    osg::Vec3 gEbene = g-b;

    osg::Vec3 gwc= direction * ((gEbene*direction)/(direction*direction));

    float downhillForce = gwc.length();
    if((gwc*direction) <0)
       downhillForce *= -1;

    return downhillForce * 100 * 9.81;
}

unsigned char Wheelchair::getButton()
{
	return wcData.state;
}
COVERPLUGIN(Wheelchair)
