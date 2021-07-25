/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */


#include "Wheelchair.h"
#include <config/CoviseConfig.h>
#include <util/byteswap.h>
#include <util/unixcompat.h>
#include <cover/coVRMSController.h>
#include <cover/VRSceneGraph.h>
#include <osg/LineSegment>
#include <osgUtil/IntersectionVisitor>
#include <osgUtil/LineSegmentIntersector>
#include "cover/coIntersection.h"


#include <cover/input/input.h>

using namespace opencover;

static float zeroAngle = 1152.;


Wheelchair::Wheelchair()
	: coVRNavigationProvider("Wheelchair",this)
{
	init();
	sbData.fl = 0;
	sbData.fr = 0;
	sbData.rl = 0;
	sbData.rr = 0;
	sbData.button = 0;
	sbControl.cmd = 0;
	sbControl.value = 0;

        stepSizeUp=200;
        stepSizeDown=2000;
        coVRNavigationManager::instance()->registerNavigationProvider(this);
}
Wheelchair::~Wheelchair()
{
        coVRNavigationManager::instance()->unregisterNavigationProvider(this);
}

bool Wheelchair::init()
{

	const std::string host = covise::coCoviseConfig::getEntry("value", "Wheelchair.serverHost", "192.168.178.36");
	unsigned short serverPort = covise::coCoviseConfig::getInt("Wheelchair.serverPort", 31319);
	unsigned short localPort = covise::coCoviseConfig::getInt("Wheelchair.localPort", 31319);
	std::cerr << "Wheelchair config: UDP: serverHost: " << host << ", localPort: " << localPort << ", serverPort: " << serverPort << std::endl;

    joystickNumber = covise::coCoviseConfig::getInt("Wheelchair.joystickNumber", 0);
    yIndex = covise::coCoviseConfig::getInt("Wheelchair.yIndex", 5);
    yScale = covise::coCoviseConfig::getFloat("Wheelchair.yScale", 10);
    xIndex = covise::coCoviseConfig::getInt("Wheelchair.xIndex", 2);
    xScale = covise::coCoviseConfig::getFloat("Wheelchair.xScale", 0.02);
    debugPrint = covise::coCoviseConfig::isOn("Wheelchair.debugPrint", false);

    dev = dynamic_cast<Joystick*>(Input::instance()->getDevice("joystick"));
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
            coVRMSController::instance()->sendSlaves(&sbData, sizeof(sbData));
        }
        else
        {
            coVRMSController::instance()->readMaster(&sbData, sizeof(sbData));
        }
}

bool Wheelchair::update()
{
    if (isEnabled())
    {
        double dT = cover->frameDuration();
        float wheelBase = 0.98;
        if (dev && dev->number_axes[joystickNumber] >= 1)
        {
            if (debugPrint)
            {
                for (int i = 0; i < dev->number_axes[joystickNumber]; i++)
                    fprintf(stderr, "%d %f    ", i, dev->axes[joystickNumber][i]);
            }

            float v = dev->axes[joystickNumber][yIndex]*yScale*coVRNavigationManager::instance()->getDriveSpeed();
            float x = dev->axes[joystickNumber][xIndex];
            if (x < 0.002 && x > -0.002)
                x = 0;

        float s = v * dT;
        osg::Vec3 V(0, -s, 0);
        wheelBase = 0.5;
        float rotAngle = 0.0;
        //fprintf(stderr, "v: %f \n", v);
        if ((s < 0.0001 && s > -0.0001)) // straight
        {
        }
        else
        {
            rotAngle = x * xScale;
        }

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
    }
       
    }
    return false;
}

void Wheelchair::MoveToFloor()
{
    float floorHeight = VRSceneGraph::instance()->floorHeight();

    //  just adjust height here

    osg::Matrix viewer = cover->getViewerMat();
    osg::Vec3 pos = viewer.getTrans();

    // down segment
    osg::Vec3 p0, q0;
    p0.set(pos[0], pos[1], floorHeight + stepSizeUp);
    q0.set(pos[0], pos[1], floorHeight - stepSizeDown);

    osg::ref_ptr<osg::LineSegment> ray[2];
    ray[0] = new osg::LineSegment(p0, q0);

    // down segment 2
    p0.set(pos[0], pos[1] + 10, floorHeight + stepSizeUp);
    q0.set(pos[0], pos[1] + 10, floorHeight - stepSizeDown);
    ray[1] = new osg::LineSegment(p0, q0);

    osg::ref_ptr<osgUtil::IntersectorGroup> igroup = new osgUtil::IntersectorGroup;
    osg::ref_ptr<osgUtil::LineSegmentIntersector> intersectors[2];
    for (int i=0; i<2; ++i)
    {
        intersectors[i] = coIntersection::instance()->newIntersector(ray[i]->start(), ray[i]->end());
        igroup->addIntersector(intersectors[i]);
    }

    osgUtil::IntersectionVisitor visitor(igroup);
    visitor.setTraversalMask(Isect::Walk);
    VRSceneGraph::instance()->getTransform()->accept(visitor);

    bool haveIsect[2];
    for (int i=0; i<2; ++i)
        haveIsect[i] = intersectors[i]->containsIntersections();
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
            TransformMat = iS *imT*oldFloorMatrix * S * TransformMat;
            oldFloorMatrix = modelTransform;
            // set new xform matrix
            VRSceneGraph::instance()->getTransform()->setMatrix(TransformMat);
            // now we have a new base matrix and we have to compute the floor height again, otherwise we will jump up and down
            //
            VRSceneGraph::instance()->getTransform()->accept(visitor);

            osgUtil::IntersectionVisitor visitor(igroup);
            igroup->reset();
            visitor.setTraversalMask(Isect::Walk);
            VRSceneGraph::instance()->getTransform()->accept(visitor);

            for (int i=0; i<2; ++i)
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
        }
    }


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
    Initialize();

}
void Wheelchair::updateThread()
{
	
}
void Wheelchair::Initialize()
{
        if (coVRMSController::instance()->isMaster())
        {
        }
}

osg::Vec3d Wheelchair::getNormal()
{
	OpenThreads::ScopedLock<OpenThreads::Mutex> lock(mutex);
	float w = (sbData.fl + sbData.fr + sbData.rl + sbData.rr) / 4.0;
	float fs = sbData.fl + sbData.fr;
	float rs = sbData.rl + sbData.rr;
	float sl = sbData.fl + sbData.rl;
	float sr = sbData.fr + sbData.rr;

	return osg::Vec3d((sl - sr) / w, (fs - rs) / w, 1.0);
}

float Wheelchair::getWeight() // weight in Kg
{
	OpenThreads::ScopedLock<OpenThreads::Mutex> lock(mutex);
	return float((sbData.fl + sbData.fr + sbData.rl + sbData.rr) / 4.0);
}

unsigned char Wheelchair::getButton()
{
	return sbData.button;
}
COVERPLUGIN(Wheelchair)

