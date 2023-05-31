/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */


#include "Skateboard.h"
#include <util/UDPComm.h>
#include <config/CoviseConfig.h>
#include <util/byteswap.h>
#include <util/unixcompat.h>
#include <cover/coVRMSController.h>
#include <cover/coVRCollaboration.h>
#include <cover/VRSceneGraph.h>
#include <osg/LineSegment>
#include <osgUtil/IntersectionVisitor>
#include <osgUtil/LineSegmentIntersector>
#include "cover/coIntersection.h"
#include <OpenVRUI/osg/mathUtils.h> //for MAKE_EULER_MAT
#include <cover/input/input.h>
#include <cover/input/deviceDiscovery.h>
using namespace opencover;

static float zeroAngle = 1152.;


Skateboard::Skateboard()
: coVRPlugin(COVER_PLUGIN_NAME)
, udp(NULL)
, coVRNavigationProvider("Skateboard",this)
{
	sbData.fl = 0;
	sbData.fr = 0;
	sbData.rl = 0;
	sbData.rr = 0;
	sbData.button = 0;
	sbControl.cmd = 0;
	sbControl.value = 0;

        stepSizeUp=2000;
        stepSizeDown=2000;
        coVRNavigationManager::instance()->registerNavigationProvider(this);
}
Skateboard::~Skateboard()
{
    coVRNavigationManager::instance()->unregisterNavigationProvider(this);
    running = false;

	delete udp;
}

bool Skateboard::init()
{
	delete udp;
        float floorHeight = VRSceneGraph::instance()->floorHeight();

	const std::string host = covise::coCoviseConfig::getEntry("value", "COVER.Plugin.Skateboard.serverHost", "192.168.178.36");
	unsigned short serverPort = covise::coCoviseConfig::getInt("COVER.Plugin.Skateboard.serverPort", 31319);
	unsigned short localPort = covise::coCoviseConfig::getInt("COVER.Plugin.Skateboard.localPort", 31320);
        float x = covise::coCoviseConfig::getFloat("x","COVER.Plugin.Skateboard.Position", 0);
        float y = covise::coCoviseConfig::getFloat("y","COVER.Plugin.Skateboard.Position", 0);
        float z = covise::coCoviseConfig::getFloat("z","COVER.Plugin.Skateboard.Position", floorHeight);
        float h = covise::coCoviseConfig::getFloat("h","COVER.Plugin.Skateboard.Position", 0);
        float p = covise::coCoviseConfig::getFloat("p","COVER.Plugin.Skateboard.Position", 0);
        float r = covise::coCoviseConfig::getFloat("r","COVER.Plugin.Skateboard.Position", 0);

        MAKE_EULER_MAT(SkateboardPos, h,p,r);
        SkateboardPos.postMultTranslate(osg::Vec3(x,y,z));
        
	std::cerr << "Skateboard config: UDP: serverHost: " << host << ", localPort: " << localPort << ", serverPort: " << serverPort << std::endl;
        bool ret = false;

        if (coVRMSController::instance()->isMaster())
        {
            std::string host = "";
            for (const auto& i : opencover::Input::instance()->discovery()->getDevices())
            {
                if (i->pluginName == "Skateboard")
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
                        std::cerr << "Skateboard: falided to open local UDP port" << localPort << std::endl;
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
        return ret;
        
}

void
Skateboard::run()
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
Skateboard::stop()
{
	doStop = true;
}


void Skateboard::syncData()
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

bool Skateboard::update()
{
    if(isEnabled())
    {
        if (coVRMSController::instance()->isMaster())
        {
	       double dT = cover->frameDuration();
	       float v;
	       osg::Vec3d normal = getNormal();
	       //fprintf(stderr, "normal(%f %f) %d\n", normal.x(), normal.y(), getButton());
	       if (getButton() == 2)
	       {
		   speed += 0.01;
	       }
	       TransformMat = VRSceneGraph::instance()->getTransform()->getMatrix();

               float a = getYAccelaration();
               speed += a * dT;
               speed -= 0.00003 * speed * speed * dT;

               const int normalYShifted = normal.y() + 1;
               if (getWeight() >= 10 && speed < 5.0f){
                   if(normalYShifted < 0)
	               speed += -0.03*normalYShifted;
               }

               const int maxSpeed = 10;
	       if(speed > maxSpeed)
	       {
		   speed = maxSpeed;
	       }
	       if(speed < -maxSpeed)
	       {
		   speed = -maxSpeed;
	       }
	       if (getWeight() < 10)
	       {
                   //speed -= a * dT;
		   speed += (speed > 0 ? -1 : 1) * 0.1;
                   if(fabs(speed) < 0.2)
                       speed = 0;
	       }
	       v = speed;
               //fprintf(stderr,"a: %f\n", a);
	    //   fprintf(stderr,"v: %f \n",v );
	       float s = v * dT;
	       osg::Vec3 V(0, s, 0);
	       float rotAngle = 0.0;
	       if ((s < 0.0001 && s > -0.0001)) // straight
	       {
		   //fprintf(stderr,"bikeTrans: %f %f %f\n",bikeTrans(3, 0), bikeTrans(3, 1), bikeTrans(3, 2) );
		   //fprintf(stderr,"V: %f %f %f\n",V[0], V[1], V[2] );
	       }
	       else if (getWeight() >= 10)
	       {
		   float wheelAngle = normal.x()/-4.0;
		   float r = tan(M_PI_2 - wheelAngle * 0.2 / (((fabs(v) * 0.2) + 1))) * wheelBase;
		   float u = 2.0 * r * M_PI;
		   rotAngle = (s / u) * 2.0 * M_PI;
		   V[1] = r * sin(rotAngle);
		   V[0] = (r - r * cos(rotAngle));
	       }


	       osg::Matrix relTrans;
	       osg::Matrix relRot;
	       relRot.makeRotate(-rotAngle, 0, 0, 1);
	       relTrans.makeTranslate(V*-1000); // m to mm (move world in the opposite direction
	       
               auto mat = getBoardMatrix();

	       TransformMat = mat * relTrans * relRot;
       
       
            coVRMSController::instance()->sendSlaves((char *)TransformMat.ptr(), sizeof(TransformMat));
        }
        else
        {
            coVRMSController::instance()->readMaster((char *)TransformMat.ptr(), sizeof(TransformMat));
        }
        VRSceneGraph::instance()->getTransform()->setMatrix(TransformMat);
                coVRCollaboration::instance()->SyncXform();
       
    }
    return false;
}

osgUtil::LineSegmentIntersector::Intersection getFirstIntersection(osg::ref_ptr<osg::LineSegment> ray, bool* haveISect){

    //  just adjust height here

    osg::ref_ptr<osgUtil::IntersectorGroup> igroup = new osgUtil::IntersectorGroup;
    osg::ref_ptr<osgUtil::LineSegmentIntersector> intersector;
    intersector = coIntersection::instance()->newIntersector(ray->start(), ray->end());
    igroup->addIntersector(intersector);

    osgUtil::IntersectionVisitor visitor(igroup);
    visitor.setTraversalMask(Isect::Walk);
    VRSceneGraph::instance()->getTransform()->accept(visitor);

    *haveISect = intersector->containsIntersections();
    if(!*haveISect){
        return {};
    }

    return intersector->getFirstIntersection();
}

osg::Matrix Skateboard::getBoardMatrix(){
    float wheelDis = wheelBase*1000.0;
    osg::Vec3 pos = SkateboardPos.getTrans();
    osg::Vec3d y{SkateboardPos(1, 0), SkateboardPos(1, 1), SkateboardPos(1, 2)};
    osg::Vec3 rearPos = pos + y * -wheelDis;


    osg::ref_ptr<osg::LineSegment> rayFront;
    {
	    // down segment
	    osg::Vec3 p0, q0;
	    p0.set(pos[0], pos[1], pos[2] + stepSizeUp);
	    q0.set(pos[0], pos[1], pos[2] - stepSizeDown);

	    rayFront = new osg::LineSegment(p0, q0);
    }
    osg::ref_ptr<osg::LineSegment> rayBack;
	{
	    // down segment
	    osg::Vec3 p0, q0;
            
	    p0.set(rearPos[0], rearPos[1], rearPos[2] + stepSizeUp);
	    q0.set(rearPos[0], rearPos[1], rearPos[2] - stepSizeDown);

	    rayBack = new osg::LineSegment(p0, q0);
	}
    bool intersects;
    auto front = getFirstIntersection(rayFront, &intersects);
    if(!intersects){
        return TransformMat;
    }
    auto back = getFirstIntersection(rayBack, &intersects);
    if(!intersects){
        return TransformMat;
    }

    auto frontNormal = front.getWorldIntersectNormal();
    frontNormal.normalize();
    auto backNormal = back.getWorldIntersectNormal();
    backNormal.normalize();
    //fprintf(stderr,"f %f \n",frontNormal*osg::Vec3(0,0,1));
    //fprintf(stderr,"b %f \n",backNormal*osg::Vec3(0,0,1) );
    if(frontNormal*osg::Vec3(0,0,1) < 0.2)
        return TransformMat;
    if(backNormal*osg::Vec3(0,0,1) < 0.2)
        return TransformMat;

    osg::Vec3d newY = front.getWorldIntersectPoint() - back.getWorldIntersectPoint();
    newY.normalize();
    osg::Vec3d newX = newY ^ frontNormal;
    newX.normalize();
    osg::Vec3d newZ = newX ^ newY;
    newZ.normalize();

    osg::Vec3d translation = front.getWorldIntersectPoint();

    osg::Matrix newMatrix;

    newMatrix(0,0) = newX.x();
    newMatrix(0,1) = newX.y();
    newMatrix(0,2) = newX.z();

    newMatrix(1,0) = newY.x();
    newMatrix(1,1) = newY.y();
    newMatrix(1,2) = newY.z();

    newMatrix(2,0) = newZ.x();
    newMatrix(2,1) = newZ.y();
    newMatrix(2,2) = newZ.z();
    
    newMatrix = newMatrix * osg::Matrix::translate(translation);
    
    osg::Matrix Nn = newMatrix;
    osg::Matrix invNn;
    invNn.invert(Nn);

    osg::Matrix NewTransform = TransformMat * invNn * SkateboardPos;
    osg::Vec3d z{NewTransform(2,0), NewTransform(2, 1), NewTransform(2, 2)};
    //fprintf(stderr,"z %f \n",z*osg::Vec3(0,0,1));
    if(z*osg::Vec3(0,0,1) < 0.2)
        return TransformMat;
     
    return  NewTransform;
}


float Skateboard::getYAccelaration()
{
    osg::Vec3d x{TransformMat(0, 0), TransformMat(0, 1), TransformMat(0, 2)};
    x.normalize();
    osg::Vec3d y{TransformMat(1, 0), TransformMat(1, 1), TransformMat(1, 2)};
    y.normalize();
    osg::Vec3d z_yz{0, TransformMat(2, 1), TransformMat(2, 2)};
    z_yz.normalize();

    float cangle = 1.0 - z_yz * osg::Vec3(0, 0, 1);
    if(z_yz[1]>0  )
        cangle *= -1;
    //fprintf(stderr,"z_yz %f x0 %f sprod: %f\n",z_yz[1],x[0],cangle);
    float a = cangle * 9.81;
    return a;
}

void Skateboard::setEnabled(bool flag)
{
    coVRNavigationProvider::setEnabled(flag);
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
    //WakeUp Skateboard
    Initialize();

}
void Skateboard::updateThread()
{
	if (udp)
	{
		char tmpBuf[10000];
		int status = udp->receive(&tmpBuf, 10000);


		if (status > 0 && status >= sizeof(SBData))
		{

			{
				OpenThreads::ScopedLock<OpenThreads::Mutex> lock(mutex);
				memcpy(&sbData, tmpBuf, sizeof(SBData));
                fprintf(stderr, "%lf %lf %lf %lf\n", sbData.fl, sbData.fr, sbData.rl, sbData.rr);
			}

		}
		else if (status == -1)
		{
            if(isEnabled()) // otherwise we are not supposed to receive anything
            { 
			    std::cerr << "Skateboard::update: error while reading data" << std::endl;
                if(udp) // try to wake up the skateboard (if the first start UDP message was lost)
                udp->send("start");
            }
			return;
		}
		else
		{
			std::cerr << "Skateboard::update: received invalid no. of bytes: recv=" << status << ", got=" << status << std::endl;
			return;
		}

	}
}
void Skateboard::Initialize()
{
        if (coVRMSController::instance()->isMaster())
        {
	    sbControl.cmd = 1;

	    ret = udp->send(&sbControl, sizeof(SBCtrlData));
        }
}

osg::Vec3d Skateboard::getNormal()
{
	OpenThreads::ScopedLock<OpenThreads::Mutex> lock(mutex);
	float w = (sbData.fl + sbData.fr + sbData.rl + sbData.rr) / 4.0;
	float fs = sbData.fl + sbData.fr;
	float rs = sbData.rl + sbData.rr;
	float sl = sbData.fl + sbData.rl;
	float sr = sbData.fr + sbData.rr;

	return osg::Vec3d((sl - sr) / w, (fs - rs) / w, 1.0);
}

float Skateboard::getWeight() // weight in Kg
{
	OpenThreads::ScopedLock<OpenThreads::Mutex> lock(mutex);
	return float((sbData.fl + sbData.fr + sbData.rl + sbData.rr) / 4.0);
}

unsigned char Skateboard::getButton()
{
	return sbData.button;
}
COVERPLUGIN(Skateboard)

