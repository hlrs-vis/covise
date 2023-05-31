/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */


#include "TacxPlugin.h"
#include <config/CoviseConfig.h>
#include <util/unixcompat.h>
#include <cover/coVRMSController.h>
#include <cover/VRSceneGraph.h>
#include <osg/LineSegment>
#include <osgUtil/IntersectionVisitor>
#include <osgUtil/LineSegmentIntersector>
#include "cover/coIntersection.h"
#include <OpenVRUI/osg/mathUtils.h> //for MAKE_EULER_MAT
using namespace opencover;

static float zeroAngle = 1152.;


TacxPlugin::TacxPlugin()
: coVRPlugin(COVER_PLUGIN_NAME)
, coVRNavigationProvider("Tacx",this)
{
        stepSizeUp=2000;
        stepSizeDown=2000;
        coVRNavigationManager::instance()->registerNavigationProvider(this);
}
TacxPlugin::~TacxPlugin()
{
        delete tacx;
        coVRNavigationManager::instance()->unregisterNavigationProvider(this);
}

bool TacxPlugin::init()
{
        delete tacx;
        float floorHeight = VRSceneGraph::instance()->floorHeight();

        float x = covise::coCoviseConfig::getFloat("x","COVER.Plugin.Tacx.Position", 0);
        float y = covise::coCoviseConfig::getFloat("y","COVER.Plugin.Tacx.Position", 0);
        float z = covise::coCoviseConfig::getFloat("z","COVER.Plugin.Tacx.Position", floorHeight);
        float h = covise::coCoviseConfig::getFloat("h","COVER.Plugin.Tacx.Position", 0);
        float p = covise::coCoviseConfig::getFloat("p","COVER.Plugin.Tacx.Position", 0);
        float r = covise::coCoviseConfig::getFloat("r","COVER.Plugin.Tacx.Position", 0);

        MAKE_EULER_MAT(TacxPos, h,p,r);
        TacxPos.postMultTranslate(osg::Vec3(x,y,z));
        tacx=nullptr;
	if(coVRMSController::instance()->isMaster())
	{
	    tacx=new Tacx();
	}
        else
        {
        }
        return true;
        
}

bool TacxPlugin::update()
{
    if(isEnabled())
    {
        if (coVRMSController::instance()->isMaster() )
        {
            if(tacx!=nullptr)
            {
               tacx->update();
	       double dT = cover->frameDuration();

	       TransformMat = VRSceneGraph::instance()->getTransform()->getMatrix();

               float a = getYAccelaration();
               if(a >0)
                   a=0;
               tacx->setForce(-a/2.0);
               fprintf(stderr,"force: %f\n",-a/2.0);
               speed=tacx->getSpeed();
	       float s = speed * dT;
	       osg::Vec3 V(0, s, 0);
	       float rotAngle = 0.0;
	       if ((s < 0.0001 && s > -0.0001)) // straight
	       {
		   //fprintf(stderr,"bikeTrans: %f %f %f\n",bikeTrans(3, 0), bikeTrans(3, 1), bikeTrans(3, 2) );
		   //fprintf(stderr,"V: %f %f %f\n",V[0], V[1], V[2] );
	       }
		   float wheelAngle = tacx->getAngle();
		   float r = tan(M_PI_2 - wheelAngle * 0.2 / (((fabs(speed) * 0.2) + 1))) * wheelBase;
		   float u = 2.0 * r * M_PI;
		   rotAngle = (s / u) * 2.0 * M_PI;
		   V[1] = r * sin(rotAngle);
		   V[0] = (r - r * cos(rotAngle));


	       osg::Matrix relTrans;
	       osg::Matrix relRot;
	       relRot.makeRotate(-rotAngle, 0, 0, 1);
	       relTrans.makeTranslate(V*-1000); // m to mm (move world in the opposite direction
	       
               auto mat = getMatrix();

	       TransformMat = mat * relTrans * relRot;
            }
       
            coVRMSController::instance()->sendSlaves((char *)TransformMat.ptr(), sizeof(TransformMat));
        }
        else
        {
            coVRMSController::instance()->readMaster((char *)TransformMat.ptr(), sizeof(TransformMat));
        }
        VRSceneGraph::instance()->getTransform()->setMatrix(TransformMat);
       
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

osg::Matrix TacxPlugin::getMatrix(){
    float wheelDis = wheelBase*1000;
    osg::Vec3 pos = TacxPos.getTrans();
    osg::Vec3d y{TacxPos(1, 0), TacxPos(1, 1), TacxPos(1, 2)};
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

    osg::Matrix NewTransform = TransformMat * invNn * TacxPos;
    osg::Vec3d z{NewTransform(2,0), NewTransform(2, 1), NewTransform(2, 2)};
    //fprintf(stderr,"z %f \n",z*osg::Vec3(0,0,1));

   /* osg::Matrix InvTacxPos;
    InvTacxPos.invert(TacxPos);
    NewTransform = NewTransform * InvTacxPos;

    newX[0] = NewTransform(0,0);
    newX[1] = NewTransform(0,1);
    newX[2] = NewTransform(0,2);
    newY[0] = NewTransform(1,0);
    newY[1] = NewTransform(1,1);
    newY[2] = NewTransform(1,2);
    newZ[0] = NewTransform(2,0);
    newZ[1] = NewTransform(2,1);
    newZ[2] = NewTransform(2,2);
    newZ[0]=0;
    newZ.normalize();
    newX = newY ^ newZ;
    newX.normalize();

    NewTransform(0,0) = newX.x();
    NewTransform(0,1) = newX.y();
    NewTransform(0,2) = newX.z();
    NewTransform(2,0) = newZ.x();
    NewTransform(2,1) = newZ.y();
    NewTransform(2,2) = newZ.z();
    NewTransform = NewTransform * TacxPos;*/
    if(z*osg::Vec3(0,0,1) < 0.2)
        return TransformMat;
     
    return  NewTransform;
}


float TacxPlugin::getYAccelaration()
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

void TacxPlugin::setEnabled(bool flag)
{
    coVRNavigationProvider::setEnabled(flag);

}

COVERPLUGIN(TacxPlugin)

