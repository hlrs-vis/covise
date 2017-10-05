#include "CameraSensor.h"

#include "cover/coVRPluginSupport.h"



CameraSensor::CameraSensor(Entity *e, OpenScenario::oscVehicle *v, osg::Matrix pos, double FOV)
{
	cameraPosition=pos;
	FoV=FOV;
	myEntity=e;
	myVehicle=v;
}

CameraSensor::~CameraSensor()
{
}

void CameraSensor::updateView()
{
	
	osg::Matrix vehicleMat = myEntity->entityGeometry->getCarGeometry()->getVehicleTransformMatrix();

	osg::Matrix scMat = opencover::cover->getObjectsScale()->getMatrix();
	osg::Matrix iscMat;
	iscMat.invert(scMat);
	osg::Matrix trans = iscMat  * cameraPosition *  vehicleMat* scMat;

	osg::Vec3 p;
	p = trans.getTrans();
	fprintf(stderr, "pos %f %f %f\n", (float)p[0], (float)p[1], (float)p[2]);
	osg::Matrix itrans;
	itrans.invert(trans);


	
	//osg::Matrix viewerTrans;
	//viewerTrans.makeTranslate(cover->getViewerMat().getTrans());
	//mat.postMult(viewerTrans);
	opencover::cover->setXformMat(itrans);
}
