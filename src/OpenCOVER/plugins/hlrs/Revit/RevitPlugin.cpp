/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

 /****************************************************************************\
 **                                                            (C)2009 HLRS  **
 **                                                                          **
 ** Description: Revit Plugin (connection to Autodesk Revit Architecture)    **
 **                                                                          **
 **                                                                          **
 ** Author: U.Woessner		                                                 **
 **                                                                          **
 ** History:  								                                 **
 ** Mar-09  v1	    				       		                             **
 **                                                                          **
 **                                                                          **
 \****************************************************************************/
#define QT_NO_EMIT

#include <algorithm>

#include "RevitPlugin.h"
#include <cover/coVRPluginSupport.h>
#include <cover/coVRFileManager.h>
#include <cover/RenderObject.h>
#include <cover/coVRMSController.h>
#include <cover/coVRConfig.h>
#include <cover/coVRSelectionManager.h>
#include <cover/coVRTui.h>
#include <cover/coVRShader.h>
#include <cover/OpenCOVER.h>
#include <cover/VRViewer.h>
#include <cover/ui/EditField.h>

#include <OpenVRUI/coInteractionManager.h>
#include <OpenVRUI/coCheckboxMenuItem.h>
#include <OpenVRUI/coButtonMenuItem.h>
#include <OpenVRUI/coSubMenuItem.h>
#include <OpenVRUI/coRowMenu.h>
#include <OpenVRUI/coCheckboxGroup.h>
#include <OpenVRUI/coButtonMenuItem.h>
#include <OpenVRUI/osg/OSGVruiUserDataCollection.h>
#include <OpenVRUI/osg/mathUtils.h>

#include <PluginUtil/PluginMessageTypes.h>

#include <vrml97/vrml/VrmlNamespace.h>


#include <osg/Geode>
#include <osg/Switch>
#include <osg/Geometry>
#include <osg/PrimitiveSet>
#include <osg/Array>
#include <osg/CullFace>
#include <osg/MatrixTransform>

#include <osgDB/ReadFile>
#include "GenNormals.h"

#include <net/covise_host.h>
#include <net/covise_socket.h>
#include <net/tokenbuffer.h>
#include <config/CoviseConfig.h>
#include <util/unixcompat.h>

#include "VrmlNodePhases.h"

using covise::TokenBuffer;
using covise::coCoviseConfig;

int ElementInfo::yPos = 3;
IKAxisInfo::IKAxisInfo()
{
	transform = nullptr;
}

//void IKAxisInfo::initIK(RigidBodyDynamics::Model* model, unsigned int parent_id)
/*	body = Body(1., Vector3d(origin.x() / 2.0, origin.y() / 2.0, origin.z() / 2.0), Vector3d(0.1, 0.1, 0.1));
	joint = Joint(RigidBodyDynamics::JointTypeRevolute, Vector3d(direction.x(), direction.y(), direction.z()));
	body_id = model->AddBody(parent_id, Xtrans(Vector3d(origin.x(), origin.y(), origin.z())), joint, body);*/
void IKAxisInfo::initIK(unsigned int myID)
{

	rotTransform = new osg::MatrixTransform();
	scaleTransform = nullptr;
}

IKInfo::IKInfo()
{
	robot = new  CRobot(Eigen::Vector3f(0.0f, 0.0f, 0.0f));
	

}

IKInfo::~IKInfo()
{
	delete iks;
}

void IKInfo::update()
{
	if (iks)
	{
		iks->update();
	}
}

void IKInfo::addHandle(osg::Node* n)
{
    iks = new IKSensor(RevitPlugin::instance(), this, n);
}

osg::Vec3 IKInfo::getPosition()
{
	osg::Matrix m;
	m.makeIdentity();
	for (unsigned int i = 0; i < axis.size()-1; ++i)
	{
		if (axis[i].transform)
		{
			m = axis[i].transform->getMatrix() * m;
		}
		m = axis[i].rotTransform->getMatrix() * m;
	}
	float z = m(3, 2);
	if (axis[axis.size() - 1].transform)
	{
		m = axis[axis.size() - 1].transform->getMatrix() * m;
	}
	m = axis[axis.size() - 1].rotTransform->getMatrix() * m;
	float x = m(3, 0), y = m(3, 1);
	return osg::Vec3(x, y, z);
}
osg::Vec3 IKInfo::getOrientation()
{
	osg::Matrix m;
	m.makeIdentity();
	for (unsigned int i = 0; i < axis.size(); ++i)
	{
		if (axis[i].transform)
		{
			m = axis[i].transform->getMatrix() * m;
		}
		m = axis[i].rotTransform->getMatrix() * m;
	}
	return osg::Vec3(m(1, 0), m(1, 1), m(1, 2));
}


void IKInfo::intiIK()
{
	unsigned int numAxis = axis.size();
	if (numAxis == 6)
	{
		numAxis = 4;
	}
	if (numAxis < 1)
	{
		return;
	}
	type = IKInfo::IKType::Rot;
	if (axis[0].type == IKAxisInfo::AxisType::Trans)
		type = IKInfo::IKType::Trans;
	else if (numAxis == 4 && axis[1].type == IKAxisInfo::AxisType::Trans)
		type = IKInfo::IKType::RotTrans;
	for (unsigned int i = 0; i < numAxis; ++i)
	{
		//Saving joint data to robot
		robot->jhandle.push_back(CJoint(0,0, REVOLUTE, "unknown")); // this is the 0 theta position, everything is relative to this
		//Saving link data  to robot
		robot->linkhadle.push_back(CLink(1/*a*/, 0/*alpha*/));
		//Calculating h_matrix and saving to robot
		Eigen::Matrix4f em;
		for (int n = 0; n < 4; n++)
			for (int m = 0; m < 4; m++)
				em(n, m) = axis[i].mat(m, n);

		robot->hmtx.push_back(em);
		robot->origHmtx.push_back(axis[i].mat);
	}
	robot->CaclulateFullTransormationMatrix();
	if (type == IKInfo::IKType::Rot)
	{
		osg::Vec3 xAxis(1, 0, 0);
		basePos = axis[0].origin;
		vA = axis[1].origin - basePos;
		vA.z() = 0;
		rA = vA.length();

		vB = axis[4].origin - axis[1].origin;
		vB.z() = 0;
		rB = vB.length();
		osg::Vec3 vAn = vA;
		vAn.normalize();
		osg::Vec3 vBn = vB;
		vBn.normalize();

		osg::Vec3 rotA = axis[0].direction;
		rotA.normalize();
		osg::Vec3 rotB = axis[1].direction;
		rotA.normalize();
		osg::Vec3 rotC = axis[2].direction;
		rotC.normalize();

		initialAngleA = getAngle(vAn, xAxis, rotA);
		initialAngleB = getAngle(vBn, vAn, rotB);

		vC = axis[3].origin - axis[2].origin;
		rC = vC.length();
		float dH = vC.z();

		initialAngleC = asin(dH / rC);
	}
	else if (type == IKInfo::IKType::RotTrans)
	{
		osg::Vec3 xAxis(1, 0, 0);
		osg::Vec3 rotA = axis[0].direction;
		rotA.normalize();
		basePos = axis[0].origin;
		vA = axis[2].origin - basePos;
		osg::Vec3 vAn = vA;
		vAn.normalize();
		initialAngleA = getAngle(vAn, xAxis, rotA);
		osg::Vec3 tmpVec = axis[2].origin - axis[0].origin;
		tmpVec[2] = 0;
		initialLength = tmpVec.length();
	}
	else if (type == IKInfo::IKType::Trans)
	{
	}


}


void IKInfo::updateIK(const osg::Vec3& targetPos, const osg::Vec3& targetDir)
{
	if (type == IKInfo::IKType::Rot)
	{
		vC = targetPos - axis[2].origin;
		float dH = vC.z();
		if (dH < -(rC - 0.10))
			dH = -(rC - 0.10);
		if (dH > 1)
			dH = 1;

		float currentAngleC = asin(dH / rC);
		rB = rC * cos(currentAngleC);

		axis[2].rotTransform->setMatrix(osg::Matrix::rotate(currentAngleC - initialAngleC, osg::Vec3(0, 0, -1)));
		axis[3].rotTransform->setMatrix(osg::Matrix::rotate(currentAngleC - initialAngleC, osg::Vec3(0, 0, -1)));

		osg::Vec3 toNew = targetPos - basePos;
		toNew.z() = 0;
		float newD = toNew.length();
		if (newD > 0.1 && newD < rA + rB)
		{

			osg::Vec3 xAxis(1, 0, 0);
			osg::Vec3 baseP = basePos;
			baseP.z() = 0;
			osg::Vec3 targetP = targetPos;
			targetP.z() = 0;
			float x = (rA * rA + newD * newD - rB * rB) / (2 * newD);
			float y = 0;
			float sqd = rA * rA - x * x;
			if (sqd > 0)
			{
				y = sqrt(sqd);
			}
			float qx = (x * ((targetP.x() - baseP.x()) / newD)) - (y * ((targetP.y() - baseP.y()) / newD));
			float qy = (x * ((targetP.y() - baseP.y()) / newD)) + (y * ((targetP.x() - baseP.x()) / newD));
			osg::Vec3 vQ(qx, qy, 0);
			osg::Vec3 tToB = baseP - targetP;
			osg::Vec3 dirA = vQ;
			osg::Vec3 dirB = (tToB + vQ) * -1;
			dirA.normalize();
			dirB.normalize();

			osg::Vec3 rotA = axis[0].direction;
			rotA.normalize();
			osg::Vec3 rotB = axis[1].direction;
			rotB.normalize();

			float newAngleA = getAngle(dirA, xAxis, rotA);
			float newAngleB = getAngle(dirB, dirA, rotB);
			if (std::isnan(newAngleA) || std::isnan(newAngleB))
			{
				fprintf(stderr, "isnan\n");
			}
			else
			{
				axis[0].rotTransform->setMatrix(osg::Matrix::rotate(newAngleA - initialAngleA, osg::Vec3(0, 0, -1)));
				axis[1].rotTransform->setMatrix(osg::Matrix::rotate(newAngleB - initialAngleB, osg::Vec3(0, 0, -1)));
			}

		} // else  zu weit auseinander

		// rotate end so that y axis points towards the specified y direction
		osg::Matrix m;
		m.makeIdentity();
		for (unsigned int i = 0; i < axis.size(); ++i)
		{
			if (axis[i].transform)
			{
				m = axis[i].transform->getMatrix() * m;
			}
			m = axis[i].rotTransform->getMatrix() * m;
		}
		auto td = targetDir;
		td[2] = 0;
		td.normalize();
		osg::Vec3 my(m(1, 0), m(1, 1), m(1, 2));
		my[2] = 0;
		my.normalize();
		float newAngle = getAngle(my, td, axis[4].direction);
		osg::Matrix oldMat = axis[4].rotTransform->getMatrix();
		axis[4].rotTransform->setMatrix(oldMat * osg::Matrix::rotate(newAngle, osg::Vec3(0, 0, 1)));

	}
	else if (type == IKInfo::IKType::RotTrans)
	{

		osg::Vec3 xAxis(1, 0, 0);
		vC = targetPos - getPosition();
		float dH = -vC.z();
		osg::Matrix oldMat = axis[2].rotTransform->getMatrix();
		axis[2].rotTransform->setMatrix(oldMat * osg::Matrix::translate(0, 0, dH));
		//scale
		float length = axis[2].rotTransform->getMatrix().getTrans().z();
		/*if (length < axis[2].min)
		{
			axis[2].rotTransform->setMatrix(oldMat);
			length = axis[2].min;
		}*/
		float scale = 1 + (length / axis[3].transform->getMatrix().getTrans().z());
		if (axis[2].scaleTransform)
			axis[2].scaleTransform->setMatrix(osg::Matrix::scale(1, 1, scale));

		// translation Axis 1
		osg::Matrix m;
		m.makeIdentity();
		if (axis[0].transform)
		{
			m = axis[0].transform->getMatrix() * m;
		}
		m = axis[0].rotTransform->getMatrix() * m;
		if (axis[1].transform)
		{
			m = axis[1].transform->getMatrix() * m;
		}
		osg::Matrix invM;
		invM.invert(m);
		osg::Vec3 toPos = targetPos * invM;
		axis[1].rotTransform->setMatrix(osg::Matrix::translate(0, 0, toPos[2] - axis[2].transform->getMatrix().getTrans().z())); // -initial z position

		toPos = targetPos - axis[0].origin;
		toPos[2] = 0;
		//axis[1].rotTransform->setMatrix(oldMat * osg::Matrix::translate(0, 0, toPos.length() - initialLength));

		toPos.normalize();
		osg::Vec3 axis0 = axis[0].direction;
		axis0.normalize();
		float newAngleA = getAngle(toPos, xAxis, axis0);

		axis[0].rotTransform->setMatrix(osg::Matrix::rotate(newAngleA - initialAngleA, osg::Vec3(0, 0, -1)));

		// rotate end so that y axis points towards the specified y direction
		m.makeIdentity();
		for (unsigned int i = 0; i < axis.size(); ++i)
		{
			if (axis[i].transform)
			{
				m = axis[i].transform->getMatrix() * m;
			}
			m = axis[i].rotTransform->getMatrix() * m;
		}
		auto td = targetDir;
		td[2] = 0;
		td.normalize();
		osg::Vec3 my(m(1, 0), m(1, 1), m(1, 2));
		my[2] = 0;
		my.normalize();
		float newAngle = getAngle(my, td, axis[3].direction);
		oldMat = axis[3].rotTransform->getMatrix();
		axis[3].rotTransform->setMatrix(oldMat * osg::Matrix::rotate(newAngle, osg::Vec3(0, 0, 1)));
	}
	else if (type == IKInfo::IKType::Trans)
	{
		osg::Vec3 xAxis(1, 0, 0);
		vC = targetPos - getPosition();
		float dH = -vC.z();
		osg::Matrix oldMat = axis[2].rotTransform->getMatrix();
		axis[2].rotTransform->setMatrix(oldMat * osg::Matrix::translate(0, 0, dH));
		//scale
		float length = axis[2].rotTransform->getMatrix().getTrans().z();
		/*if (length < axis[2].min)
		{
			axis[2].rotTransform->setMatrix(oldMat);
			length = axis[2].min;
		}*/
		float scale = 1+(length / axis[3].transform->getMatrix().getTrans().z());
		if (axis[2].scaleTransform)
			axis[2].scaleTransform->setMatrix(osg::Matrix::scale(1,1,scale));
		// translation Axis 1
		osg::Matrix m;
		m.makeIdentity();
		if (axis[0].transform)
		{
			m = axis[0].transform->getMatrix() * m;
		}
		osg::Matrix invM;
		invM.invert(m);
		osg::Vec3 toPos = targetPos * invM ;
		axis[0].rotTransform->setMatrix(osg::Matrix::translate(0, 0, toPos[2] - axis[1].transform->getMatrix().getTrans().z())); // -initial z position
		m = axis[0].rotTransform->getMatrix() * m;
		if (axis[1].transform)
		{
			m = axis[1].transform->getMatrix() * m;
		}
		invM.invert(m);
		toPos = targetPos * invM;
		axis[1].rotTransform->setMatrix(osg::Matrix::translate(0, 0, toPos[2] - axis[2].transform->getMatrix().getTrans().z()));// -initial z position
		// translation Axis 2

		// rotate end so that y axis points towards the specified y direction
		m.makeIdentity();
		for (unsigned int i = 0; i < axis.size(); ++i)
		{
			if (axis[i].transform)
			{
				m = axis[i].transform->getMatrix() * m;
			}
			m = axis[i].rotTransform->getMatrix() * m;
		}
		auto td = targetDir;
		td[2] = 0;
		td.normalize();
		osg::Vec3 my(m(1, 0), m(1, 1), m(1, 2));
		my[2] = 0;
		my.normalize();
		float newAngle = getAngle(my, td, axis[3].direction);
		oldMat = axis[3].rotTransform->getMatrix();
		axis[3].rotTransform->setMatrix(oldMat * osg::Matrix::rotate(newAngle, osg::Vec3(0, 0, 1)));
	}
}


IKSensor::IKSensor(RevitPlugin *r,IKInfo* i, osg::Node* n)
	: coPickSensor(n)
{
	myIKI = i;
	revitPlugin = r;
}

IKSensor::~IKSensor()
{

	if (interaction->isRegistered())
	{
		coInteractionManager::the()->unregisterInteraction(interaction);
	}

	RevitPlugin::instance()->unregisterInteraction(myIKI);
	if (active)
		disactivate();
}
void IKSensor::miss()
{
	scheduleUnregister = true;
	hitActive = false;
	//fprintf(stderr, "miss\n");
}
int IKSensor::hit(vruiHit* hit)
{
	int ret = coPickSensor::hit(hit);
	scheduleUnregister = false;
	RevitPlugin::instance()->registerInteraction(myIKI);
	//fprintf(stderr, "hit\n");
	return ret;
}
void IKSensor::update()
{
	if (scheduleUnregister && interaction->isRegistered()&& !interaction->isRunning() &&!RevitPlugin::instance()->isInteractionRunning())
	{
		scheduleUnregister = false;
		coInteractionManager::the()->unregisterInteraction(interaction);
		RevitPlugin::instance()->unregisterInteraction(myIKI);
	}
}
/*
* Called when the mouse is over the Annotation. No Click necessary!
*
*/
void IKSensor::activate()
{
	revitPlugin->registerInteraction(myIKI);
	active = 1;
	//myAnnotation->setIcon(1);
}

/*
* Called when the mouse is no more over the annotation
*
*/
void IKSensor::disactivate()
{
	revitPlugin->unregisterInteraction(myIKI);
	active = 0;
	//myAnnotation->setIcon(0);
}

bool RevitPlugin::isInteractionRunning()
{
	if (currentIKI)
	{
		return(currentIKI->iks->getInteraction()->isRunning());
	}
	return false;
}
void RevitPlugin::registerInteraction(IKInfo* i)
{
	if(!isInteractionRunning())
	currentIKI = i;
	//fprintf(stderr, "RevitPlugin::registerInteraction\n");
}
void RevitPlugin::unregisterInteraction(IKInfo* i)
{
	if (!isInteractionRunning())
	currentIKI = nullptr;
	//fprintf(stderr, "RevitPlugin::unregisterInteraction\n");
}


void RevitPlugin::updateIK()
{
	for (auto& iki : ikInfos)
	{

		//osg::Vec3 targetPos(RevitPlugin::instance()->xPos->number(), RevitPlugin::instance()->yPos->number(), RevitPlugin::instance()->zPos->number());
		//osg::Vec3 targetDir(RevitPlugin::instance()->xOri->number(), RevitPlugin::instance()->yOri->number(), RevitPlugin::instance()->zOri->number());


		//iki->updateIK(targetPos, targetDir);
	}
}

void RevitPlugin::key(int type, int keySym, int mod)
{
	if (type == osgGA::GUIEventAdapter::KEYDOWN && mod & osgGA::GUIEventAdapter::MODKEY_ALT)
	{
		if (keySym == 'p')
		{
			currentPhase++;
			if (currentPhase >= phaseInfos.size())
				currentPhase = phaseInfos.size() -1;
			for (const auto& pi : phaseInfos)
			{
				if (pi->ID == currentPhase)
				{
					pi->button->setState(true);
					setPhase(pi->PhaseName);
					break;
				}
			}
		}
		if (keySym == 'P')
		{
			currentPhase--;
			if (currentPhase < 0)
				currentPhase = 0;
			for (const auto& pi : phaseInfos)
			{
				if (pi->ID == currentPhase)
				{
					pi->button->setState(true);
					setPhase(pi->PhaseName);
					break;
				}
			}
	    }
	}
}

void IKInfo::updateGeometry()
{
	/*UpdateKinematicsCustom(*model, &Q, NULL, NULL);

	for (int i = 0; i < axis.size(); i++)
	{
		if (axis[i].rotTransform)
		{
			axis[i].rotTransform->setMatrix(osg::Matrix::rotate(Q[i], osg::Vec3(0, 0, 1)));
		}
	}*/
}

RevitDesignOption::RevitDesignOption(RevitDesignOptionSet *s)
{
	set = s;
	ID = -1;
}

RevitDesignOptionSet::RevitDesignOptionSet()
{
	ID = -1;
}
RevitDesignOptionSet::~RevitDesignOptionSet()
{
	delete designoptionsCombo;
}

void RevitDesignOptionSet::createSelectionList()
{
	delete designoptionsCombo;
	std::string ItemName = name;
	std::transform(ItemName.begin(), ItemName.end(), ItemName.begin(), [](char ch) {
		return ch == ' ' ? '_' : ch;
		});
	designoptionsCombo = new ui::SelectionList(RevitPlugin::instance()->revitMenu, ItemName +std::to_string(ID));
	designoptionsCombo->setText(name);

	std::vector<std::string> items;
	for (const auto& des : designOptions)
		items.push_back(des.name);
	designoptionsCombo->setList(items);
	designoptionsCombo->setCallback([this](int item)
		{
			int num = 0;
			for (const auto& des : designOptions)
			{
				if (item == num)
				{
					TokenBuffer tb;
					tb << des.ID;
					tb << DocumentID;
					Message m(tb);
					m.type = (int)RevitPlugin::MSG_SelectDesignOption;
					RevitPlugin::instance()->sendMessage(m);
					break;
				}
				num++;
			}
			
		});
}
static void matrix2array(const osg::Matrix &m, osg::Matrix::value_type *a)
{
	for (unsigned y = 0; y < 4; ++y)
		for (unsigned x = 0; x < 4; ++x)
		{
			a[y * 4 + x] = m(x, y);
		}
}

static void array2matrix(osg::Matrix &m, const osg::Matrix::value_type *a)
{
	for (unsigned y = 0; y < 4; ++y)
		for (unsigned x = 0; x < 4; ++x)
		{
			m(x, y) = a[y * 4 + x];
		}
}
ARMarkerInfo::ARMarkerInfo()
{
}

void ARMarkerInfo::setValues(int id, int docID, int mid, std::string& n, double an, double of, osg::Matrix& m, osg::Matrix& hm, int hID, double s, std::string mt)
{
	ID = id;
	DocumentID = docID;
	MarkerID = mid;
	name = n;
	angle = an;
	offset = of;
	mat = m;
	markerType = mt;
	invHost.invert(hm);
	invMarker.invert(m);
	MarkerToHost = mat * invHost;

	hostMat = hm;
	hostID = hID;
	size = s;
	osg::Matrix offsetMat = mat;
	if (markerType != "ObjectMarker")
	{
		offsetMat *= invMarker;
	}
	
	marker = MarkerTracking::instance()->getOrCreateMarker((markerType+std::to_string(MarkerID)), std::to_string(MarkerID), size, offsetMat, true, markerType == "ObjectMarker");

	fprintf(stderr, "ObjectMarkerPos in feet: %d %d   %f %f %f\n", this->MarkerID, ID, mat.getTrans().x() / (1000 * REVIT_FEET_TO_M), mat.getTrans().y() / (1000 * REVIT_FEET_TO_M), mat.getTrans().z() / (1000 * REVIT_FEET_TO_M));
}

void ARMarkerInfo::update()
{
	if (marker)
	{
		if (marker->isVisible())
		{
			if (!marker->isObjectMarker())
			{

				osg::Matrix mm = marker->getMarkerTrans();

				osg::Matrix leftCameraTrans = VRViewer::instance()->getViewerMat();
				if (coVRConfig::instance()->stereoState())
				{
					leftCameraTrans = osg::Matrix::translate(-(VRViewer::instance()->getSeparation() / 2.0), 0, 0) * VRViewer::instance()->getViewerMat();
				}
				osg::Matrix MarkerInWorld = mm * leftCameraTrans;
				osg::Matrix MarkerInLocalCoords = MarkerInWorld * cover->getInvBaseMat(); // unit is m
				osg::Vec3 trans = MarkerInLocalCoords.getTrans();
				trans *= REVIT_M_TO_FEET;
				MarkerInLocalCoords.setTrans(trans);
				MarkerInLocalCoords.orthoNormalize(MarkerInLocalCoords);

				//mm = mm * marker->OpenGLToOSGMatrix;

				
				//fprintf(stderr, "MarkerPos in feet: %d %d   %f %f %f\n", this->MarkerID, ID, MarkerInLocalCoords.getTrans().x() , MarkerInLocalCoords.getTrans().y(), MarkerInLocalCoords.getTrans().z() );

				trans = MarkerInLocalCoords.getTrans() - hostMat.getTrans();
				trans[2] = 0;
				if((cover->frameTime() - lastUpdate) > 1.0 && trans.length() > 0.2)
				{
					hostMat.setTrans(hostMat.getTrans() + trans);
					lastUpdate = cover->frameTime();
					TokenBuffer stb;
					stb << hostID;
					stb << DocumentID;
					stb << (double)trans.x();
					stb << (double)trans.y();
					stb << (double)trans.z();

					Message message(stb);
					message.type = (int)RevitPlugin::MSG_SetTransform;
					RevitPlugin::instance()->sendMessage(message);
					fprintf(stderr, "MarkerTrans: %d %d   %f %f %f\n", this->MarkerID,ID, trans.x(), trans.y(), trans.z());
				}
			}
		}
	}
}

RevitInfo::RevitInfo()
{
}
RevitInfo::~RevitInfo()
{
}

ElementInfo::ElementInfo()
{
	group = nullptr;
};

ElementInfo::~ElementInfo()
{
	for (std::list<RevitParameter *>::iterator it = parameters.begin();
		it != parameters.end(); it++)
	{
		delete *it;
	}
	delete group;
};
void ElementInfo::addParameter(RevitParameter *p)
{
	if (group == NULL)
	{
		group = new ui::Group(RevitPlugin::instance()->parameterMenu,name);
	}
	yPos++;
	p->createUI(group, parameters.size());
	parameters.push_back(p);
}

RevitParameter::~RevitParameter()
{
	delete uiLabel;
	delete uiElement;
}

void RevitParameter::createUI(ui::Group *group, int pos)
{
	uiLabel = new ui::Label(group,name);
	uiLabel->setText(name);
	uiElement = nullptr;
	switch (StorageType)
	{
	case RevitPlugin::Double:
	{
		ui::EditField *ef = new ui::EditField(group,name + "ef");
		ef->setValue(d);
		ef->setCallback([this,ef](std::string value)
			{
				TokenBuffer tb;
				tb << element->ID;
				tb << element->DocumentID;
				tb << ID;
				tb << std::stod(ef->value());
				Message m(tb);
				m.type = (int)RevitPlugin::MSG_SetParameter;
				RevitPlugin::instance()->sendMessage(m);
			});
		uiElement = ef;
	}
	break;
	case RevitPlugin::ElementId:
	{
		ui::EditField* ef = new ui::EditField(group, name + "eid");
		ef->setValue(i);
		ef->setCallback([this, ef](std::string value)
			{
				TokenBuffer tb;
				tb << element->ID;
				tb << element->DocumentID;
				tb << ID;
				tb << std::stoi(ef->value());
				Message m(tb);
				m.type = (int)RevitPlugin::MSG_SetParameter;
				RevitPlugin::instance()->sendMessage(m);
			});
		uiElement = ef;
	}
	break;
	case RevitPlugin::Integer:
	{
		ui::EditField* ef = new ui::EditField(group, name + "ei");
		ef->setValue(i);
		ef->setCallback([this, ef](std::string value)
			{
				TokenBuffer tb;
				tb << element->ID;
				tb << element->DocumentID;
				tb << ID;
				tb << std::stoi(ef->value());
				Message m(tb);
				m.type = (int)RevitPlugin::MSG_SetParameter;
				RevitPlugin::instance()->sendMessage(m);
			});
		uiElement = ef;
	}
	break;
	case RevitPlugin::String:
	{
		ui::EditField* ef = new ui::EditField(group, name + "e");
		ef->setValue(s);
		ef->setCallback([this, ef](std::string value)
			{
				TokenBuffer tb;
				tb << element->ID;
				tb << element->DocumentID;
				tb << ID;
				tb << ef->value();
				Message m(tb);
				m.type = (int)RevitPlugin::MSG_SetParameter;
				RevitPlugin::instance()->sendMessage(m);
			});
		uiElement = ef;
	}
	break;
	default:
	{
		ui::EditField* ef = new ui::EditField(group, name + "et");
		ef->setValue(s);
		ef->setCallback([this, ef](std::string value)
			{
				TokenBuffer tb;
				tb << element->ID;
				tb << element->DocumentID;
				tb << ID;
				tb << ef->value();
				Message m(tb);
				m.type = (int)RevitPlugin::MSG_SetParameter;
				RevitPlugin::instance()->sendMessage(m);
			});
		uiElement = ef;
	}
	break;
	}

}

RevitViewpointEntry::RevitViewpointEntry(osg::Vec3 pos, osg::Vec3 dir, osg::Vec3 up, RevitPlugin *plugin, std::string n, int id, int docID, osg::Group *myGroup)
	: menuEntry(NULL)
{
	myPlugin = plugin;
	name = n;
	entryNumber = plugin->maxEntryNumber++;
	eyePosition = pos;
	viewDirection = -dir;
	viewDirection.normalize();
	upDirection = up;
	upDirection.normalize();
	ID = id;
	documentID = docID;

	menuEntry = new ui::Button(plugin->viewpointMenu,"viewpoint"+std::to_string(entryNumber),plugin->viewpointGroup);
	menuEntry->setText(name);
	menuEntry->setState(false);
	menuEntry->setCallback([this](bool state) {if (state) activate(); });
	isActive = false;
	myTransform = dynamic_cast<osg::MatrixTransform*>(myGroup);

}

void RevitViewpointEntry::setValues(osg::Vec3 pos, osg::Vec3 dir, osg::Vec3 up, std::string n)
{
	name = n;
	eyePosition = pos;
	viewDirection = -dir;
	upDirection = up;
}

RevitViewpointEntry::~RevitViewpointEntry()
{
	delete menuEntry;
}

void RevitViewpointEntry::deactivate()
{
	menuEntry->setState(false);
	isActive = false;
}

void RevitViewpointEntry::setActive(bool a)
{
	menuEntry->setState(a);
}
void RevitViewpointEntry::activate()
{
	if(!menuEntry->state())
	    menuEntry->setState(true);
	isActive = true;
	osg::Matrix mat, rotMat;
	mat.makeTranslate(eyePosition[0], eyePosition[1], eyePosition[2]);
	osg::Vec3 xDir = viewDirection ^ upDirection;
	xDir.normalize();

	mat(0, 0) = xDir[0];
	mat(0, 1) = xDir[1];
	mat(0, 2) = xDir[2];
	mat(1, 0) = viewDirection[0];
	mat(1, 1) = viewDirection[1];
	mat(1, 2) = viewDirection[2];
	mat(2, 0) = upDirection[0];
	mat(2, 1) = upDirection[1];
	mat(2, 2) = upDirection[2];


	osg::Matrix ivMat;
	ivMat.invert(mat* myPlugin->RevitScale * myPlugin->NorthRotMat * myPlugin->RevitGeoRefference); // viewpoint positions are in the master project coordinate system, not in the local one.
	//ivMat.invert(mat*myTransform->getMatrix());
    ivMat =  ivMat * osg::Matrix::scale(REVIT_FEET_TO_M, REVIT_FEET_TO_M, REVIT_FEET_TO_M);

	osg::Matrix scMat;
	osg::Matrix iscMat;
	float scaleFactor = 1000;
	cover->setScale(scaleFactor);
	scMat.makeScale(scaleFactor, scaleFactor, scaleFactor);
	iscMat.makeScale(1.0 / scaleFactor, 1.0 / scaleFactor, 1.0 / scaleFactor);
	ivMat.postMult(scMat);
	ivMat.preMult(iscMat);
	osg::Matrix viewerTrans;
	viewerTrans.makeTranslate(cover->getViewerMat().getTrans());
	ivMat.postMult(viewerTrans);
	cover->setXformMat(ivMat);
}



ObjectParamater::ObjectParamater(TokenBuffer& tb,int i)
{
	num = i;
	int pID;
	tb >> pID;
	tb >> Name;
	tb >> StorageType;
	tb >> ParameterType;
	switch (StorageType)
	{
	case RevitPlugin::Double:
		tb >> d;
		break;
	case RevitPlugin::ElementId:
		tb >> ElementReferenceID;
		tb >> Value;
		break;
	case RevitPlugin::Integer:
		tb >> i;
		break;
	case RevitPlugin::String:
		tb >> Value;
		break;
	default:
		tb >> Value;
		break;
	}
}

ObjectParamater::~ObjectParamater()
{
	delete Label;
}

void ObjectParamater::createMenu()
{
	Label = new ui::Label(RevitPlugin::instance()->parametersMenu, std::to_string(num) + "_ValLabel");

	switch (StorageType)
	{
	case RevitPlugin::Double:
		Label->setText(Name +  " = " + std::to_string(d));
		break;
	case RevitPlugin::ElementId:
		Label->setText(Name +  " = " + Value);
		break;
	case RevitPlugin::Integer:
    {
        size_t pos = ParameterType.find("bool");
        if (pos != std::string::npos)
        {
            if (i > 0)
                Label->setText(Name + " = true");
            else
                Label->setText(Name + " = false");
        }
        else
        {
            Label->setText(Name + " = " + std::to_string(i));
        }
    }
		break;
	case RevitPlugin::String:
		Label->setText(Name +  " = " + Value);
		break;
	default:
		Label->setText(Name + " = " + Value);
		break;
	}
}

FamilyType::FamilyType(TokenBuffer& tb)
{
	tb >> Name;
	tb >> ID;
	tb >> FamilyName;
}
void FamilyType::createMenuEntry()
{

	selectType = new ui::Action(RevitPlugin::instance()->typesMenu, Name + FamilyName);
	selectType->setText(Name);
	selectType->setCallback([this]() {
		TokenBuffer stb;
		stb << RevitPlugin::instance()->currentObjectID;
		stb << RevitPlugin::instance()->currentDocumentID;
		stb << Name;
		stb << ID;

		Message message(stb);
		message.type = (int)RevitPlugin::MSG_SelectType;
		RevitPlugin::instance()->sendMessage(message);
		});
}
void FamilyType::createFamilyLabel()
{
	FamilyLabel = new ui::Label(RevitPlugin::instance()->typesMenu, FamilyName + "_Label");
	FamilyLabel->setText(FamilyName);
}

FamilyType::~FamilyType()
{
	delete selectType;
	delete FamilyLabel;
}
ObjectInfo::ObjectInfo(TokenBuffer& tb)
{
	int numTypes = 0;
	int numParams = 0;
	int objID = 0;
	int docID = 0;
	tb >> objID;
	tb >> docID;
	tb >> TypeName;
	tb >> TypeID;
	tb >> CategoryName;
	tb >> flipInfo;
	tb >> numTypes;

	TypeNameLabel = new ui::Label(RevitPlugin::instance()->objectInfoMenu, TypeName + "_TNLabel");
	TypeNameLabel->setText(CategoryName);
	for (int i = 0; i < numTypes; i++)
	{
		types.push_back(new FamilyType(tb));
	}
	tb >> numParams;
	for (int i = 0; i < numParams; i++)
	{
		parameters.push_back(new ObjectParamater(tb,i));
	}
	std::sort(types.begin(), types.end(),
		[](const FamilyType* LHS, const FamilyType* RHS)
		{ if (LHS->FamilyName == RHS->FamilyName) return (LHS->Name < RHS->Name); return (LHS->FamilyName < RHS->FamilyName); });
	std::string lastFamilyName;
	for (FamilyType*& ft : types)
	{
		if (lastFamilyName != ft->FamilyName)
		{
			lastFamilyName = ft->FamilyName;
			ft->createFamilyLabel();
		}
		ft->createMenuEntry();
	}
	std::sort(parameters.begin(), parameters.end(),
		[](const ObjectParamater* LHS, const ObjectParamater* RHS)
		{ return (LHS->Name < RHS->Name); });
	for (ObjectParamater*& pt : parameters)
	{
		pt->createMenu();
	}

	if (flipInfo & 1)
	{
		flipLR = new ui::Action(RevitPlugin::instance()->objectInfoMenu, "flipLR");
		flipLR->setText("flip hand");
		flipLR->setCallback([this]() {
			RevitPlugin::instance()->flip(0);
			});
	}
	if (flipInfo & 2)
	{
		flipIO = new ui::Action(RevitPlugin::instance()->objectInfoMenu, "flipIO");
		flipIO->setText("flip facing");
		flipIO->setCallback([this]() {
			RevitPlugin::instance()->flip(1);
			});
	}
}
void RevitPlugin::flip(int dir)
{

	TokenBuffer stb;
	stb << currentObjectID;
	stb << currentDocumentID;
	stb << dir;

	Message message(stb);
	message.type = (int)RevitPlugin::MSG_Flip;
	RevitPlugin::instance()->sendMessage(message);
}

ObjectInfo::~ObjectInfo()
{
	for (const auto& oi : types)
	{
		delete oi;
	}
	for (const auto& op : parameters)
	{
		delete op;
	}
	delete flipIO;
	delete flipLR;
	delete TypeNameLabel;
}

void RevitViewpointEntry::updateCamera()
{
	osg::Matrix m;
	std::string path;
	TokenBuffer stb;
	stb << ID;
	stb << documentID;

	osg::Matrix mat = cover->getXformMat();
	osg::Matrix viewerTrans;
	viewerTrans.makeTranslate(cover->getViewerMat().getTrans());
	osg::Matrix itransMat;
	itransMat.invert(viewerTrans);
	mat.postMult(itransMat);


	osg::Matrix scMat;
	osg::Matrix iscMat;
	float scaleFactor = cover->getScale();
	scMat.makeScale(scaleFactor, scaleFactor, scaleFactor);
	iscMat.makeScale(1.0 / scaleFactor, 1.0 / scaleFactor, 1.0 / scaleFactor);
	mat.postMult(iscMat);
	mat.preMult(scMat);

	osg::Matrix irotMat = mat;
	irotMat.setTrans(0, 0, 0);

	osg::Matrix rotMat;
	rotMat.invert(irotMat);
	mat.postMult(rotMat);
	osg::Vec3 eyePos = mat.getTrans();
	eyePosition[0] = -eyePos[0] / REVIT_FEET_TO_M;
	eyePosition[1] = -eyePos[1] / REVIT_FEET_TO_M;
	eyePosition[2] = -eyePos[2] / REVIT_FEET_TO_M;

	viewDirection[0] = rotMat(1, 0);
	viewDirection[1] = rotMat(1, 1);
	viewDirection[2] = rotMat(1, 2);
	upDirection[0] = rotMat(2, 0);
	upDirection[1] = rotMat(2, 1);
	upDirection[2] = rotMat(2, 2);

	stb << (double)eyePosition[0];
	stb << (double)eyePosition[1];
	stb << (double)eyePosition[2];
	stb << (double)viewDirection[0];
	stb << (double)viewDirection[1];
	stb << (double)viewDirection[2];
	stb << (double)upDirection[0];
	stb << (double)upDirection[1];
	stb << (double)upDirection[2];

	Message message(stb);
	message.type = (int)RevitPlugin::MSG_UpdateView;
	RevitPlugin::instance()->sendMessage(message);
}


/*
void RevitPlugin::tabletEvent(coTUIElement* tUIItem)
{
	if (tUIItem == addCameraTUIButton)
	{
	}
	else if (tUIItem == viewsCombo)
	{
		TokenBuffer tb;
		tb << viewsCombo->getSelectedEntry();
		Message message(tb);
		message.type = (int)RevitPlugin::MSG_SetView;
		RevitPlugin::instance()->sendMessage(message);
		message.type = (int)RevitPlugin::MSG_Resend;
		RevitPlugin::instance()->sendMessage(message);
	}
	else
	{
		for (const auto& it : viewpointEntries)
		{
			if (it->getTUIItem() == tUIItem)
			{
				it->activate();
				break;
			}
		}
	}
}*/

void RevitPlugin::createMenu()
{


	revitMenu = new ui::Menu("Revit", this);
	revitMenu->setText("Revit");
	/*xPos = new ui::EditField(revitMenu, "X");
	yPos = new ui::EditField(revitMenu, "Y");
	zPos = new ui::EditField(revitMenu, "Z");
	xPos->setCallback([this](const std::string &) {
		updateIK();
		});
	yPos->setCallback([this](const std::string&) {
		updateIK();
		});
	zPos->setCallback([this](const std::string&) {
		updateIK();
		});
	xOri = new ui::EditField(revitMenu, "H");
	yOri = new ui::EditField(revitMenu, "P");
	zOri = new ui::EditField(revitMenu, "R");
	xOri->setCallback([this](const std::string&) {
		updateIK();
		});
	yOri->setCallback([this](const std::string&) {
		updateIK();
		});
	zOri->setCallback([this](const std::string&) {
		updateIK();
		});*/
	viewpointMenu = new ui::Menu(revitMenu, "RevitViewpoints");
	viewpointMenu->setText("Viewpoints");
	parameterMenu = new ui::Menu(revitMenu, "RevitParameters");
	parameterMenu->setText("Parameters");
	phaseMenu = new ui::Menu(revitMenu, "RevitPhases");
	phaseMenu->setText("Phases");

	PhaseGroup = new ui::ButtonGroup(phaseMenu, "RevitPhasesGroup");

	objectInfoMenu = new ui::Menu(revitMenu, "ObjectInfo");
	objectInfoMenu->setText("ObjectInfo");

	objectInfoGroup = new ui::ButtonGroup(objectInfoMenu, "ObjectInfoGroup");


	typesMenu = new ui::Menu(objectInfoMenu, "Types");
	typesMenu->setText("Types");

	TypesGroup = new ui::ButtonGroup(typesMenu, "TypesGroup");

	parametersMenu = new ui::Menu(objectInfoMenu, "Parameters");
	parametersMenu->setText("Parameters");

	ParametersGroup = new ui::ButtonGroup(typesMenu, "ParametersGroup");

	selectObjectInteraction = new vrui::coCombinedButtonInteraction(vrui::coInteraction::ButtonAction, "selectObject", vrui::coInteraction::High);
	selectObject = new ui::Action(objectInfoMenu, "selectObject");
	selectObject->setText("select");
	selectObject->setCallback([this]() {
		coInteractionManager::the()->registerInteraction(selectObjectInteraction);
		});

	roomInfoMenu = new ui::Menu(revitMenu,"RoomInfo");
	roomInfoMenu->setText("Room Info");
	viewpointGroup = new ui::ButtonGroup(viewpointMenu, "revitViewpoints");
	viewsCombo = new ui::SelectionList(revitMenu, "views");

	toggleRoomLabels = new ui::Button(roomInfoMenu, "ShowLabels");
    toggleRoomLabels->setText("Show Labels");
	toggleRoomLabels->setCallback([this](bool state) {for (const auto& r : roomInfos) { if (state) r->label->show(); else r->label->hide(); }});
	toggleRoomLabels->setState(false);


	label1 = new ui::Label(roomInfoMenu,"RoomInfoLabel1");
	label1->setText("No Room");

	addCameraButton = new ui::Action(viewpointMenu,"AddCamera");
	addCameraButton->setText("Add Camera");
    addCameraButton->setCallback([this]() {
		//sendClipplaneModeToGui();
		});
	updateCameraButton = new ui::Action(viewpointMenu, "UpdateCamera");
	updateCameraButton->setText("Update Camera");
	updateCameraButton->setCallback([this]() {
		for (const auto& it : viewpointEntries)
		{
			if (it->isActive)
			{
				it->updateCamera();
			}
		}
		});

}


void RevitPlugin::destroyMenu()
{
	for (const auto& set : designOptionSets)
		delete set;
	designOptionSets.clear();
	delete revitMenu;
}

RevitPlugin::RevitPlugin() 
: coVRPlugin(COVER_PLUGIN_NAME)
, ui::Owner("RevitPlugin", cover->ui)
{
	fprintf(stderr, "RevitPlugin::RevitPlugin\n");

/*
	std::string str = "robot.xml";
	//Set origin at O_zero
	CRobot robot(Vector3f(0.0f, 0.0f, 0.0f));
	dh_table dht;
	dh_parametrs joint1;
	joint1.joint_name = "joint1";
	joint1.alpha = 0;  //Angle between zi and zi - 1 along xi
	joint1.d = 0;
	joint1.a = 200;    //Length of common normal
	joint1.theta = 0;  //Angle between xi and xi-1 along zi    (variavle for revolute)
	joint1.z_joint_type = REVOLUTE; //Joint type at z-1 axis
	dht.push_back(joint1);

	dh_parametrs joint2;
	joint2.joint_name = "joint2";
	joint2.alpha = 0;  //Angle between zi and zi - 1 along xi
	joint2.d = 0;
	joint2.a = 200;    //Length of common normal
	joint2.theta = 0;  //Angle between xi and xi-1 along zi    (variavle for revolute)
	joint2.z_joint_type = REVOLUTE; //Joint type at z-1 axis
	dht.push_back(joint2);

	dh_parametrs joint3;
	joint3.joint_name = "joint3";
	joint3.alpha = 90;  //Angle between zi and zi - 1 along xi
	joint3.d = 0;
	joint3.a = 200;    //Length of common normal
	joint3.theta = 0;  //Angle between xi and xi-1 along zi    (variavle for revolute)
	joint3.z_joint_type = REVOLUTE; //Joint type at z-1 axis
	//dht.push_back(joint3);

	robot.LoadConfig(dht);
	CAlgoFactory factory;
	VectorXf des(6);
	float speccfc = 0.001f;
	des << 200.0f, 200.0f, 0.0f, 0.0f, 0.0f, 0.0f;

	CAlgoAbstract* pJpt = factory.GiveMeSolver(DUMPEDLEASTSQUARES, des, robot); //JACOBIANTRANSPOSE 
	pJpt->SetAdditionalParametr(speccfc);
	pJpt->CalculateData();
	robot.PrintConfiguration();*/

	
	
	plugin = this;
	MoveFinished = true;
    setViewpoint = true;
	bool avail = false;
	ignoreDepthOnly = coCoviseConfig::isOn("ignoreDepthOnly", "COVER.Plugin.Revit", false,&avail);
	int port = coCoviseConfig::getInt("port", "COVER.Plugin.Revit.Server", 31821);
    textureDir = coCoviseConfig::getEntry("textures", "COVER.Plugin.Revit", "C:/Program Files (x86)/Common Files/Autodesk Shared/Materials/Textures");
    localTextureDir = coCoviseConfig::getEntry("localTextures", "COVER.Plugin.Revit", "c:/tmp");
	toRevit = NULL;
    serverConn = NULL;
    if (coVRMSController::instance()->isMaster())
    {
        serverConn = new ServerConnection(port, 1234, Message::UNDEFINED);
        if (!serverConn->getSocket())
        {
            cout << "tried to open server Port " << port << endl;
            cout << "Creation of server failed!" << endl;
            cout << "Port-Binding failed! Port already bound?" << endl;
            delete serverConn;
            serverConn = NULL;
        }
        else
        {
            cover->watchFileDescriptor(serverConn->getSocket()->get_id());
        }
    }

	struct linger linger;
	linger.l_onoff = 0;
	linger.l_linger = 0;
	cout << "Set socket options..." << endl;
	if (serverConn)
	{
		setsockopt(serverConn->get_id(NULL), SOL_SOCKET, SO_LINGER, (char *)&linger, sizeof(linger));

		cout << "Set server to listen mode..." << endl;
		serverConn->listen();
		if (!serverConn->is_connected()) // could not open server port
		{
			fprintf(stderr, "Could not open server port %d\n", port);
			delete serverConn;
			serverConn = NULL;
		}
	}
	msg = new Message;

}

bool RevitPlugin::init()
{
	cover->addPlugin("Annotation"); // we would like to have the Annotation plugin
	cover->addPlugin("Move"); // we would like to have the Move plugin
	globalmtl = new osg::Material;
	globalmtl->ref();
	globalmtl->setColorMode(osg::Material::OFF);
	globalmtl->setAmbient(osg::Material::FRONT_AND_BACK, osg::Vec4(0.2f, 0.2f, 0.2f, 1.0));
	globalmtl->setDiffuse(osg::Material::FRONT_AND_BACK, osg::Vec4(0.9f, 0.9f, 0.9f, 1.0));
	globalmtl->setSpecular(osg::Material::FRONT_AND_BACK, osg::Vec4(0.9f, 0.9f, 0.9f, 1.0));
	globalmtl->setEmission(osg::Material::FRONT_AND_BACK, osg::Vec4(0.0f, 0.0f, 0.0f, 1.0));
	globalmtl->setShininess(osg::Material::FRONT_AND_BACK, 16.0f);
	revitGroup = new osg::MatrixTransform();
	revitGroup->setName("RevitGeometry");
	scaleFactor = REVIT_FEET_TO_M; // Revit internal units are always feet
	revitGroup->setMatrix(osg::Matrix::scale(scaleFactor, scaleFactor, scaleFactor));
	cover->setScale(1000.0);
	currentGroup.push(revitGroup.get());
	cover->getObjectsRoot()->addChild(revitGroup.get());
	createMenu();
	VrmlNamespace::addBuiltIn(VrmlNodePhases::defineType());


	return true;
}
// this is called if the plugin is removed at runtime
RevitPlugin::~RevitPlugin()
{
	destroyMenu();
	while (currentGroup.size() > 1)
		currentGroup.pop();

    if (revitGroup)
    {
        revitGroup->removeChild(0, revitGroup->getNumChildren());
        cover->getObjectsRoot()->removeChild(revitGroup.get());
    }

    if (serverConn && serverConn->getSocket())
        cover->unwatchFileDescriptor(serverConn->getSocket()->get_id());
	delete serverConn;
	serverConn = NULL;

    if (toRevit && toRevit->getSocket())
        cover->unwatchFileDescriptor(toRevit->getSocket()->get_id());
	delete msg;
	toRevit = NULL;
	if(selectObjectInteraction->isRegistered())
	    coInteractionManager::the()->unregisterInteraction(selectObjectInteraction);
	delete selectObjectInteraction;
}


void RevitPlugin::deactivateAllViewpoints()
{
	for (const auto& it : viewpointEntries)
	{
		it->deactivate();
	}
}

void RevitPlugin::setDefaultMaterial(osg::StateSet *geoState)
{
	geoState->setAttributeAndModes(globalmtl.get(), osg::StateAttribute::ON);
}

bool RevitPlugin::sendMessage(Message &m)
{
	if (toRevit) // false on slaves
	{
		return toRevit->sendMessage(&m);
	}
    return false;
}


void RevitPlugin::message(int toWhom, int type, int len, const void *buf)
{
	if (type == PluginMessageTypes::MoveAddMoveNode)
	{
	}
	else if (type == PluginMessageTypes::MoveMoveNodeFinished)
	{
		MoveFinished = true;
		std::string path;
		TokenBuffer tb((const char *)buf, len);
		tb >> path;
		tb >> path;

		osg::Node *selectedNode = coVRSelectionManager::validPath(path);
		if (selectedNode)
		{
			info = dynamic_cast<RevitInfo *>(OSGVruiUserDataCollection::getUserData(selectedNode, "RevitInfo"));
			if (info)
			{
				osg::Matrix m;
				for (int i = 0; i < 4; i++)
					for (int j = 0; j < 4; j++)
						tb >> m(i, j);
				osg::Matrix currentBaseMat, dcsMat;
				osg::Group* currentNode = selectedNode->getParent(0);
				currentBaseMat.makeIdentity();
				while (currentNode != NULL && currentNode != revitGroup)
				{
					if (dynamic_cast<osg::MatrixTransform *>(currentNode))
					{
						dcsMat = ((osg::MatrixTransform *)currentNode)->getMatrix();
						currentBaseMat.postMult(dcsMat);
					}
					if (currentNode->getNumParents() > 0)
						currentNode = currentNode->getParent(0);
					else
						currentNode = NULL;
				}
				currentBaseMat(3, 0) = 0;
				currentBaseMat(3, 1) = 0;
				currentBaseMat(3, 2) = 0;
				m = m* currentBaseMat;
				TokenBuffer stb;
				stb << MovedID;
				stb << MovedDocumentID;
				stb << (double)m.getTrans().x();
				stb << (double)m.getTrans().y();
				stb << (double)m.getTrans().z();

				Message message(stb);
				message.type = (int)RevitPlugin::MSG_SetTransform;
				RevitPlugin::instance()->sendMessage(message);

			}
		}

	}
	else if (type == PluginMessageTypes::AnnotationMessage) // An AnnotationMessage has been received
	{
		AnnotationMessage *mm = (AnnotationMessage *)buf;

		switch (mm->token)
		{
		case ANNOTATION_MESSAGE_TOKEN_MOVEADD: // MOVE/ADD
		{

			int revitID = getRevitAnnotationID(mm->id);
			if (revitID > 0)
			{
				changeAnnotation(revitID, mm);
			}
			else if (revitID == -1)
			{
				createNewAnnotation(mm->id, mm);
			}
			break;
		} // case moveadd

		case ANNOTATION_MESSAGE_TOKEN_REMOVE: // Remove an annotation
		{
			int revitID = getRevitAnnotationID(mm->id);
			if (revitID > 0)
			{
				TokenBuffer stb;
				stb << revitID;

				Message message(stb);
				message.type = (int)RevitPlugin::MSG_DeleteObject;
				RevitPlugin::instance()->sendMessage(message);
			}
			else if (revitID == -1)
			{
			}
			break;
		} // case remove

		case ANNOTATION_MESSAGE_TOKEN_SELECT: // annotation selected (right-clicked)
		{

			break;
		} // case select

		case ANNOTATION_MESSAGE_TOKEN_COLOR: // Change annotation color
		{
			break;
		} // case color

		case ANNOTATION_MESSAGE_TOKEN_DELETEALL: // Deletes all Annotations
		{
			break;
		} // case deleteall

		// Release current lock on a specific annotation
		// TODO: Possibly remove this, as unlock all
		// does what this is supposed to do
		case ANNOTATION_MESSAGE_TOKEN_UNLOCK:
		{
			break;
		} //case unlock

		case ANNOTATION_MESSAGE_TOKEN_SCALE: // scale an annotation
		{

			break;
		} //case scale

		case ANNOTATION_MESSAGE_TOKEN_SCALEALL: //scale all Annotations
		{

			break;
		} //case scaleall

		case ANNOTATION_MESSAGE_TOKEN_COLORALL: //change all annotation's colors
		{
			break;
		} //case colorall

		// release lock on all annotations that are owned by sender
		case ANNOTATION_MESSAGE_TOKEN_UNLOCKALL:
		{
			break;
		} //case unlockall

		case ANNOTATION_MESSAGE_TOKEN_FORCEUNLOCK:
		{
			break;
		}

		case ANNOTATION_MESSAGE_TOKEN_HIDE: //hide an annotation
		{
			break;
		} //case hide

		case ANNOTATION_MESSAGE_TOKEN_HIDEALL: //hide all annotations
		{
			break;
		} //case hideall

		default:
			std::cerr
				<< "Annotation: Error: Bogus Annotation message with Token "
				<< (int)mm->token << std::endl;
		} //switch mm->token
	} //if type == ann_message
	else if (type == PluginMessageTypes::AnnotationTextMessage)
	{
		std::string text;
		TokenBuffer tb((const char *)buf, len);
		int id, owner,docID;
		const char *ctext;
		tb >> id;
		tb >> docID;
		tb >> owner;
		tb >> ctext;
		text = ctext;
		int AnnotationID = getRevitAnnotationID(id);
		if (AnnotationID > 0)
		{
			TokenBuffer stb;
			stb << AnnotationID;
			stb << docID;
			stb << text;

			Message message(stb);
			message.type = (int)RevitPlugin::MSG_ChangeAnnotationText;
			RevitPlugin::instance()->sendMessage(message);
		}
	}
	else if (type == PluginMessageTypes::MoveMoveNode)
	{
		osg::Matrix m;
		std::string path;
		TokenBuffer tb((const char *)buf, len);
		tb >> path;
		tb >> path;
		osg::Node *selectedNode = coVRSelectionManager::validPath(path);
		if (selectedNode)
		{
			info = dynamic_cast<RevitInfo *>(OSGVruiUserDataCollection::getUserData(selectedNode, "RevitInfo"));
			if (info)
			{
				for (int i = 0; i < 4; i++)
					for (int j = 0; j < 4; j++)
						tb >> m(i, j);
				if (MovedID != info->ObjectID)
				{
					MoveFinished = true;
				}

				lastMoveMat = invStartMoveMat*m;
				//invStartMoveMat.invert(m);

				if (MoveFinished)
				{
					MoveFinished = false;
					MovedID = info->ObjectID;
					MovedDocumentID = info->DocumentID;
					invStartMoveMat.invert(m);

					/*   TokenBuffer stb;
					   stb << info->ObjectID;
					   stb << (double)lastMoveMat.getTrans().x();
					   stb << (double)lastMoveMat.getTrans().y();
					   stb << (double)lastMoveMat.getTrans().z();

					   Message message(stb);
					   message.type = (int)RevitPlugin::MSG_SetTransform;
					   RevitPlugin::instance()->sendMessage(message);*/
				}
			}
		}
	}
	else if (type >= PluginMessageTypes::HLRS_Revit_Message && type <= (PluginMessageTypes::HLRS_Revit_Message + 100))
	{
        Message m{ type - PluginMessageTypes::HLRS_Revit_Message + MSG_NewObject , DataHandle{(char*)buf, len, false} };
		handleMessage(&m);
	}

}

RevitPlugin *RevitPlugin::plugin = NULL;
void
RevitPlugin::handleMessage(Message *m)
{
	//cerr << "got Message" << endl;
	//m->print();
	enum MessageTypes type = (enum MessageTypes)m->type;

	switch (type)
	{
	case MSG_Views:
	{
		TokenBuffer tb(m);
		int numViews;
		tb >> numViews;
		std::vector<std::string> items;
		for (int i = 0; i < numViews; i++)
		{
			std::string ViewName;
			tb >> ViewName;
			items.push_back(ViewName);
		}
		viewsCombo->setList(items);
	}
	break;
	
    case MSG_Finished:
    {
        for (auto Mat = MaterialInfos.begin(); Mat != MaterialInfos.end(); Mat++)
        {
            if (Mat->second->diffuseTexture->requestTexture)
            {
                requestTexture(Mat->second->ID, Mat->second->diffuseTexture);
            }
            if (Mat->second->bumpTexture->requestTexture)
            {
                requestTexture(Mat->second->ID, Mat->second->bumpTexture);
            }
        }
    }
    break;
	case MSG_RoomInfo:
	{
		TokenBuffer tb(m);
		double area;
		const char *roomNumber, *roomName, *levelName;
		tb >> roomNumber;
		tb >> roomName;
		tb >> area;
		tb >> levelName;
		char info[1000];
		sprintf(info, "Nr.: %s\n%s\nArea: %3.7lfm^2\nLevel: %s", roomNumber, roomName, area / 10.0, levelName);
		label1->setText(info);
		//fprintf(stderr,"Room %s %s Area: %lf Level: %s\n", roomNumber,roomName,area,levelName);
	}
	break;
	case MSG_ObjectInfo:
	{
		TokenBuffer tb(m);
		delete currentObjectInfo;
		currentObjectInfo = new ObjectInfo(tb);
	}
	break;
	case MSG_NewParameter:
	{
		TokenBuffer tb(m);
		int ID;
		int docID;
		tb >> ID;
		tb >> docID;
		int numParams;
		tb >> numParams;
		std::map<int, ElementInfo *>::iterator it = ElementIDMap[docID].find(ID);
		if (it != ElementIDMap[docID].end())
		{
			for (int i = 0; i < numParams; i++)
			{
				fprintf(stderr, "PFound: %d\n", ID);
				int pID;
				tb >> pID;
				const char *name;
				tb >> name;
				int StorageType;
				tb >> StorageType;
				std::string ParameterType;
                tb >> ParameterType;
				ElementInfo *ei = it->second;
				RevitParameter *p = new RevitParameter(pID, std::string(name), StorageType, ParameterType, (int)ei->parameters.size(), ei);
				switch (StorageType)
				{
				case Double:
					tb >> p->d;
					break;
				case ElementId:
					tb >> p->ElementReferenceID;
					break;
				case Integer:
					tb >> p->i;
					break;
				case String:
					tb >> p->s;
					break;
				default:
					tb >> p->s;
					break;
				}
				ei->addParameter(p);
			}
		}
	}
	break;
	case MSG_NewGroup:
	{
		TokenBuffer tb(m);
		int ID,docID;
		tb >> ID;
		tb >> docID;
		const char *name;
		tb >> name;
		osg::Group *newGroup = new osg::Group();
		newGroup->setName(name);
		currentGroup.top()->addChild(newGroup);

		RevitInfo *info = new RevitInfo();
		info->ObjectID = ID;
		info->DocumentID = docID;
		OSGVruiUserDataCollection::setUserData(newGroup, "RevitInfo", info);
		currentGroup.push(newGroup);
	}
	break;
	case MSG_NewDoorGroup:
	{
		TokenBuffer tb(m);
		int ID, docID;
		tb >> ID;
		tb >> docID;
		const char *name;
		tb >> name;
		osg::MatrixTransform *newTrans = new osg::MatrixTransform();
		newTrans->setName(name);
		currentGroup.top()->addChild(newTrans);
		doors.push_back(new DoorInfo(ID, name, newTrans, tb));
		RevitInfo *info = new RevitInfo();
		info->ObjectID = ID;
		info->DocumentID = docID;
		OSGVruiUserDataCollection::setUserData(newTrans, "RevitInfo", info);
		currentGroup.push(newTrans);
	}
	break;
	case MSG_NewARMarker:
	{
		TokenBuffer tb(m);
		int ID, docID;
		tb >> ID;
		tb >> docID;
		std::string name;
		tb >> name;
		float x, y, z;
		tb >> x;
		tb >> y;
		tb >> z;
		osg::Vec3 pos(x, y, z);
		tb >> x;
		tb >> y;
		tb >> z;
		osg::Vec3 ox(x, y, z);
		tb >> x;
		tb >> y;
		tb >> z;
		osg::Vec3 oy(x, y, z);
		tb >> x;
		tb >> y;
		tb >> z;
		osg::Vec3 oz(x, y, z);
		osg::Matrix mat;
		mat(0, 0) = ox.x(); mat(0, 1) = ox.y(); mat(0, 2) = ox.z();
		mat(1, 0) = oy.x(); mat(1, 1) = oy.y(); mat(1, 2) = oy.z();
		mat(2, 0) = oz.x(); mat(2, 1) = oz.y(); mat(2, 2) = oz.z();
		mat(3, 0) = pos.x(); mat(3, 1) = pos.y(); mat(3, 2) = pos.z();

		tb >> x;
		tb >> y;
		tb >> z;
		osg::Vec3 hostPos(x, y, z);
		tb >> x;
		tb >> y;
		tb >> z;
		osg::Vec3 hox(x, y, z);
		tb >> x;
		tb >> y;
		tb >> z;
		osg::Vec3 hoy(x, y, z);
		tb >> x;
		tb >> y;
		tb >> z;
		osg::Vec3 hoz(x, y, z);
		osg::Matrix hostMat;
		hostMat(0, 0) = hox.x(); hostMat(0, 1) = hox.y(); hostMat(0, 2) = hox.z();
		hostMat(1, 0) = hoy.x(); hostMat(1, 1) = hoy.y(); hostMat(1, 2) = hoy.z();
		hostMat(2, 0) = hoz.x(); hostMat(2, 1) = hoz.y(); hostMat(2, 2) = hoz.z();
		hostMat(3, 0) = hostPos.x(); hostMat(3, 1) = hostPos.y(); hostMat(3, 2) = hostPos.z();
		int MarkerID;
		tb >> MarkerID;
		double offset;
		tb >> offset;
		double angle;
		tb >> angle;
		int hostID;
		tb >> hostID;
		double size;
		tb >> size;
		std::string markerType;
		tb >> markerType;
		size = size * REVIT_FEET_TO_M * 1000;
		mat.setTrans(mat.getTrans()* REVIT_FEET_TO_M * 1000);
		ARMarkerInfo* mi;
		auto it = ARMarkers.find(ID);
		if (it != ARMarkers.end())
		{
			mi = it->second;
		}
		else
		{
			mi = new ARMarkerInfo();
			ARMarkers[ID]=mi;
		}
		mi->setValues(ID,docID, MarkerID, name, angle, offset, mat, hostMat, hostID,size,markerType);
	}
	break;
	case MSG_DeleteElement:
	{
		TokenBuffer tb(m);
		int ID, docID;
		tb >> ID;
		tb >> docID;
		std::map<int, ElementInfo*>::iterator it = ElementIDMap[docID].find(ID);
		if (it != ElementIDMap[docID].end())
		{
			//fprintf(stderr, "DFound: %d\n", ID);
			ElementInfo *ei = it->second;
			for (std::list<osg::Node *>::iterator nodesIt = ei->nodes.begin(); nodesIt != ei->nodes.end(); nodesIt++)
			{
				osg::Node *n = *nodesIt;
				const osg::MatrixTransform *mt = dynamic_cast<const osg::MatrixTransform *>(n);
				if (mt != NULL)
				{// this could be a door

					for (std::list<DoorInfo *>::iterator it = activeDoors.begin(); it != activeDoors.end();it++)
					{
						if ((*it)->transformNode == mt)
						{
							activeDoors.erase(it);
							break;
						}
					}
					for (std::list<DoorInfo *>::iterator it = doors.begin(); it != doors.end(); it++)
					{
						if ((*it)->transformNode == mt)
						{
							DoorInfo *di = *it;
							doors.erase(it);
							delete di;
							break;
						}
					}
				}
				while (n->getNumParents())
				{
					n->getParent(0)->removeChild(n);
				}
				//fprintf(stderr, "DeleteID: %d\n", ID);
			}
			delete ei;
			ElementIDMap[docID].erase(it);
			for (int i = 0; i < ikInfos.size(); i++)
			{
				if (ikInfos[i]->ID == ID && ikInfos[i]->DocumentID == docID)
				{
					delete ikInfos[i];
					ikInfos.erase(ikInfos.begin()+i);
					break;
				}
			}
		}
	}
	break;
	case MSG_NewTransform:
	{
		TokenBuffer tb(m);
		int ID, docID;
		tb >> ID;
		tb >> docID;
		const char *name;
		tb >> name;
		osg::MatrixTransform *newTrans = new osg::MatrixTransform();
		osg::Matrix m;
		m.makeIdentity();
		float x, y, z;
		tb >> x;
		tb >> y;
		tb >> z;
		m(0, 0) = x;
		m(0, 1) = y;
		m(0, 2) = z;
		tb >> x;
		tb >> y;
		tb >> z;
		m(1, 0) = x;
		m(1, 1) = y;
		m(1, 2) = z;
		tb >> x;
		tb >> y;
		tb >> z;
		m(2, 0) = x;
		m(2, 1) = y;
		m(2, 2) = z;
		tb >> x;
		tb >> y;
		tb >> z;
		m(3, 0) = x;
		m(3, 1) = y;
		m(3, 2) = z;
		newTrans->setMatrix(m);
		newTrans->setName(name);
		currentGroup.top()->addChild(newTrans);
		currentGroup.push(newTrans);
		RevitInfo* info = new RevitInfo();
		info->ObjectID = ID;
		info->DocumentID = docID;
		OSGVruiUserDataCollection::setUserData(newTrans, "RevitInfo", info);
	}
	break;
	case MSG_EndGroup:
	{
		currentGroup.pop();
	}
	break;
	case MSG_NewInstance:
	{
		TokenBuffer tb(m);
		int ID, docID;
		tb >> ID;
		tb >> docID;
		const char *name;
		tb >> name;
		osg::MatrixTransform *newTrans = new osg::MatrixTransform();
		osg::Matrix m;
		m.makeIdentity();
		float x, y, z;
		tb >> x;
		tb >> y;
		tb >> z;
		m(0, 0) = x;
		m(0, 1) = y;
		m(0, 2) = z;
		tb >> x;
		tb >> y;
		tb >> z;
		m(1, 0) = x;
		m(1, 1) = y;
		m(1, 2) = z;
		tb >> x;
		tb >> y;
		tb >> z;
		m(2, 0) = x;
		m(2, 1) = y;
		m(2, 2) = z;
		tb >> x;
		tb >> y;
		tb >> z;
		m(3, 0) = x;
		m(3, 1) = y;
		m(3, 2) = z;
		newTrans->setMatrix(m);
		newTrans->setName(name);
		currentGroup.top()->addChild(newTrans);
		currentGroup.push(newTrans);
	}
	break;
	case MSG_EndInstance:
	{
		currentGroup.pop();
	}
	break;
	case MSG_ClearAll:
	{
		while (currentGroup.size() > 1)
			currentGroup.pop();

		revitGroup->removeChild(0, revitGroup->getNumChildren());

		firstDocument = true;

		// remove viewpoints
		maxEntryNumber = 0;
		for (const auto& it : viewpointEntries)
		{
			delete it;
		}
		viewpointEntries.clear();
		for (auto& iter : ElementIDMap)
		{
			for (std::map<int, ElementInfo*>::iterator it = iter.begin(); it != iter.end(); it++)
			{
				delete (it->second);
			}
		}
		ElementIDMap.clear();

		for (auto& iter : roomInfos)
		{
			delete iter;
		}
		roomInfos.clear();
		for (std::map<int, MaterialInfo *>::iterator it = MaterialInfos.begin(); it != MaterialInfos.end(); it++)
		{
			delete (it->second);
		}
		MaterialInfos.clear();

		for (std::list<DoorInfo *>::iterator it = doors.begin(); it != doors.end(); it++)
		{
			delete *it;
		}
		ikInfos.clear();
		doors.clear();
		activeDoors.clear();
		for (const auto& pi: phaseInfos)
		{
			delete pi;
		}
		phaseInfos.clear();
		for (const auto &set: designOptionSets)
			delete set;
		designOptionSets.clear();
		

	}
	break;
    case MSG_ViewPhase:
    {
		std::string currentPhaseName;
		std::string phaseFilter;
		TokenBuffer tb(m);
		tb >> currentPhaseName;
		tb >> phaseFilter;
		setPhase(currentPhaseName);

		for (const auto& pi : phaseInfos)
		{
			if (pi->PhaseName == currentPhaseName)
			{
				PhaseGroup->setActiveButton(pi->button);
				//pi->button->setState(true);
				break;
			}
		}

		
    }
    break;
	case MSG_Phases:
	{
		int numPhases=0;
		TokenBuffer tb(m);
	    tb >> numPhases;
		for (int i = 0; i < numPhases; i++)
		{
			PhaseInfo* pi = new PhaseInfo();
			pi->ID = i;
			tb >> pi->PhaseName;
			pi->button = new ui::Button(phaseMenu, "phase" + std::to_string(phaseInfos.size()),PhaseGroup);
			pi->button->setText(pi->PhaseName);
			pi->button->setCallback([this,pi](bool state) {if (state) RevitPlugin::instance()->setPhase(pi->PhaseName); });
			phaseMenu->add(pi->button);

			for (const auto& pi1 : phaseInfos)
			{
				if (pi1->PhaseName == pi->PhaseName)
				{
					// we already got this phase, probably from an inline, ignore the new definition
					delete pi;
					pi = nullptr;
					break;
				}
			}
			if(pi)
			{
			    phaseInfos.push_back(pi);
			}
		}

		if (VrmlNodePhases::instance())
		{
			VrmlNodePhases::instance()->setNumPhases(phaseInfos.size());
		}
		if (currentPhase > numPhases - 1)
		{
			setPhase(0);
		}
	}
	break;
	case MSG_IKInfo:
	{
		TokenBuffer tb(m);
		IKInfo* iki = new IKInfo();
		tb >> iki->ID;
		tb >> iki->DocumentID;
		int numAxis = 0;
		tb >> numAxis;
		iki->axis.resize(numAxis);
		osg::Matrix mat,iMat;
		for (int i = 0; i < numAxis; i++)
		{
			int level;
			tb >> level;
			level--;
			tb >> iki->axis[level].origin;
			tb >> iki->axis[level].direction;
			tb >> iki->axis[level].min;
			tb >> iki->axis[level].max;
			int type=IKAxisInfo::Rot;
			tb >> type;
			iki->axis[level].type=IKAxisInfo::AxisType(type);
		}
		unsigned int node_id = 0;
		for (int i = 0; i < numAxis; i++)
		{
			osg::Matrix m,r;
			iki->axis[i].sumMat = mat;
			iki->axis[i].invSumMat = iMat;
			m.makeTranslate(iki->axis[i].origin);
			r.makeRotate(osg::Vec3(0, 0, 1), iki->axis[i].direction);
			mat = r*m;
			iMat.invert(mat);
			iki->axis[i].mat = mat * iki->axis[i].invSumMat ;
			iki->axis[i].invMat.invert(iki->axis[i].mat);

			node_id++;
			iki->axis[i].initIK( node_id);
		}
		iki->intiIK();
		ikInfos.push_back(iki);
		osg::Matrix m;
		m.makeIdentity();
		if (numAxis == 5)
			numAxis = 4;
		for (int i = 0; i < numAxis; i++)
		{
			m =iki->axis[i].mat*m;
		}
		/*if (iki->type == IKInfo::IKType::Rot)
		{
			xPos->setValue(iki->axis[4].origin.x());
			yPos->setValue(iki->axis[4].origin.y());
			zPos->setValue(iki->axis[3].origin.z());
		}
		else if(iki->type == IKInfo::IKType::RotTrans)
		{
			xPos->setValue(iki->axis[3].origin.x());
			yPos->setValue(iki->axis[3].origin.y());
			zPos->setValue(iki->axis[3].origin.z());
		}
		else if (iki->type == IKInfo::IKType::Trans)
		{
			xPos->setValue(iki->axis[3].origin.x());
			yPos->setValue(iki->axis[3].origin.y());
			zPos->setValue(iki->axis[3].origin.z());
		}

		yOri->setValue(1.0);*/
	}
	break;
	case MSG_NewAnnotation:
	{
		TokenBuffer tb(m);
		int ID, docID;
		tb >> ID;
		tb >> docID;
		float x, y, z;
		const char *text;
		tb >> x;
		tb >> y;
		tb >> z;
		tb >> text;

		int AID = getAnnotationID(ID);
		AnnotationMessage am;
		am.token = ANNOTATION_MESSAGE_TOKEN_MOVEADD;
		am.id = AID;
		am.sender = 101;
		am.color = 0.4;
		osg::Matrix trans;
		osg::Matrix orientation;
		trans.makeTranslate(x*scaleFactor, y*scaleFactor, z*scaleFactor); // get rid of scale part
		orientation.makeRotate(3.0, -1, 0, 0);
		matrix2array(trans, am.translation());
		matrix2array(orientation, am.orientation());
		cover->sendMessage(this, "Annotation",
			PluginMessageTypes::AnnotationMessage, sizeof(AnnotationMessage), &am);

		TokenBuffer tb3;
		tb3 << AID;
		tb3 << 101; // owner
		tb3 << text;
		cover->sendMessage(this, "Annotation",
			PluginMessageTypes::AnnotationTextMessage, tb3.getData().length(), tb3.getData().data());
		break;
	}
    case MSG_DocumentInfo:
    {
        TokenBuffer tb(m);	
        const char *fileName;
        tb >> fileName;
		tb >> TrueNorthAngle;
		tb >> ProjectHeight;
		ProjectHeight *= REVIT_FEET_TO_M;
		double eastWest;
		double northSouth;
		double xo=0.0, yo = 0.0, zo = 0.0;
		double pox=0.0, poy = 0.0, poz = 0.0;
		int GeoReference;
		tb >> eastWest;
		tb >> northSouth;
		tb >> xo;
		tb >> yo;
		tb >> zo;
		//tb >> pox;
		//tb >> poy;
		//tb >> poz;
		tb >> GeoReference;
		if(firstDocument)
		{
            firstDocument = false;
            NorthRotMat = osg::Matrix::rotate(TrueNorthAngle, osg::Vec3(0, 0, 1));
            RevitScale = osg::Matrix::scale(REVIT_FEET_TO_M, REVIT_FEET_TO_M, REVIT_FEET_TO_M);
			if (GeoReference == 1)
			{
				RevitGeoRefference = osg::Matrix::translate((xo + pox * 1000.0) * REVIT_FEET_TO_M, (yo  + poy * 1000.0) * REVIT_FEET_TO_M, (zo + poz * 1000.0) * REVIT_FEET_TO_M);
			}
			else
			{
				RevitGeoRefference.makeIdentity();
			}
            revitGroup->setMatrix(RevitScale * NorthRotMat * RevitGeoRefference);
		}
        if (fileName != currentRevitFile)
        {
            setViewpoint = true;
            currentRevitFile = fileName;
        }
        else
        {
            setViewpoint = false;
        }
        break;
    }
	case MSG_DesignOptionSets:
	{
		TokenBuffer tb(m);
		int documentID;
		tb >> documentID;
		int numSets;
		tb >> numSets;
		for (int i = 0; i < numSets; i++)
		{
			int setID;
			std::string setName;
			tb >> setID;
            tb >> setName;
            bool found = false;
            for (const auto& s : designOptionSets)
            {
                if (s->ID == setID)
                {
                    found = true;
                    break;
                }
            }
            if (!found)
            {
                RevitDesignOptionSet* set = new RevitDesignOptionSet();
                designOptionSets.push_back(set);
                set->DocumentID = documentID;
                set->ID = setID;
                set->name = setName;
                int numDOs;
                tb >> numDOs;
                for (int n = 0; n < numDOs; n++)
                {
                    RevitDesignOption DO(set);
                    tb >> DO.ID;
                    tb >> DO.name;
                    tb >> DO.visible;
                    set->designOptions.push_back(DO);
                }
                set->createSelectionList();
            }
            else
            {
                int numDOs;
                tb >> numDOs;
                for (int n = 0; n < numDOs; n++)
                {
                int ID;
                std::string DOName;
                bool DOvisible;
                    tb >> ID;
                    tb >> DOName;
                    tb >> DOvisible;
                }
            }
		}
		break;
	}
	case MSG_AddRoomInfo:
	{
		TokenBuffer tb(m);
		int ID;
		int documentID;
		tb >> ID;
		tb >> documentID;
		const char* name;
		tb >> name;
		double area;
		tb >> area;
		float x, y, z;
		tb >> x;
		tb >> y;
		tb >> z;
		osg::Vec3 pos(x, y, z);
		bool foundIt = false;
		for (const auto& it : roomInfos)
		{
			if (it->ID == ID && it->documentID == documentID)
			{
				foundIt = true;
				RevitRoomInfo* room = it;
				room->area = area;
				room->name = name;
				room->textPosition = pos;
				room->updateBillboard();
				break;
			}
		}
		if (!foundIt)
		{
			roomInfos.push_back(new RevitRoomInfo(pos, name, ID, documentID, area));
		}
	}
	break;
	case MSG_AddView:
	{
		TokenBuffer tb(m);
		int ID;
		int documentID;
		tb >> ID;
		tb >> documentID;
		const char *name;
		tb >> name;
		float x, y, z;
		tb >> x;
		tb >> y;
		tb >> z;
		osg::Vec3 pos(x, y, z);
		tb >> x;
		tb >> y;
		tb >> z;
		osg::Vec3 dir(x, y, z);
		tb >> x;
		tb >> y;
		tb >> z;
		osg::Vec3 up(x, y, z);

		bool foundIt = false;

		for (const auto &it : viewpointEntries)
		{
			if (it->ID == ID && it->documentID == documentID)
			{
				foundIt = true;
				RevitViewpointEntry *vpe = it;
				if (vpe->isActive)
				{
					vpe->setValues(pos, dir, up, name);
					vpe->activate();
				}
                if (setViewpoint && strncasecmp("Start",vpe->getName().c_str(),5)==0)
                {
					vpe->setActive(true);
                    vpe->activate();
                }
				break;
			}
		}
		if (!foundIt)
		{

			// add viewpoint to menu
			RevitViewpointEntry *vpe = new RevitViewpointEntry(pos, dir, up, this, name, ID, documentID,currentGroup.top());
			viewpointEntries.push_back(vpe);
		/*	sort(viewpointEntries.begin(), viewpointEntries.end(), [](const RevitViewpointEntry *a, const RevitViewpointEntry *b) {
				const std::string& an = a->getName();
				const std::string& bn = b->getName();
				for (size_t c = 0; c < an.size() && c < bn.size(); c++) {
					if (std::tolower(an[c]) != std::tolower(bn[c]))
						return (std::tolower(an[c]) < std::tolower(bn[c]));
				}
				return an.size() < bn.size();
				});

			for (const auto& it : viewpointEntries)
			{
				viewpointMenu->add(it->getMenuItem());
			}*/

			if (setViewpoint && strncasecmp("Start", vpe->getName().c_str(), 5) == 0)
			{
				vpe->setActive(true);
				vpe->activate();
				viewpointGroup->setDefaultValue(viewpointEntries.size() - 1);
			}
		}
	}
	break;
	case MSG_NewAnnotationID:
	{
		TokenBuffer tb(m);
		int annotationID;
		int ID;
		tb >> annotationID;
		tb >> ID;
		while (annotationIDs.size() <= annotationID)
			annotationIDs.push_back(-1);
		annotationIDs[annotationID] = ID;
		// check if we have cached changes for this Annotation and send it to Revit.
	}
	case MSG_NewMaterial:
	{
		TokenBuffer tb(m);
		MaterialInfo*mi = new MaterialInfo(tb);
		MaterialInfos[mi->ID] = mi;

		mi->geoState = new osg::StateSet;
		setDefaultMaterial(mi->geoState);
		mi->geoState->setMode(GL_LIGHTING, osg::StateAttribute::ON);
		osg::Material *localmtl = new osg::Material;
		localmtl->setColorMode(osg::Material::OFF);
		localmtl->setAmbient(osg::Material::FRONT_AND_BACK, osg::Vec4(mi->r / 255.0f, mi->g / 255.0f, mi->b / 255.0f, mi->a / 255.0f));
		localmtl->setDiffuse(osg::Material::FRONT_AND_BACK, osg::Vec4(mi->r / 255.0f, mi->g / 255.0f, mi->b / 255.0f, mi->a / 255.0f));
		localmtl->setSpecular(osg::Material::FRONT_AND_BACK, osg::Vec4(0.9f, 0.9f, 0.9f, 1.0));
		localmtl->setEmission(osg::Material::FRONT_AND_BACK, osg::Vec4(0.0f, 0.0f, 0.0f, 1.0));
		localmtl->setShininess(osg::Material::FRONT_AND_BACK, 16.0f);

		mi->geoState->setAttributeAndModes(localmtl, osg::StateAttribute::ON);
		if (mi->diffuseTexture->texturePath != "")
		{
            std::string fileName = mi->diffuseTexture->texturePath;
            osg::ref_ptr<osg::Image> diffuseImage = readImage(fileName);

            if (diffuseImage.valid())
			{
                mi->updateTexture(TextureInfo::diffuse, diffuseImage.get());
			}
            else
            {
                mi->diffuseTexture->requestTexture = true;
            }
		}

		if (mi->bumpTexture->texturePath != "")
		{
			std::string fileName = mi->bumpTexture->texturePath;
			osg::ref_ptr<osg::Image> bumpImage = readImage(fileName);
			if (!bumpImage.valid())
			{
				//osg::notify(osg::ALWAYS) << "Can't open image file" << fileName << " couldn't get it from remote" << std::endl;
                mi->bumpTexture->requestTexture = true;
			}
			else
			{

                mi->updateTexture(TextureInfo::bump,bumpImage.get());
			}
		}
		break;
	}
	case MSG_NewObject:
	{
		TokenBuffer tb(m);
		int ID;
		int docID;
		int GeometryType;

		bool isHandle = false;
		bool doWalk;
		if (m->data.getLength() == 0)
			return;
		tb >> ID;
		tb >> docID;
		if (docID >= ElementIDMap.size())
			ElementIDMap.resize(docID + 1);
		ElementInfo *ei;
		auto it = ElementIDMap[docID].find(ID);
		if (it != ElementIDMap[docID].end())
		{
			ei = it->second;
			//fprintf(stderr, "NFound: %d\n", ID);
		}
		else
		{
			ei = new ElementInfo();
			ElementIDMap[docID][ID] = ei;
			//fprintf(stderr, "NewID: %d\n", ID);
		}
		const char *name;
		tb >> name;
		ei->name = name;
		ei->ID = ID;
		ei->DocumentID = docID;
		tb >> GeometryType;
		tb >> doWalk;
		tb >> ei->createdPhase;
		tb >> ei->demolishedPhase;
		IKInfo* currentIK = nullptr;
		int level = -1;
		for (auto const& ik : ikInfos)
		{
			if (ik->ID == ID && ik->DocumentID == docID)
			{
				currentIK = ik;
				size_t pos = ei->name.find("Part");
				if (pos != std::string::npos)
				{
					level = std::stoi(ei->name.substr(pos + 4));
					level--; // level should start at 0
				}
				else
				{

					size_t pos = ei->name.find("IKHandle"); // Handle is last level
					if (pos != std::string::npos)
					{
						level = ik->axis.size() - 1;
						isHandle = true;
					}
				}
			}
		}
		if (GeometryType == OBJ_TYPE_Mesh)
		{

			osg::Geode *geode = new osg::Geode();
			geode->setName(name);
			osg::Geometry *geom = new osg::Geometry();
			cover->setRenderStrategy(geom);
			geode->addDrawable(geom);

			if (!doWalk) // don't walk on anything else than objects marked with doWalk
			{
				geode->setNodeMask(geode->getNodeMask() & ~Isect::Walk);
			}

			// set up geometry
			bool isTwoSided = false;
			tb >> isTwoSided;

			int numTriangles;
			tb >> numTriangles;

			osg::Vec3Array *vert = new osg::Vec3Array;
			osg::DrawArrays *triangles = new osg::DrawArrays(osg::PrimitiveSet::TRIANGLES, 0, numTriangles * 3);
			for (int i = 0; i < numTriangles; i++)
			{
				float x, y, z;
				tb >> x;
				tb >> y;
				tb >> z;
				vert->push_back(osg::Vec3(x, y, z));
				tb >> x;
				tb >> y;
				tb >> z;
				vert->push_back(osg::Vec3(x, y, z));
				tb >> x;
				tb >> y;
				tb >> z;
				vert->push_back(osg::Vec3(x, y, z));
			}

			if (currentIK && level >= 0)
			{
				//transform all vertices
				osg::Matrix m =  currentIK->axis[level].invSumMat * currentIK->axis[level].invMat;
				for (int i=0;i<vert->size();i++)
				{
					(*vert)[i] = (*vert)[i] *m;
				}
			}

			bool isDepthOnly = false;
			tb >> isDepthOnly;

			unsigned char r, g, b, a;
			int MaterialID;
			tb >> r;
			tb >> g;
			tb >> b;
			tb >> a;
			tb >> MaterialID;

			geom->setVertexArray(vert);
			geom->addPrimitiveSet(triangles);
			GenNormalsVisitor *sv = new GenNormalsVisitor(45.0);
			sv->apply(*geode);
			ei->nodes.push_back(geode);

			osg::StateSet *geoState;
			MaterialInfo *mi = getMaterial(MaterialID);
			if (mi && mi->shader != NULL)
			{
				geoState = mi->geoState;
				geode->setStateSet(geoState);
				mi->shader->apply(geode, geom); // generates tangents if required
			}
			else
			{
				geoState = geode->getOrCreateStateSet();
				setDefaultMaterial(geoState);
				geoState->setMode(GL_LIGHTING, osg::StateAttribute::ON);
				geode->setStateSet(geoState);
				osg::Material *localmtl = new osg::Material;
				localmtl->setColorMode(osg::Material::OFF);
				localmtl->setAmbient(osg::Material::FRONT_AND_BACK, osg::Vec4(r / 255.0f, g / 255.0f, b / 255.0f, a / 255.0f));
				localmtl->setDiffuse(osg::Material::FRONT_AND_BACK, osg::Vec4(r / 255.0f, g / 255.0f, b / 255.0f, a / 255.0f));
				localmtl->setSpecular(osg::Material::FRONT_AND_BACK, osg::Vec4(0.9f, 0.9f, 0.9f, 1.0));
				localmtl->setEmission(osg::Material::FRONT_AND_BACK, osg::Vec4(0.0f, 0.0f, 0.0f, 1.0));
				localmtl->setShininess(osg::Material::FRONT_AND_BACK, 16.0f);

				geoState->setAttributeAndModes(localmtl, osg::StateAttribute::ON);
			}

			if (a < 250 || (mi!=nullptr && mi->isTransparent))
			{
				geoState->setRenderingHint(osg::StateSet::TRANSPARENT_BIN);
				geoState->setMode(GL_BLEND, osg::StateAttribute::ON);
				geoState->setNestRenderBins(false);
			}
			else
			{
				geoState->setRenderingHint(osg::StateSet::OPAQUE_BIN);
				geoState->setMode(GL_BLEND, osg::StateAttribute::OFF);
				geoState->setNestRenderBins(false);
			}

			if (!isTwoSided)
			{
				osg::CullFace *cullFace = new osg::CullFace();
				cullFace->setMode(osg::CullFace::BACK);
				geoState->setAttributeAndModes(cullFace, osg::StateAttribute::ON);
			}
			if (isDepthOnly && !ignoreDepthOnly)
			{
				// after Video but before all normal geometry
				geoState->setRenderBinDetails(-1, "RenderBin");
				geoState->setAttributeAndModes(cover->getNoFrameBuffer().get(), osg::StateAttribute::ON);
			}
			if(currentGroup.size() == 1)// only add Revit Move info to top level geometry, otherwise the group node already has a RevitInfo
			{
				RevitInfo* info = new RevitInfo();
				info->DocumentID = docID;
				info->ObjectID = ID;
				OSGVruiUserDataCollection::setUserData(geode, "RevitInfo", info);
			}
			if (currentIK && level >= 0)
			{
				if (currentIK->axis[level].transform == nullptr)
				{
					currentIK->axis[level].transform = new osg::MatrixTransform();
					currentIK->axis[level].transform->setName(std::string("level") + std::to_string(level+1));
					currentIK->axis[level].transform->setMatrix(currentIK->axis[level].mat);
					if (currentIK->axis[level].type == IKAxisInfo::Scale)
					{
						currentIK->axis[level].scaleTransform = new osg::MatrixTransform();
						currentIK->axis[level].scaleTransform->setName(std::string("scale") + std::to_string(level + 1)); 
						currentIK->axis[level].transform->addChild(currentIK->axis[level].scaleTransform);
					}
					if(currentIK->axis[level].type == IKAxisInfo::Trans)
						currentIK->axis[level].rotTransform->setName(std::string("trans") + std::to_string(level + 1));
					else
						currentIK->axis[level].rotTransform->setName(std::string("rot") + std::to_string(level + 1));
					currentIK->axis[level].transform->addChild(currentIK->axis[level].rotTransform);

					if (level == 0)
					{
						currentGroup.top()->addChild(currentIK->axis[level].transform);
					}
					else
					{
						for (int i = 0; i < level; i++)
						{
							if (currentIK->axis[i].transform == nullptr)
							{
								if (currentIK->axis[i].type == IKAxisInfo::Scale)
								{
									currentIK->axis[i].scaleTransform = new osg::MatrixTransform();
									currentIK->axis[i].scaleTransform->setName(std::string("scale") + std::to_string(i + 1));
								}
								currentIK->axis[i].transform = new osg::MatrixTransform();
								currentIK->axis[i].transform->setName(std::string("level") + std::to_string(i+1));
								currentIK->axis[i].transform->setMatrix(currentIK->axis[i].mat);

								if (currentIK->axis[i].type == IKAxisInfo::Trans || currentIK->axis[i].type == IKAxisInfo::Scale)
									currentIK->axis[i].rotTransform->setName(std::string("trans") + std::to_string(i + 1));
								else
									currentIK->axis[i].rotTransform->setName(std::string("rot") + std::to_string(i + 1));
								if (currentIK->axis[i].scaleTransform)
								{
									currentIK->axis[i].transform->addChild(currentIK->axis[i].scaleTransform);
								}
								currentIK->axis[i].transform->addChild(currentIK->axis[i].rotTransform);
								{

									if (i == 0)
									{
										currentGroup.top()->addChild(currentIK->axis[i].transform);
									}
									else
									{
										currentIK->axis[i - 1].rotTransform->addChild(currentIK->axis[i].transform);
									}
								}
							}
						}
						currentIK->axis[level - 1].rotTransform->addChild(currentIK->axis[level].transform);
					}
				}
				if (currentIK->axis[level].type == IKAxisInfo::Scale)
				{
					currentIK->axis[level].scaleTransform->addChild(geode);
				}
				else
				{ 
					currentIK->axis[level].rotTransform->addChild(geode);
				}
				if (isHandle)
				{
					currentIK->addHandle(geode);
				}

			}
			else
			{
				currentGroup.top()->addChild(geode);
			}
			setPhaseVisible(ei);
		}
		else if (GeometryType == OBJ_TYPE_Inline)
		{
			std::string url;
			tb >> url;
			osg::MatrixTransform *mt = new osg::MatrixTransform();
			mt->setMatrix(osg::Matrix::scale(REVIT_M_TO_FEET, REVIT_M_TO_FEET, REVIT_M_TO_FEET));
			char buffer[333];
			sprintf(buffer, "Default %d", ID);
			mt->setName(buffer);
			currentGroup.top()->addChild(mt);
			osg::Node *inlineNode = NULL;
			auto it = inlineNodes.find(url);
			if (it != inlineNodes.end())
			{
				inlineNode = it->second;
			}
			else
			{
				osg::Group *g=new osg::Group();
				inlineNode = coVRFileManager::instance()->loadFile(url.c_str(),NULL,g);
				if (inlineNode)
				{
					inlineNodes[url] = inlineNode;
				}
				g->removeChild(inlineNode);
			}
			if (inlineNode)
			{
				mt->addChild(inlineNode);
			}
			if (!doWalk) // don't walk on anything else than objects marked with doWalk
			{
				if(inlineNode)
				{
				inlineNode->setNodeMask(inlineNode->getNodeMask() & ~Isect::Walk);
				}
			}


			bool isDepthOnly = false;
			tb >> isDepthOnly;
			if (isDepthOnly && !ignoreDepthOnly)
			{
				// after Video but before all normal geometry
				mt->getOrCreateStateSet()->setRenderBinDetails(-1, "RenderBin");
				mt->getOrCreateStateSet()->setAttributeAndModes(cover->getNoFrameBuffer().get(), osg::StateAttribute::ON);
			}
			ei->nodes.push_back(mt);
			RevitInfo *info = new RevitInfo();
			info->ObjectID = ID;
			info->DocumentID = docID;
			OSGVruiUserDataCollection::setUserData(mt, "RevitInfo", info);

			setPhaseVisible(ei);
		}

	}
	break;
    case MSG_File:
    {
        TokenBuffer tb(msg);
        if(tb.getData().getLength()>0)
        {
        const char *buf=nullptr;
        int numBytes;
        std::string fileName;
        int MatID;
        tb >> MatID;
        tb >> fileName;
        localTextureFile = localTextureDir + "/" + fileName;
        tb >> numBytes;
		if(numBytes > 0)
		{
            buf = tb.getBinary(numBytes);
		}
        if (numBytes > 0)
        {
#ifdef _WIN32
            int fd = open(localTextureFile.c_str(), O_RDWR | O_CREAT | O_BINARY, 0777);
#else
            int fd = open(localTextureFile.c_str(), O_RDWR | O_CREAT, 0777);
#endif
            if (fd != -1)
            {
                if (write(fd, buf, numBytes) != numBytes)
                {
                    osg::notify(osg::ALWAYS) << "remoteFetch: " << localTextureFile << " write error" << std::endl;
                }
                close(fd);
            }
            else
            {

                osg::notify(osg::ALWAYS) << "remoteFetch: " << localTextureFile << " can't open File" << std::endl;;
            }

            osg::Image * image = osgDB::readImageFile(localTextureFile);
            if (image == NULL)
            {
                
                    osg::notify(osg::ALWAYS) << "Can't open image file" << fileName << " could not get it from remote" << std::endl;
            }

        }
        }
        break;
    }
	/*   case MSG_NewPolyMesh:
	   {
		   cerr << "not used anymore" << endl;
		   TokenBuffer tb(m);

		   int numPoints;
		   int numTriangles;
		   int numNormals;
		   int numUVs;
		   tb >> numPoints;
		   tb >> numTriangles;
		   tb >> numNormals;
		   tb >> numUVs;
		   osg::Geode *geode = new osg::Geode();
		   //geode->setName(name);
		   osg::Geometry *geom = new osg::Geometry();
		   geom->setUseDisplayList(coVRConfig::instance()->useDisplayLists());
		   geom->setUseVertexBufferObjects(coVRConfig::instance()->useVBOs());
		   geode->addDrawable(geom);

		   // set up geometry

		   osg::Vec3Array *points = new osg::Vec3Array;
		   points->resize(numPoints);
		   for (int i = 0; i < numPoints; i++)
		   {
			   tb >> (*points)[i][0];
			   tb >> (*points)[i][1];
			   tb >> (*points)[i][2];
		   }
		   osg::Vec3Array *norms = new osg::Vec3Array;
		   norms->resize(numNormals);
		   for (int i = 0; i < numNormals; i++)
		   {
			   tb >> (*norms)[i][0];
			   tb >> (*norms)[i][1];
			   tb >> (*norms)[i][2];
		   }
		   osg::Vec2Array *UVs = new osg::Vec2Array;
		   UVs->resize(numUVs);
		   for (int i = 0; i < numUVs; i++)
		   {
			   tb >> (*UVs)[i][0];
			   tb >> (*UVs)[i][1];
		   }

		   osg::Vec3Array *vert = new osg::Vec3Array;
		   vert->reserve(numTriangles*3);

		   osg::Vec3Array *normals = NULL;
		   if(numNormals == numPoints)
		   {
			   normals = new osg::Vec3Array;
			   normals->reserve(numTriangles*3);
		   }
		   if(numNormals == 1)
		   {
			   normals = new osg::Vec3Array;
			   normals->push_back((*norms)[0]);
			   normals->push_back((*norms)[0]);
			   normals->push_back((*norms)[0]);
		   }
		   osg::Vec2Array *texcoords = NULL;
		   if(numUVs == numPoints)
		   {
			   texcoords = new osg::Vec2Array;
			   texcoords->reserve(numTriangles*3);
		   }
		   osg::DrawArrays *triangles = new osg::DrawArrays(osg::PrimitiveSet::TRIANGLES, 0, numTriangles * 3);
		   for (int i = 0; i < numTriangles; i++)
		   {
			   int v1,v2,v3;
			   tb >> v1;
			   tb >> v2;
			   tb >> v3;
			   vert->push_back((*points)[v1]);
			   vert->push_back((*points)[v2]);
			   vert->push_back((*points)[v3]);
			   if(numUVs == numPoints)
			   {
				   texcoords->push_back((*UVs)[v1]);
				   texcoords->push_back((*UVs)[v2]);
				   texcoords->push_back((*UVs)[v3]);
			   }
			   if(numNormals == numPoints)
			   {
				   normals->push_back((*norms)[v1]);
				   normals->push_back((*norms)[v2]);
				   normals->push_back((*norms)[v3]);
			   }
		   }

		   unsigned char r, g, b, a;
		   int MaterialID;
		   tb >> r;
		   tb >> g;
		   tb >> b;
		   tb >> a;
		   tb >> MaterialID;

		   osg::StateSet *geoState;
		   MaterialInfo *mi = getMaterial(MaterialID);
		   if (mi)
		   {
			   geoState = mi->geoState;
			   geode->setStateSet(geoState);
		   }
		   else
		   {
			   geoState = geode->getOrCreateStateSet();
			   setDefaultMaterial(geoState);
			   geoState->setMode(GL_LIGHTING, osg::StateAttribute::ON);
			   geode->setStateSet(geoState);
			   osg::Material *localmtl = new osg::Material;
			   localmtl->setColorMode(osg::Material::AMBIENT_AND_DIFFUSE);
			   localmtl->setAmbient(osg::Material::FRONT_AND_BACK, osg::Vec4(r / 255.0f, g / 255.0f, b / 255.0f, a / 255.0f));
			   localmtl->setDiffuse(osg::Material::FRONT_AND_BACK, osg::Vec4(r / 255.0f, g / 255.0f, b / 255.0f, a / 255.0f));
			   localmtl->setSpecular(osg::Material::FRONT_AND_BACK, osg::Vec4(0.9f, 0.9f, 0.9f, 1.0));
			   localmtl->setEmission(osg::Material::FRONT_AND_BACK, osg::Vec4(0.0f, 0.0f, 0.0f, 1.0));
			   localmtl->setShininess(osg::Material::FRONT_AND_BACK, 16.0f);

			   geoState->setAttributeAndModes(localmtl, osg::StateAttribute::ON);
		   }

		   if (a < 250)
		   {
			   geoState->setRenderingHint(osg::StateSet::TRANSPARENT_BIN);
			   geoState->setMode(GL_BLEND, osg::StateAttribute::ON);
			   geoState->setNestRenderBins(false);
		   }
		   else
		   {
			   geoState->setRenderingHint(osg::StateSet::OPAQUE_BIN);
			   geoState->setMode(GL_BLEND, osg::StateAttribute::OFF);
			   geoState->setNestRenderBins(false);
		   }


		   geom->setVertexArray(vert);
		   if(numNormals == numPoints)
		   {
			   geom->setNormalArray(normals);
			   geom->getNormalArray()->setBinding(osg::Array::BIND_PER_VERTEX);
		   }
		   else if ( numNormals == 1)
		   {
			   geom->setNormalArray(normals);
			   geom->getNormalArray()->setBinding(osg::Array::BIND_OVERALL);
		   }
		   if(texcoords !=NULL)
		   {
			   geom->setTexCoordArray(0,texcoords);
		   }
		   geom->addPrimitiveSet(triangles);
		   //GenNormalsVisitor *sv = new GenNormalsVisitor(45.0);
		   //sv->apply(*geode);
		   //ei->nodes.push_back(geode);
		   currentGroup.top()->addChild(geode);

	   }
	   break;*/
	default:
		switch (m->type)
		{
		case Message::SOCKET_CLOSED:
		case Message::CLOSE_SOCKET:
	        if (coVRMSController::instance()->isMaster())
            {
            if(toRevit!=nullptr && toRevit->getSocket() !=nullptr)
            {
            cover->unwatchFileDescriptor(toRevit->getSocket()->get_id());
			toRevit.reset(nullptr);
            }
            }

			cerr << "connection to Revit closed" << endl;
			break;
		default:
			cerr << "Unknown message [" << m->type << "] "  << endl;
			break;
		}
	}
}
PhaseInfo::~PhaseInfo()
{
	delete button;
}

void RevitPlugin::setPhase(std::string phaseName)
{
	for (const auto& pi : phaseInfos)
	{
		if (pi->PhaseName == phaseName)
		{
			setPhase(pi->ID);
			break;
		}
	}

}
void RevitPlugin::setPhase(int phase)
{
	if (phase == currentPhase)
		return;
	for (const auto& pi : phaseInfos)
	{
		if (pi->ID == phase)
		{
			currentPhase = pi->ID;
			for (auto& iter : ElementIDMap)
			{
				for (std::map<int, ElementInfo*>::iterator it = iter.begin(); it != iter.end(); it++)
				{
					setPhaseVisible(it->second);
				}
			}if (VrmlNodePhases::instance())
			{
				VrmlNodePhases::instance()->setPhase(currentPhase);
				VrmlNodePhases::instance()->setPhase(pi->PhaseName);
			}
			return;
		}
	}

}

osg::Image *RevitPlugin::readImage(std::string fileName)
{
	if (fileName.length() > 2 && fileName[1] == ':')
	{
		// this is an absolute windows path, remove the drive
		fileName = fileName.substr(2);
	}
    std::size_t found = fileName.find_first_of("|", 0);
    if (found != std::string::npos)
    {
        fileName = fileName.substr(0, found);
    }
    std::replace( fileName.begin(), fileName.end(), '\\', '/');

    found = fileName.find("Autodesk Shared/Materials/Textures//", 0);
    if (found != std::string::npos)
    {
        fprintf(stderr,"%s %d",fileName.c_str(), (int)found);
        fileName = fileName.substr(found+36);
        fprintf(stderr,"new %s %d",fileName.c_str(), (int)found);
    }
    localTextureFile = localTextureDir + "/" + fileName;
    found = fileName.find_last_of('\\');
    std::string fn;
    if (found != std::string::npos)
    {
        fn = fileName.substr(found+1);
    }
    else
    {
        found = fileName.find_last_of('/');
        if (found != std::string::npos)
        {
            fn = fileName.substr(found+1);
        }
		else
		{
			fn = fileName;
		}
    }
    std::string localTextureFileOnly = textureDir + "/" + fn;
    std::string texFile = textureDir + "/" + fileName;

    osg::Image *diffuseImage = osgDB::readImageFile(texFile);
    if (diffuseImage==NULL)
    {
        diffuseImage = osgDB::readImageFile(localTextureFile);
        if (diffuseImage == NULL)
        {
			diffuseImage = osgDB::readImageFile(textureDir+"/1/Mats/"+fn);
			if (diffuseImage == NULL)
			{
				diffuseImage = osgDB::readImageFile(fileName);
				if (diffuseImage == NULL)
				{
					diffuseImage = osgDB::readImageFile(localTextureFileOnly);
					if (diffuseImage == NULL)
					{
						cerr << "did not find it under any of its names, even not " << localTextureFileOnly << endl;
						return NULL;
					}
				}
			}
        }
    }
    return diffuseImage;
}

bool
RevitPlugin::checkDoors()
{
    bool needUpdate = false;

	osg::Matrix headmat = cover->getViewerMat();
	headmat *= cover->getInvBaseMat();
	osg::Vec3 ViewerPosition = headmat.getTrans();
	for (std::list<DoorInfo *>::iterator it = doors.begin(); it != doors.end(); it++)
	{
		(*it)->checkStart(ViewerPosition);
	}
	if (activeDoors.size() > 0)
	{
        needUpdate = true;
	}
	for (std::list<DoorInfo *>::iterator it = activeDoors.begin(); it != activeDoors.end();)
	{
		if ((*it)->update(ViewerPosition))
		{
			it++;
		}
		else
		{
			it = activeDoors.erase(it);
		}
	}

    return needUpdate;
}

bool
RevitPlugin::update()
{
	if (selectObjectInteraction->isRegistered())
	{
		if (selectObjectInteraction->wasStarted())
		{
			coInteractionManager::the()->unregisterInteraction(selectObjectInteraction);

			const osg::NodePath& intersectedNodePath = cover->getIntersectedNodePath();
			bool isSceneNode = false;
			for (std::vector<osg::Node*>::const_iterator iter = intersectedNodePath.begin();
				iter != intersectedNodePath.end();
				++iter)
			{
				auto currentNode = (*iter);
				info = dynamic_cast<RevitInfo*>(OSGVruiUserDataCollection::getUserData(currentNode, "RevitInfo"));
				if (info)
				{

					TokenBuffer stb;
					currentObjectID = info->ObjectID;
					currentDocumentID = info->DocumentID;
					stb << currentObjectID;
					stb << currentDocumentID;

					Message message(stb);
					message.type = (int)RevitPlugin::MSG_ObjectInfo;
					RevitPlugin::instance()->sendMessage(message);
				}
			}

		}
	}
	for (auto & iki : ikInfos)
	{
		iki->update();
	}
	if (currentIKI && currentIKI->iks->getInteraction()->wasStarted())
	{

			startCompleteMat.makeIdentity();
			osg::Node* parent = currentIKI->axis[0].transform;
			while (parent!=nullptr && (parent = parent->getParent(0))!=nullptr)
			{
				osg::MatrixTransform* mt = dynamic_cast<osg::MatrixTransform*>(parent);
				if (mt)
				{
					startCompleteMat = startCompleteMat  * mt->getMatrix() ;
				}
				if (parent == revitGroup)
					break;
			}
		    invStartCompleteMat.invert(startCompleteMat);
			startHand = cover->getPointerMat(); 
			invStartHand.invert(startHand);

			startPosition = currentIKI->getPosition();
			startOrientation = currentIKI->getOrientation();
	}
	if (currentIKI)
	{
		//if(!currentIKI->iks->getInteraction()->isRunning())
		//fprintf(stderr, "notRunning\n");
	}
	if (currentIKI && currentIKI->iks->getInteraction()->isRunning())
	{
			osg::Matrix moveMat = startCompleteMat * cover->getBaseMat() * invStartHand * cover->getPointerMat() * cover->getInvBaseMat() * invStartCompleteMat;
		
			currentIKI->updateIK(startPosition * moveMat, osg::Matrix::transform3x3(startOrientation, moveMat));
	}
	for (const auto& m : ARMarkers)
	{
		m.second->update();
	}
    return checkDoors();
}
void RevitPlugin::setPhaseVisible(ElementInfo *ei)
{
	bool visible=false;
	if (ei->createdPhase <= currentPhase && (ei->demolishedPhase == -1 || ei->demolishedPhase > currentPhase))
	{
		visible = true;
	}

	for (std::list<osg::Node*>::iterator nodesIt = ei->nodes.begin(); nodesIt != ei->nodes.end(); nodesIt++)
	{
		osg::Node* n = *nodesIt;
		if(visible)
			n->setNodeMask((n->getNodeMask()&127)   | Isect::Visible);
		else
		    n->setNodeMask((n->getNodeMask() & 127) & ~Isect::Visible);

	}
}

void
RevitPlugin::preFrame()
{
	if (serverConn && serverConn->is_connected() && serverConn->check_for_input()) // we have a server and received a connect
	{
		//   std::cout << "Trying serverConn..." << std::endl;
		toRevit = serverConn->spawn_connection();
		if (toRevit && toRevit->is_connected())
		{
			fprintf(stderr, "Connected to Revit\n");
            cover->watchFileDescriptor(toRevit->getSocket()->get_id());
		}
	}
	char gotMsg = '\0';
	if (coVRMSController::instance()->isMaster())
	{
		if (toRevit)
		{
			static double lastTime = 0;
			if (cover->frameTime() > lastTime + 4)
			{
				TokenBuffer stb;
				
				osg::Matrix projectHeightMatrix;
				projectHeightMatrix.makeTranslate(osg::Vec3(0,0,ProjectHeight));
				osg::Matrix mat = RevitScale * NorthRotMat * projectHeightMatrix * RevitGeoRefference * cover->getBaseMat();
				osg::Matrix invMat;
				invMat.invert(mat);
				osg::Matrix viewerTrans = cover->getViewerMat() * invMat;
				
				
				osg::Vec3 eyePos = viewerTrans.getTrans();
				osg::Vec3 viewDir;
				viewDir[0] = viewerTrans(1, 0);
				viewDir[1] = viewerTrans(1, 1);
				viewDir[2] = viewerTrans(1, 2);
				viewDir.normalize();
				
				static osg::Vec3 oldEyePos(0, 0, 0);
				if ((eyePos - oldEyePos).length() > 0.3) // if distance to old pos > 30cm
				{
					oldEyePos = eyePos;
					lastTime = cover->frameTime();

					double eyePosition[3];
					double viewDirection[3];
					eyePosition[0] = eyePos[0];
					eyePosition[1] = eyePos[1];
					eyePosition[2] = eyePos[2];

					viewDirection[0] = viewDir[0];
					viewDirection[1] = viewDir[1];
					viewDirection[2] = viewDir[2];
					

					stb << (double)eyePosition[0];
					stb << (double)eyePosition[1];
					stb << (double)eyePosition[2];
					stb << (double)viewDirection[0];
					stb << (double)viewDirection[1];
					stb << (double)viewDirection[2];

					Message message(stb);
					message.type = (int)RevitPlugin::MSG_AvatarPosition;
					RevitPlugin::instance()->sendMessage(message);
				}
			}
		}
		while (toRevit && toRevit->check_for_input())
		{
			toRevit->recv_msg(msg);
			if (msg)
			{
				gotMsg = '\1';
				coVRMSController::instance()->sendSlaves(&gotMsg, sizeof(char));
				coVRMSController::instance()->sendSlaves(msg);

				cover->sendMessage(this, coVRPluginSupport::TO_SAME_OTHERS, PluginMessageTypes::HLRS_Revit_Message + msg->type - MSG_NewObject, msg->data.length(), msg->data.data());
				handleMessage(msg);
			}
			else
			{
				gotMsg = '\0';
				cerr << "could not read message" << endl;
				break;
			}
		}
		gotMsg = '\0';
		coVRMSController::instance()->sendSlaves(&gotMsg, sizeof(char));
	}
	else
	{
		do
		{
			coVRMSController::instance()->readMaster(&gotMsg, sizeof(char));
			if (gotMsg != '\0')
			{
				coVRMSController::instance()->readMaster(msg);
				handleMessage(msg);
			}
		} while (gotMsg != '\0');
	}
}

int RevitPlugin::getRevitAnnotationID(int ai)
{
	if (ai >= annotationIDs.size())
		return -1; // never seen this Annotation (-2 == has already been created but revitID has not yet been received)
	else return annotationIDs[ai];
}
int RevitPlugin::getAnnotationID(int revitID)
{
	for (int i = 0; i < annotationIDs.size(); i++)
	{
		if (annotationIDs[i] == revitID)
			return i;
	}
	int newID = annotationIDs.size();
	annotationIDs.push_back(revitID);
	return newID;
}


void RevitPlugin::createNewAnnotation(int id, AnnotationMessage *am)
{
	while (annotationIDs.size() <= id)
		annotationIDs.push_back(-1);
	annotationIDs[id] = -2;
	osg::Matrix trans;
	osg::Matrix ori;
	for (unsigned y = 0; y < 4; ++y)
	{
		for (unsigned x = 0; x < 4; ++x)
		{
			ori(x, y) = am->_orientation[y * 4 + x];
			trans(x, y) = am->_translation[y * 4 + x];
		}
	}
	coCoord orientation(ori);
	TokenBuffer stb;
	stb << id;
	stb << (double)trans.getTrans()[0] / scaleFactor;
	stb << (double)trans.getTrans()[1] / scaleFactor;
	stb << (double)trans.getTrans()[2] / scaleFactor;
	stb << (double)orientation.hpr[0];
	stb << (double)orientation.hpr[1];
	stb << (double)orientation.hpr[2];
	char tmpText[100];
	sprintf(tmpText, "Annotation %d", id);
	stb << tmpText;

	Message message(stb);
	message.type = (int)RevitPlugin::MSG_NewAnnotation;
	RevitPlugin::instance()->sendMessage(message);
}
void RevitPlugin::changeAnnotation(int id, AnnotationMessage *am)
{
	osg::Matrix trans;
	osg::Matrix ori;
	for (unsigned y = 0; y < 4; ++y)
	{
		for (unsigned x = 0; x < 4; ++x)
		{
			ori(x, y) = am->_orientation[y * 4 + x];
			trans(x, y) = am->_translation[y * 4 + x];
		}
	}
	coCoord orientation(ori);
	TokenBuffer stb;
	stb << id;
	stb << (double)trans.getTrans()[0] / scaleFactor;
	stb << (double)trans.getTrans()[1] / scaleFactor;
	stb << (double)trans.getTrans()[2] / scaleFactor;
	stb << (double)orientation.hpr[0];
	stb << (double)orientation.hpr[1];
	stb << (double)orientation.hpr[2];

	Message message(stb);
	message.type = (int)RevitPlugin::MSG_ChangeAnnotation;
	RevitPlugin::instance()->sendMessage(message);
}
MaterialInfo * RevitPlugin::getMaterial(int revitID)
{
	std::map<int, MaterialInfo *>::iterator it = MaterialInfos.find(revitID);
	if (it != MaterialInfos.end())
	{
		return(it->second);
	}
	return NULL;


}



// pretend types, something like this
struct pixel
{
	uint8_t red;
	uint8_t green;
	uint8_t blue;
	uint8_t alpha;
};


// determine intensity of pixel, from 0 - 1
const double intensity(const pixel& pPixel)
{
	const double r = static_cast<double>(pPixel.red);
	const double g = static_cast<double>(pPixel.green);
	const double b = static_cast<double>(pPixel.blue);

	const double average = (r + g + b) / 3.0;

	return average / 255.0;
}

const int repeat(int pX, int pMax)
{
	pMax -= 1;
	if (pX > pMax)
	{
		return 0;
	}
	else if (pX < 0)
	{
		return pMax;
	}
	else
	{
		return pX;
	}
}

// transform -1 - 1 to 0 - 255
const uint8_t map_component(double pX)
{
	return (pX + 1.0) * (255.0 / 2.0);
}

void RevitPlugin::requestTexture(int matID, TextureInfo * texture)
{
    std::string filePathName = texture->texturePath;
    std::size_t found = filePathName.find_first_of("|", 0);
    if (found != std::string::npos)
    {
        filePathName = filePathName.substr(0, found);
    }
    found = filePathName.find_last_of('\\');
    std::string fileName;
    if (found != std::string::npos)
    {
        fileName = filePathName.substr(found+1);
    }
    else
    {
        found = filePathName.find_last_of('/');
        if (found != std::string::npos)
        {
            fileName = filePathName.substr(found+1);
        }
    }

    char gotMsg = '\1';
    if (coVRMSController::instance()->isMaster())
    {
        TokenBuffer stb;
        stb << matID;
        stb << filePathName;
        stb << fileName;
        Message message(stb);
        message.type = (int)RevitPlugin::MSG_File;
        if (RevitPlugin::instance()->sendMessage(message) == false)
        {
            gotMsg = '\0';
			coVRMSController::instance()->sendSlaves(&gotMsg, sizeof(char));
        }
        else
        {
            while (toRevit)
            {
                toRevit->recv_msg(msg);
                if (msg)
                {
                    gotMsg = '\1';
                    coVRMSController::instance()->sendSlaves(&gotMsg, sizeof(char));
                    coVRMSController::instance()->sendSlaves(msg);
                    cover->sendMessage(this, coVRPluginSupport::TO_SAME_OTHERS, PluginMessageTypes::HLRS_Revit_Message + msg->type - MSG_NewObject, msg->data.length(), msg->data.data());
                    handleMessage(msg);
                    if (msg->type == MSG_File)
                        break; // done
                }
                else
                {
                    gotMsg = '\0';
                    cerr << "could not read message" << endl;
                    break;
                }
            }
            gotMsg = '\0';
            coVRMSController::instance()->sendSlaves(&gotMsg, sizeof(char));
        }
    }
    else
    {
        do
        {
            coVRMSController::instance()->readMaster(&gotMsg, sizeof(char));
            if (gotMsg != '\0')
            {
                coVRMSController::instance()->readMaster(msg);
                handleMessage(msg);
            }
        } while (gotMsg != '\0');
    }

}

osg::Image *MaterialInfo::createNormalMap(osg::Image *srcImage, double pStrength)
{
	// assume square texture, not necessarily true in real code
	osg::Image *result = new osg::Image();
	int width = srcImage->s();
	int height = srcImage->t();
	result->allocateImage(width, height, 1, GL_RGB, srcImage->getDataType());
	if (srcImage->getPixelFormat() == GL_LUMINANCE || srcImage->getPixelFormat() == GL_INTENSITY || srcImage->getPixelFormat() == GL_ALPHA)
	{
		for (size_t row = 0; row < height; ++row)
		{
			for (size_t column = 0; column < width; ++column)
			{
				// surrounding pixels
				const pixel *topLeft = (const pixel*)srcImage->data(repeat(column - 1, width), repeat(row - 1, height));
				const pixel *top = (const pixel*)srcImage->data(repeat(column, width), repeat(row - 1, height));
				const pixel *topRight = (const pixel*)srcImage->data(repeat(column + 1, width), repeat(row - 1, height));
				const pixel *right = (const pixel*)srcImage->data(repeat(column + 1, width), repeat(row, height));
				const pixel *bottomRight = (const pixel*)srcImage->data(repeat(column + 1, width), repeat(row + 1, height));
				const pixel *bottom = (const pixel*)srcImage->data(repeat(column, width), repeat(row + 1, height));
				const pixel *bottomLeft = (const pixel*)srcImage->data(repeat(column - 1, width), repeat(row + 1, height));
				const pixel *left = (const pixel*)srcImage->data(repeat(column - 1, width), repeat(row, height));

				// their intensities
				const double tl = topLeft->red / 255.0;
				const double t = top->red / 255.0;
				const double tr = topRight->red / 255.0;
				const double r = right->red / 255.0;
				const double br = bottomRight->red / 255.0;
				const double b = bottom->red / 255.0;
				const double bl = bottomLeft->red / 255.0;
				const double l = left->red / 255.0;

				// sobel filter
				const double dX = (tr + 2.0 * r + br) - (tl + 2.0 * l + bl);
				const double dY = (bl + 2.0 * b + br) - (tl + 2.0 * t + tr);
				const double dZ = 1.0 / pStrength;

				osg::Vec3d v(dX, dY, dZ);
				v.normalize();

				// convert to rgb
				unsigned char* dst_pixel = result->data(column, row, 0);
				*(dst_pixel++) = map_component(v.x());
				*(dst_pixel++) = map_component(v.y());
				*(dst_pixel++) = map_component(v.z());
			}
		}
	}
	else
	{
		for (size_t row = 0; row < height; ++row)
		{
			for (size_t column = 0; column < width; ++column)
			{
				// surrounding pixels
				const pixel *topLeft = (const pixel*)srcImage->data(repeat(column - 1, width), repeat(row - 1, height));
				const pixel *top = (const pixel*)srcImage->data(repeat(column, width), repeat(row - 1, height));
				const pixel *topRight = (const pixel*)srcImage->data(repeat(column + 1, width), repeat(row - 1, height));
				const pixel *right = (const pixel*)srcImage->data(repeat(column + 1, width), repeat(row, height));
				const pixel *bottomRight = (const pixel*)srcImage->data(repeat(column + 1, width), repeat(row + 1, height));
				const pixel *bottom = (const pixel*)srcImage->data(repeat(column, width), repeat(row + 1, height));
				const pixel *bottomLeft = (const pixel*)srcImage->data(repeat(column - 1, width), repeat(row + 1, height));
				const pixel *left = (const pixel*)srcImage->data(repeat(column - 1, width), repeat(row, height));

				// their intensities
				const double tl = intensity(*topLeft);
				const double t = intensity(*top);
				const double tr = intensity(*topRight);
				const double r = intensity(*right);
				const double br = intensity(*bottomRight);
				const double b = intensity(*bottom);
				const double bl = intensity(*bottomLeft);
				const double l = intensity(*left);

				// sobel filter
				const double dX = (tr + 2.0 * r + br) - (tl + 2.0 * l + bl);
				const double dY = (bl + 2.0 * b + br) - (tl + 2.0 * t + tr);
				const double dZ = 1.0 / pStrength;

				osg::Vec3d v(dX, dY, dZ);
				v.normalize();

				// convert to rgb
				unsigned char* dst_pixel = result->data(column, row, 0);
				*(dst_pixel++) = map_component(v.x());
				*(dst_pixel++) = map_component(v.y());
				*(dst_pixel++) = map_component(v.z());
			}
		}
	}

	return result;
}



COVERPLUGIN(RevitPlugin)

TextureInfo::TextureInfo(TokenBuffer & tb)
{
	tb >> texturePath;
	tb >> sx;
	tb >> sy;
	tb >> ox;
	tb >> oy;
	tb >> angle;
	tb >> r;
	tb >> g;
	tb >> b;
	amount = 1.0;
	requestTexture=false;
	image = nullptr;
}

MaterialInfo::MaterialInfo(TokenBuffer& tb)
{
	tb >> ID;
	tb >> DocumentID;
	diffuseTexture = new TextureInfo(tb);
	diffuseTexture->type = TextureInfo::diffuse;
	bumpTexture = new TextureInfo(tb);
	bumpTexture->type = TextureInfo::bump;
	tb >> bumpTexture->amount;
	r = diffuseTexture->r;
	g = diffuseTexture->g;
	b = diffuseTexture->b;
	a = 255;
	shader = NULL;
	isTransparent = false;
	if (diffuseTexture->image != nullptr)
	{
		isTransparent = diffuseTexture->image->isImageTranslucent();
	}
}

void MaterialInfo::updateTexture(TextureInfo::textureType type, osg::Image * image)
{
    int textureUnit=0;
    if (type == TextureInfo::bump && diffuseTexture->image != NULL)
        textureUnit = 1;

    osg::Texture2D *texture = new osg::Texture2D(image);
	texture->setResizeNonPowerOfTwoHint(false);
    texture->setFilter(osg::Texture2D::MIN_FILTER, osg::Texture2D::LINEAR);
    texture->setFilter(osg::Texture2D::MAG_FILTER, osg::Texture2D::LINEAR);
    texture->setWrap(osg::Texture2D::WRAP_S, osg::Texture2D::REPEAT);
    texture->setWrap(osg::Texture2D::WRAP_T, osg::Texture2D::REPEAT);
    geoState->setTextureAttribute(textureUnit, texture);
    textureUnit++;

    if (type == TextureInfo::diffuse)
    {
        osg::Uniform *revitSX = new osg::Uniform("revitSX", (float)(diffuseTexture->sx * REVIT_FEET_TO_M));
        osg::Uniform *revitSY = new osg::Uniform("revitSY", (float)(diffuseTexture->sy * REVIT_FEET_TO_M));
        osg::Uniform *revitOX = new osg::Uniform("revitOX", (float)(diffuseTexture->ox * REVIT_FEET_TO_M));
        osg::Uniform *revitOY = new osg::Uniform("revitOY", (float)(diffuseTexture->oy * REVIT_FEET_TO_M));
        osg::Uniform *revitAngle = new osg::Uniform("revitAngle", (float)diffuseTexture->angle);
        geoState->addUniform(revitSX);
        geoState->addUniform(revitSY);
        geoState->addUniform(revitOX);
        geoState->addUniform(revitOY);
        geoState->addUniform(revitAngle);
        if(bumpTexture->image == NULL)
            shader = coVRShaderList::instance()->get("RevitDiffuse");
        diffuseTexture->image = image;
		isTransparent = diffuseTexture->image->isImageTranslucent();
    }
    else if (type == TextureInfo::bump)
    {


        osg::Image *ni = createNormalMap(image, bumpTexture->amount);
        osg::Texture2D *normalTexture = new osg::Texture2D(ni);
		normalTexture->setResizeNonPowerOfTwoHint(false);
        normalTexture->setFilter(osg::Texture2D::MIN_FILTER, osg::Texture2D::LINEAR);
        normalTexture->setFilter(osg::Texture2D::MAG_FILTER, osg::Texture2D::LINEAR);
        normalTexture->setWrap(osg::Texture2D::WRAP_S, osg::Texture2D::REPEAT);
        normalTexture->setWrap(osg::Texture2D::WRAP_T, osg::Texture2D::REPEAT);
        geoState->setTextureAttribute(textureUnit, normalTexture);
        textureUnit++;

        osg::Uniform *revitSX = new osg::Uniform("revitSX", (float)(bumpTexture->sx * REVIT_FEET_TO_M));
        osg::Uniform *revitSY = new osg::Uniform("revitSY", (float)(bumpTexture->sy * REVIT_FEET_TO_M));
        osg::Uniform *revitOX = new osg::Uniform("revitOX", (float)(bumpTexture->ox * REVIT_FEET_TO_M));
        osg::Uniform *revitOY = new osg::Uniform("revitOY", (float)(bumpTexture->oy * REVIT_FEET_TO_M));
        osg::Uniform *revitAngle = new osg::Uniform("revitAngle", (float)bumpTexture->angle);
        geoState->addUniform(revitSX);
        geoState->addUniform(revitSY);
        geoState->addUniform(revitOX);
        geoState->addUniform(revitOY);
        geoState->addUniform(revitAngle);
        if (diffuseTexture->image != NULL)
        {
            shader = coVRShaderList::instance()->get("RevitDiffuseBump");
        }
        else
        {
            shader = coVRShaderList::instance()->get("RevitBumpOnly");
        }
        bumpTexture->image = image;
    }
}


DoorInfo::DoorInfo(int id, const char *Name, osg::MatrixTransform *tn, TokenBuffer &tb)
{
	ID = id;
	name = Name;
	transformNode = tn;
	tb >> HandFlipped;
	tb >> HandOrientation;
	tb >> FaceFlipped;
	tb >> FaceOrientation;
	tb >> Origin;
	int dir;
	tb >> dir;
	isSliding = SlidingDirection(dir);
	osg::Vec3 BBMin;
	osg::Vec3 BBMax;
	tb >> BBMin;
	tb >> BBMax;
	tb >> maxDistance;
	boundingBox.set(BBMin, BBMax);
	if (isSliding != SlidingDirection::dirNone)
	{
		if(maxDistance == 0)
		    maxDistance = boundingBox.xMax() - boundingBox.xMin();
		if (maxDistance == 0)
			maxDistance = 1;
		HandOrientation.normalize(); // HandOrientation is in Revit World coordinates so either transform this back to local coordinates orjust use X for now.
		Direction = osg::Vec3(1, 0, 0)*maxDistance;
		if (strncmp(Name, "DoorMovingParts_Right", 21) == 0)
		{
			//Direction *= -1;
		}
		else if(strncmp(Name, "DoorMovingParts_Left", 20) == 0)
		{
			Direction *= -1;
		}
		else
		{
			Direction *= isSliding;
		}
	//	if (!HandFlipped)
	//	{
	//		Direction *= -1;
	//	}
	}
	else
	{
		Direction = HandOrientation ^ FaceOrientation;
		Direction.normalize();
		maxDistance = M_PI_2;
		//fprintf(stderr, "HandFlipped %d\n", (int)HandFlipped);
		//fprintf(stderr, "FaceFlipped %d\n", (int)FaceFlipped);

		Direction *= -1;
		if(Origin.x() == 10000)
		{
			Origin = boundingBox._max;
			if (strncmp(Name, "DoorMovingParts_Right", 21) == 0)
			{
				Origin = boundingBox._min;
			}
		}
		if (strncmp(Name, "DoorMovingParts_Right", 21) == 0)
		{
			Direction *= -1;
		}
	}

	Center = boundingBox.center();
	// transform Center to Object Coordinates
	osg::Matrix tr;
	tr.makeIdentity();
	//cerr << "LocalToVRML: hitPoint: "<<hitPoint[0]<<' '<<hitPoint[1]<<' '<<hitPoint[2]<<endl;
	const osg::Node *parent = transformNode->getParent(0);
	while (parent != NULL && parent != cover->getObjectsRoot())
	{
		osg::Matrix dcsMat;
		const osg::MatrixTransform *mtParent = dynamic_cast<const osg::MatrixTransform *>(parent);
		if (mtParent)
		{
			dcsMat = mtParent->getMatrix();
			tr.postMult(dcsMat);
		}
		if (parent->getNumParents())
			parent = parent->getParent(0);
		else
			parent = NULL;
	}
	Center = Center * tr;
	activationDistance2 = 3; // three m
	activationDistance2 *= activationDistance2; // squared
	animationTime = 1.0; // one second
	isActive = false;
	left = false;
	entered = false;
}

void DoorInfo::checkStart(osg::Vec3 &viewerPosition)
{
	if (!isActive)
	{
		if ((Center - viewerPosition).length2() < activationDistance2)
		{
			isActive = true;
			RevitPlugin::instance()->activeDoors.push_back(this);
			entered = true;
			left = false;
			startTime = cover->frameTime();
		}
	}
}

void DoorInfo::translateDoor(float fraction)
{
	if (isSliding)
	{
		transformNode->setMatrix(osg::Matrix::translate(Direction*maxDistance*fraction));
	}
	else
	{
		transformNode->setMatrix(osg::Matrix::translate(-Origin) * osg::Matrix::rotate(maxDistance*fraction, Direction)*osg::Matrix::translate(Origin));
	}
}

bool DoorInfo::update(osg::Vec3 &viewerPosition)
{
	if (!left && ((Center - viewerPosition).length2() > activationDistance2))
	{
		left = true;
		startTime = cover->frameTime();
	}
	if (left)
	{
		if (cover->frameTime() < startTime + animationTime)
		{
			float fraction = 1.0 - ((cover->frameTime() - startTime) / animationTime);
			translateDoor(fraction);
		}
		else
		{
			translateDoor(0.0);
			left = false;
			isActive = false;
			return false;
		}
	}
	if (entered)
	{
		if (cover->frameTime() < startTime + animationTime)
		{
			float fraction = (cover->frameTime() - startTime) / animationTime;
			translateDoor(fraction);
		}
		else
		{
			translateDoor(1.0);
			entered = false;
		}
	}
	return true;
}


RevitRoomInfo::RevitRoomInfo(osg::Vec3 pos, std::string n, int id, int docID, double a)
{
	name = n;
	textPosition = pos;
	ID = id;
	documentID = docID;
	area = a;
	label = nullptr;
	updateBillboard();
}
RevitRoomInfo::~RevitRoomInfo()
{
	delete label;
}
void RevitRoomInfo::updateBillboard()
{
	osg::Vec4 fgcolor(0.0f, 1.0f, 0.0f, 1.0f);
	float fontSize = 0.02f * cover->getSceneSize();
	delete label;
	label = new coVRLabel(name.c_str(), 1, 1, fgcolor,VRViewer::instance()->getBackgroundColor());
	label->setDepthScale(false);
	label->reAttachTo(RevitPlugin::instance()->getCurrentGroup());
	label->setPosition(textPosition);
	if (RevitPlugin::instance()->toggleRoomLabels->state())
		label->show();
	else
		label->hide();
}
