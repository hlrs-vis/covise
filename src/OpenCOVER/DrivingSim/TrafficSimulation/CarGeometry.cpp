/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "CarGeometry.h"

#include <cover/coVRPluginSupport.h>
#include <cover/coVRFileManager.h>
#include <cover/coVRMSController.h>

#include <osg/Transform>
#include <osg/Switch>
#include <osg/Node>
#include <osg/Group>
#include <osg/LOD>
#include <osgDB/ReadFile>

using namespace covise;
using namespace opencover;
using namespace TrafficSimulation;

//#include <math.h>
#define PI 3.141592653589793238

std::map<std::string, osg::Node *> CarGeometry::filenameNodeMap;

CarGeometry::CarGeometry(CarGeometry *geo, std::string name, osg::Group* rootNode)
    : carNode(NULL)
    , carParts(NULL)
    , boundingCircleRadius(0.0)
	
{
    // new instance of already loaded car geometry //

    wheelFL = NULL;
    wheelFR = NULL;
    wheelBL = NULL;
    wheelBR = NULL;
    // TRANSFORMATION //
    //
    carTransform = new osg::MatrixTransform();
    carTransform->setName(name);
    // no need to do intersection tests on cars
    carTransform->setNodeMask(carTransform->getNodeMask() & ~(Isect::Intersection | Isect::Collision | Isect::Walk));
	if(geo->parentGroup)
		parentGroup = geo->parentGroup;
	if (rootNode != nullptr)
		parentGroup = rootNode;
	if (parentGroup != nullptr)
	{
		parentGroup->addChild(carTransform);
	}
	else
	{
		cover->getObjectsRoot()->addChild(carTransform);
	}

    // LOD //
    //
    carLOD = new osg::LOD();
    carLOD->setRange(0, 0.0, 3.4e38); // initial value: infinity
    carLOD->setRange(1, 0.0, 3.4e38);
    carTransform->addChild(carLOD);

    // CAR BODY //
    //
    // only reference, no copy
    carNode = geo->getCarNode();
    if (carNode)
    {
        boundingbox_ = geo->getBoundingBox();
        carLOD->addChild(carNode);
    }

    // INDIVIDUAL PARTS //
    //
    // copied parts, e.g. wheels, indicators,...
    osg::Matrix VRMLRootMat;
    VRMLRootMat.makeRotate(M_PI / 2.0, osg::Vec3(1.0, 0.0, 0.0));

    carParts = new osg::MatrixTransform();
    carParts->setMatrix(VRMLRootMat);
    carParts->setName("carParts");
    carLOD->addChild(carParts);

    carPartsSwitchList = geo->getCarPartsSwitchList();
    if (!carPartsSwitchList.empty())
    {
        // clone all children //
        for (std::map<std::string, osg::Switch *>::iterator entry = carPartsSwitchList.begin(); entry != carPartsSwitchList.end(); ++entry)
        {
            if ((*entry).second)
            {
                osg::Object *clonedObject = (*entry).second->clone(osg::CopyOp::DEEP_COPY_ALL);
                (*entry).second = NULL;
                osg::Switch *clonedSwitch = dynamic_cast<osg::Switch *>(clonedObject);
                if (clonedSwitch)
                {
                    (*entry).second = clonedSwitch;
                    carParts->addChild(clonedSwitch);
                }
            }
        }
    }
    carPartsTransformList = geo->getCarPartsTransformList();
    if (!carPartsTransformList.empty())
    {
        // clone all children //
        for (std::map<std::string, osg::MatrixTransform *>::iterator entry = carPartsTransformList.begin(); entry != carPartsTransformList.end(); ++entry)
        {
            if ((*entry).second)
            {
                osg::Object *clonedObject = (*entry).second->clone(osg::CopyOp::DEEP_COPY_ALL);
                (*entry).second = NULL;
                osg::MatrixTransform *clonedMTrafo = dynamic_cast<osg::MatrixTransform *>(clonedObject);
                if (clonedMTrafo)
                {
                    (*entry).second = clonedMTrafo;
                    carParts->addChild(clonedMTrafo);
                }
            }
        }
    }

    if (carPartsTransformList.find("wheelFL") != carPartsTransformList.end())
        wheelFL = carPartsTransformList["wheelFL"];
    if (carPartsTransformList.find("wheelFR") != carPartsTransformList.end())
        wheelFR = carPartsTransformList["wheelFR"];
    if (carPartsTransformList.find("wheelBL") != carPartsTransformList.end())
        wheelBL = carPartsTransformList["wheelBL"];
    if (carPartsTransformList.find("wheelBR") != carPartsTransformList.end())
        wheelBR = carPartsTransformList["wheelBR"];

    // bounding radius //
    osg::BoundingSphere boundingSphere = carTransform->computeBound();
    boundingCircleRadius = (boundingSphere.center() - carTransform->getMatrix().getTrans()).length() + boundingSphere.radius();
}

CarGeometry::CarGeometry(std::string name, std::string file, bool addToSceneGraph, osg::Group* rootNode)
    : carNode(NULL)
    , carParts(NULL)
    , boundingCircleRadius(0.0)
{
    // load car geometry file for the first time //

    wheelFL = NULL;
    wheelFR = NULL;
    wheelBL = NULL;
    wheelBR = NULL;
    // TRANSFORMATION //
    //
    carTransform = new osg::MatrixTransform();
	if (rootNode != nullptr)
		parentGroup = rootNode;
    if (addToSceneGraph)
    {
		
		if (parentGroup != nullptr)
		{
			parentGroup->addChild(carTransform);
		}
		else
		{
			cover->getObjectsRoot()->addChild(carTransform);
		}
        // no need to do intersection tests on cars
        carTransform->setNodeMask(carTransform->getNodeMask() & ~(Isect::Intersection | Isect::Collision | Isect::Walk | Isect::Update));
    }
    carTransform->setName(name);

    // LOD //
    //
    carLOD = new osg::LOD();
    carLOD->setRange(0, 0.0, 3.4e38); // initial value: infinity
    carLOD->setRange(1, 0.0, 3.4e38);
    carTransform->addChild(carLOD);

    // CAR BODY //
    //
    // only reference, no copy
    carNode = getCarNode(file);
    if (carNode)
    {
        // calculate boundingbox
        calculateBoundingBoxVisitor bboxVisitor;
        carNode->accept(bboxVisitor);
        boundingbox_ = bboxVisitor.getBoundingBox();

        // append geometry to LOD node
        carLOD->addChild(carNode);
    }

    // search for indicators, etc, copied //
    osg::Matrix VRMLRootMat;
    VRMLRootMat.makeRotate(M_PI / 2.0, osg::Vec3(1.0, 0.0, 0.0));

    carParts = new osg::MatrixTransform();
    carParts->setMatrix(VRMLRootMat);
    carParts->setName("carParts");
    carLOD->addChild(carParts);

    osg::Group *carNodeGroup = carNode->asGroup();
    if (carNodeGroup)
    {
        separateSwitchGeo("indicatorsLeftON", "indicatorsLeftOFF");
        separateSwitchGeo("indicatorsRightON", "indicatorsRightOFF");

        separateSwitchGeo("brakeLightsON", "brakeLightsOFF");

        separateWheel("wheelFL");
        separateWheel("wheelFR");
        separateWheel("wheelBL");
        separateWheel("wheelBR");

        wheelFL = carPartsTransformList["wheelFL"];
        wheelFR = carPartsTransformList["wheelFR"];
        wheelBL = carPartsTransformList["wheelBL"];
        wheelBR = carPartsTransformList["wheelBR"];
    }

    // bounding radius //
    osg::BoundingSphere boundingSphere = carTransform->computeBound();
    boundingCircleRadius = (boundingSphere.center() - carTransform->getMatrix().getTrans()).length() + boundingSphere.radius();
}

CarGeometry::~CarGeometry()
{
	while(carTransform->getNumParents()>0)
        carTransform->getParent(0)->removeChild(carTransform);
}

void
CarGeometry::setLODrange(double range)
{
    carLOD->setRange(0, 0.0, range);
    carLOD->setRange(1, 0.0, range);
}

bool
CarGeometry::separateWheel(std::string name)
{
    osg::Group *carNodeGroup = carNode->asGroup();
    if (carNodeGroup)
    {
        findNodesByNameVisitor geoVisitor(name);
        carNodeGroup->accept(geoVisitor);
        osg::MatrixTransform *wheel = dynamic_cast<osg::MatrixTransform *>(geoVisitor.getFirst());
        if (wheel)
        {
            carPartsTransformList[name] = wheel;
            carParts->addChild(wheel);
            wheel->getParent(0)->removeChild(wheel); // aus carNode loeschen
            return true;
        }
    }
    return false;
}

bool
CarGeometry::separateSwitchGeo(std::string nameOn, std::string nameOFF)
{
    osg::Group *carNodeGroup = carNode->asGroup();
    if (carNodeGroup)
    {
        findNodesByNameVisitor geoVisitor;

        geoVisitor.setNameToFind(nameOn);
        carNodeGroup->accept(geoVisitor);
        nodeList_t switchNodeListON = geoVisitor.getNodeList();

        geoVisitor.setNameToFind(nameOFF);
        carNodeGroup->accept(geoVisitor);
        nodeList_t switchNodeListOFF = geoVisitor.getNodeList();

        if (!switchNodeListON.empty() && !switchNodeListOFF.empty())
        {
            // ON //
            osg::Switch *switchNodeON = new osg::Switch();
            switchNodeON->setName(nameOn);
            for (nodeList_t::iterator NodeIt = switchNodeListON.begin(); NodeIt != switchNodeListON.end(); ++NodeIt)
            {
                // 			if(node->getNumParents() != 1)  {
                // 				std::cout << "WARNING: Number of Parents not equal 1, Node: " << (*NodeIt)->getName() << std::endl;
                // 				continue;
                // 			}
                switchNodeON->addChild(*NodeIt, true);
                (*NodeIt)->getParent(0)->removeChild(*NodeIt); // aus carNode loeschen
            }

            // OFF //
            osg::Switch *switchNodeOFF = new osg::Switch();
            switchNodeOFF->setName(nameOFF);
            for (nodeList_t::iterator NodeIt = switchNodeListOFF.begin(); NodeIt != switchNodeListOFF.end(); ++NodeIt)
            {
                switchNodeOFF->addChild(*NodeIt, true);
                (*NodeIt)->getParent(0)->removeChild(*NodeIt); // aus carNode loeschen
            }

            // APPEND //
            if (switchNodeOFF->getNumChildren() && switchNodeON->getNumChildren())
            {
                carPartsSwitchList[nameOn] = switchNodeON;
                carParts->addChild(switchNodeON);
                carPartsSwitchList[nameOFF] = switchNodeOFF;
                carParts->addChild(switchNodeOFF);
                return true;
            }
            //else {
            //	std::cout << "NOOOO INDICATORS!!!!!!!!" << std::endl;
            //}
        }
    }
    return false;
}

void
CarGeometry::updateCarParts(double t, double dt, VehicleState &vehState)
{
    // Brakelights //
    //
    // stopping (du) or braking (ddu)
    osg::Switch *switchNode;
    if (vehState.ddu <= -0.2 || vehState.du <= 0.2)
    {
        vehState.brakeLight = true;
    }
    else
    {
        vehState.brakeLight = false;
    }
    if (vehState.brakeLight != currentState.brakeLight)
    {
        if (vehState.brakeLight)
        {
            if ((switchNode = carPartsSwitchList["brakeLightsON"]))
                switchNode->setAllChildrenOn();
            if ((switchNode = carPartsSwitchList["brakeLightsOFF"]))
                switchNode->setAllChildrenOff();
        }
        else
        {
            if ((switchNode = carPartsSwitchList["brakeLightsON"]))
                switchNode->setAllChildrenOff();
            if ((switchNode = carPartsSwitchList["brakeLightsOFF"]))
                switchNode->setAllChildrenOn();
        }
    }

    // Indicators //
    //
    bool flash = ((int(vehState.indicatorTstart + 2.0 * t) % 2) != 0); // flash frequence = 1/2.0s
    if (vehState.indicatorLeft == true)
    {
        if (flash != currentState.flashState)
        {
            if (flash)
            {
                if ((switchNode = carPartsSwitchList["indicatorsLeftON"]))
                    switchNode->setAllChildrenOn();
                if ((switchNode = carPartsSwitchList["indicatorsLeftOFF"]))
                    switchNode->setAllChildrenOff();
            }
            else
            {
                if ((switchNode = carPartsSwitchList["indicatorsLeftON"]))
                    switchNode->setAllChildrenOff();
                if ((switchNode = carPartsSwitchList["indicatorsLeftOFF"]))
                    switchNode->setAllChildrenOn();
            }
        }
    }
    else
    {
        if (vehState.indicatorLeft != currentState.indicatorLeft)
        {
            if ((switchNode = carPartsSwitchList["indicatorsLeftON"]))
                switchNode->setAllChildrenOff();
            if ((switchNode = carPartsSwitchList["indicatorsLeftOFF"]))
                switchNode->setAllChildrenOn();
        }
    }
    if (vehState.indicatorRight == true)
    {
        if (flash != currentState.flashState)
        {
            if (flash)
            {
                if ((switchNode = carPartsSwitchList["indicatorsRightON"]))
                    switchNode->setAllChildrenOn();
                if ((switchNode = carPartsSwitchList["indicatorsRightOFF"]))
                    switchNode->setAllChildrenOff();
            }
            else
            {
                if ((switchNode = carPartsSwitchList["indicatorsRightON"]))
                    switchNode->setAllChildrenOff();
                if ((switchNode = carPartsSwitchList["indicatorsRightOFF"]))
                    switchNode->setAllChildrenOn();
            }
        }
    }
    else
    {
        if (vehState.indicatorRight != currentState.indicatorRight)
        {
            if ((switchNode = carPartsSwitchList["indicatorsRightON"]))
                switchNode->setAllChildrenOff();
            if ((switchNode = carPartsSwitchList["indicatorsRightOFF"]))
                switchNode->setAllChildrenOn();
        }
    }

    currentState = vehState;
    currentState.flashState = flash;

// Wheels //
//
//! TODO: als parameter definierbar machen
#define WHEEL_RADIUS 0.3

    double phi = vehState.du / WHEEL_RADIUS * dt; // w*t = v/r * t
    osg::Matrix addPhi;
    addPhi.makeRotate(-phi, osg::Vec3(0.0, 0.0, 1.0)); // (VRML axis)
    if (wheelFL != NULL )
    {
        wheelFL->preMult(addPhi);
    }
    if (wheelFR != NULL)
    {
        wheelFR->preMult(addPhi);
    }
    if (wheelBL != NULL)
    {
        wheelBL->preMult(addPhi);
    }
    if (wheelBR != NULL)
    {
        wheelBR->preMult(addPhi);
    }
}

void CarGeometry::setTransform(vehicleUtil::Transform &roadTransform, double heading)
{
    osg::Matrix m;
    m.makeRotate(heading, 0, 0, 1);
    m.setTrans(roadTransform.v().x(), roadTransform.v().y(), roadTransform.v().z());
    carTransform->setMatrix(m);
}

void CarGeometry::setTransformOrig(vehicleUtil::Transform &roadTransform, double heading)
{
    vehicleUtil::Quaternion qzaaa(heading, vehicleUtil::Vector3D(0, 0, 1));

    vehicleUtil::Quaternion q = roadTransform.q() * qzaaa;
    osg::Matrix m;
    m.makeRotate(osg::Quat(q.x(), q.y(), q.z(), q.w()));
    m.setTrans(roadTransform.v().x(), roadTransform.v().y(), roadTransform.v().z());
    carTransform->setMatrix(m);
}

void CarGeometry::setTransformByCoordinates(osg::Vec3 &pos, osg::Vec3 &xVec)
{
	osg::Matrix m;
	osg::Vec3 up(0,0,1);
	osg::Vec3 right;
	right = up^xVec;//Kreuzprodukt
	xVec.normalize();
	right.normalize();
	m(0,0)=xVec[0];m(0,1)=xVec[1];m(0,2)=xVec[2];
	m(1,0)=right[0];m(1,1)=right[1];m(1,2)=right[2];
	m(2,0)=up[0];m(2,1)=up[1];m(2,2)=up[2];
    m.setTrans(pos);
    carTransform->setMatrix(m);
}

void CarGeometry::setTransform(osg::Matrix m)
{
	carTransform->setMatrix(m);
}

osg::MatrixTransform *CarGeometry::getTransform()
{
	return carTransform;
}

double CarGeometry::getBoundingCircleRadius()
{
    return boundingCircleRadius;
}

const osg::Matrix &CarGeometry::getVehicleTransformMatrix()
{
    return carTransform->getMatrix();
}

bool CarGeometry::fileExist(const char *fileName)
{
    FILE *file;
    file = ::fopen(fileName, "r");
    //delete name;
    if (file)
    {
        ::fclose(file);
        return true;
    }
    return false;
}

osg::Node *CarGeometry::getCarNode(std::string file)
{
    std::map<std::string, osg::Node *>::iterator nodeMapIt = filenameNodeMap.find(file);
    if (nodeMapIt == filenameNodeMap.end())
    {
        osg::Node *carNode = NULL;
        osg::Group *carGroup = new osg::Group(); //carGroup is argument for coVRFileManager::instance()->loadFile() so that the loaded osg::Node isn't hooked to the opencover root node automatically...

        if (fileExist(file.c_str()))
        {
            carNode = coVRFileManager::instance()->loadFile(file.c_str(), NULL, carGroup);
        }
        else
        {
            std::cerr << "CarGeometry::getCarNode(): File not YET found: " << file << "..." << std::endl;
        }

        /*if(!carNode) {
         const char *filename = coVRFileManager::instance()->getName((std::string("share/covise/materials/vehicles/") + file).c_str());
         if(filename) {
            carNode = coVRFileManager::instance()->loadFile(filename, NULL, carGroup);
         }
         else {
            std::cerr << "CarGeometry::getCarNode(): File not found: " << filename << "..." << std::endl;
            return NULL;
         }
      }*/
        if (!carNode)
        {
            carNode = carGroup;
        }

        if (carNode)
        {
            filenameNodeMap.insert(std::pair<std::string, osg::Node *>(file, carNode));
            for (int i = 0; i < file.length(); i++)
            {
                if (file[i] == '/')
                    file[i] = '_';
                if (file[i] == '.')
                    file[i] = '_';
                if (file[i] == '\\')
                    file[i] = '_';
            }
            carNode->setName(file);
            return carNode;
        }
        else
        {
            std::cerr << "CarGeometry::getCarNode(): Couldn't load file: " << file << "..." << std::endl;
            return NULL;
        }
    }
    else
    {
        return nodeMapIt->second;
    }
}

void CarGeometry::removeFromSceneGraph()
{
	while (carTransform->getNumParents() > 0)
		carTransform->getParent(0)->removeChild(carTransform);
}

osg::Node *CarGeometry::getDefaultCarNode()
{
    return getCarNode("cars/warthog.osg");
}
