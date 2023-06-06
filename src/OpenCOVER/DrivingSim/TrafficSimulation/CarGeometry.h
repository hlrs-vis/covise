/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef CarGeometry_h
#define CarGeometry_h

#include "VehicleGeometry.h"
#include "VehicleUtils.h"

#include <osg/Node>
#include <osg/MatrixTransform>
#include <osg/BoundingBox>
#include <vector>

#include <VehicleUtil/RoadSystem/Types.h>

namespace TrafficSimulation
{
    class TRAFFICSIMULATIONEXPORT CarGeometry : public VehicleGeometry
    {
    public:
        CarGeometry(CarGeometry*, std::string, osg::Group* rootNode = NULL);
        CarGeometry(std::string = "no name", std::string = "cars/hotcar.osg", bool = true, osg::Group* rootNode = NULL);
        ~CarGeometry();

        void setTransform(vehicleUtil::Transform&, double);
        void setTransformOrig(vehicleUtil::Transform&, double);

        void setTransformByCoordinates(osg::Vec3& pos, osg::Vec3& xVec);
        void setTransform(osg::Matrix m);
        osg::MatrixTransform *getTransform();

        double getBoundingCircleRadius();
        osg::BoundingBox& getBoundingBox()
        {
            return boundingbox_;
        }

        const osg::Matrix& getVehicleTransformMatrix();

        osg::Node* getCarNode()
        {
            return carNode;
        }
        std::map<std::string, osg::Switch*> getCarPartsSwitchList()
        {
            return carPartsSwitchList;
        }
        std::map<std::string, osg::MatrixTransform*> getCarPartsTransformList()
        {
            return carPartsTransformList;
        }
        void updateCarParts(double t, double dt, VehicleState& vehState);

        void setLODrange(double);

        void removeFromSceneGraph();

    protected:
        static bool fileExist(const char* fileName);
        static std::map<std::string, osg::Node*> filenameNodeMap;

        osg::MatrixTransform* carTransform;

        static osg::Node* getCarNode(std::string);
        static osg::Node* getDefaultCarNode();

        osg::Node* carNode;
        osg::Group* parentGroup = nullptr;
        osg::LOD* carLOD;
        osg::MatrixTransform* carParts;
        std::map<std::string, osg::Switch*> carPartsSwitchList;
        std::map<std::string, osg::MatrixTransform*> carPartsTransformList;
        osg::MatrixTransform* wheelFL;
        osg::MatrixTransform* wheelFR;
        osg::MatrixTransform* wheelBL;
        osg::MatrixTransform* wheelBR;

        bool separateWheel(std::string name);
        bool separateSwitchGeo(std::string nameON, std::string nameOFF);

        double boundingCircleRadius;
        osg::BoundingBox boundingbox_;
        VehicleState currentState;
    };
}

#endif
