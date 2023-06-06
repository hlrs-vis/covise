/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef PedestrianGeometry_h
#define PedestrianGeometry_h

#include "PedestrianFactory.h"
#include "PedestrianUtils.h"
#include "coEntity.h"
#include <cover/coVRPluginSupport.h>
#include <osg/Transform>
#include <osg/Group>
#include <osg/LOD>
#include <osg/MatrixTransform>
#include <osgCal/CoreModel>
#include <osgCal/Model>

namespace TrafficSimulation
{
    struct TRAFFICSIMULATIONEXPORT PedestrianAnimations
    {
        PedestrianAnimations(int _i = 0, double _iv = 0.0f, int _s = 1, double _sv = 0.6f, int _w = 2, double _wv = 1.5f, int _j = 3, double _jv = 3.0f, int _li = 4, int _wi = 5)
            : idleIdx(_i)
            , idleVel(_iv)
            , slowIdx(_s)
            , slowVel(_sv)
            , walkIdx(_w)
            , walkVel(_wv)
            , jogIdx(_j)
            , jogVel(_jv)
            , lookIdx(_li)
            , waveIdx(_wi)
        {
        }

        int idleIdx;
        double idleVel;
        int slowIdx;
        double slowVel;
        int walkIdx;
        double walkVel;
        int jogIdx;
        double jogVel;
        int lookIdx;
        int waveIdx;
    };

    class Pedestrian;
    class TRAFFICSIMULATIONEXPORT PedestrianGeometry : public coEntity
    {
    public:
        PedestrianGeometry(std::string& name, std::string& modelFile, double scale, double lod, const PedestrianAnimations& a, osg::Group* group);
        ~PedestrianGeometry();

        void setPedestrian(Pedestrian* p)
        {
            myPed = p;
        }

        void setTransform(vehicleUtil::Transform&, double);
        osg::MatrixTransform *getTransform() {return pedTransform;}

        bool isGeometryWithinLOD();
        bool isGeometryWithinRange(const double r) const;

        void removeFromSceneGraph();

        void setWalkingSpeed(double speed);

        void update(double dt);

        void executeLook(double factor = 1.0);
        void executeWave(double factor = 1.0);
        void executeAction(int idx, double factor = 1.0);
        bool isActive() { return activeState; };
        void setActive(bool state) { activeState = state; };

    protected:
        bool floatEq(double a, double b);
        bool activeState = true;

        std::string geometryName;

        osg::ref_ptr<osg::Group> pedGroup;
        osg::ref_ptr<osg::MatrixTransform> pedTransform;
        osg::ref_ptr<osg::LOD> pedLOD;

        osg::ref_ptr<osgCal::Model> pedModel;
        osg::ref_ptr<osgCal::BasicMeshAdder> meshAdder;

        PedestrianAnimations anim;
        osg::Matrix mScale;
        double rangeLOD;
        double timeFactorScale;
        double animOffset;

        double currentSpeed;
        double lastSpeed;

        osg::Node::NodeMask mask;

        Pedestrian* myPed;
    };
}

#endif
