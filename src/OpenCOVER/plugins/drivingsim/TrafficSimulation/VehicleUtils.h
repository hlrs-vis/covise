/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef VehicleUtils_h
#define VehicleUtils_h

#include "RoadSystem/Road.h"
#include "RoadSystem/Junction.h"
#include "Vehicle.h"
#include <limits>

#include <osg/Node>
#include <osg/NodeVisitor>
#include <osg/BoundingBox>
#include <osg/MatrixTransform>

/** Visitor that calculates the Boundingbox.
 * Traverses all child nodes and calculates a Boundingbox.
 * Supported nodes: MatrixTransform, Geode.
 */
class calculateBoundingBoxVisitor : public osg::NodeVisitor
{
public:
    calculateBoundingBoxVisitor();

    osg::BoundingBox &getBoundingBox()
    {
        return boundingbox_;
    }

    virtual void apply(osg::Geode &geode);
    virtual void apply(osg::MatrixTransform &node);

protected:
    osg::BoundingBox boundingbox_;
    osg::Matrix transformationMatrix_;
};
/** calculateBoundingBoxVisitor */

/** Visitor to find nodes with a specific name.
 * Traverses all child nodes.
 * Supported nodes: all
 */
typedef std::vector<osg::Node *> nodeList_t;

class findNodesByNameVisitor : public osg::NodeVisitor
{
public:
    findNodesByNameVisitor();
    findNodesByNameVisitor(const std::string &searchName);

    void setNameToFind(const std::string &searchName);
    virtual void apply(osg::Node &searchNode);
    //virtual void	apply(osg::Transform &searchNode);

    osg::Node *getFirst();
    nodeList_t &getNodeList()
    {
        return foundNodeList;
    }
    int getListLength()
    {
        return lengthNodeList;
    }

private:
    std::string searchForName;

    nodeList_t foundNodeList;
    int lengthNodeList;
};
/** findNodesByNameVisitor */

/** VehicleState.
 * Todo: The whole vehicle state in one struct or class.
 */
struct VehicleState
{
    VehicleState(double _u = 0.0, double _v = 0.0, double _du = 0.0, double _dv = 0.0, double _ddu = 0.0, double _hdg = 0.0, int _dir = 1)
        : u(_u)
        , v(_v)
        , du(_du)
        , dv(_dv)
        , ddu(_ddu)
        , hdg(_hdg)
        , dir(_dir)
        , indicatorLeft(false)
        , indicatorRight(false)
        , brakeLight(false)
        , flashState(false)
        , indicatorTstart(0.0)
        , junctionState(VehicleState::JUNCTION_NONE)
    {
    }

    double u; // longitudinal position
    double v; // transversal
    double du; // longitudinal speed
    double dv; // transversal
    double ddu; // longitudinal acceleration
    double hdg; // heading
    int dir; // positive or negative direction on street

    bool indicatorLeft; // true if driving to the left
    bool indicatorRight;
    bool brakeLight;
    bool flashState;
    double indicatorTstart; // start time of lane change

    enum junctionState_t
    {
        JUNCTION_LEFT,
        JUNCTION_RIGHT,
        JUNCTION_NONE,
    };
    junctionState_t junctionState;
};
/** VehicleState */

class RoadTransitionList : public std::list<RoadTransition>
{
public:
    double getLength();
    double getLength(RoadTransitionList::iterator, RoadTransitionList::iterator);
};

class RouteFindingNode
{
public:
    RouteFindingNode(RoadTransition, const RoadTransition &, const Vector2D &, RouteFindingNode * = NULL);
    ~RouteFindingNode();

    void addChild(RouteFindingNode *);

    Junction *getEndJunction();
    const RoadTransition &getLastTransition();

    RouteFindingNode *getParent();

    bool isEndNode();
    bool foundGoal();

    double getDistanceToNode();

    double getHeuristicNodeToGoal();

    double getCost();

    RoadTransitionList &backchainTransitionList();

protected:
    RoadTransitionList transList;
    Junction *endJunction;
    double sj;
    double sh;
    bool endNode;
    bool goalFound;

    RouteFindingNode *parent;
    std::list<RouteFindingNode *> childrenList;
};

struct RouteFindingNodeCompare
{
    bool operator()(RouteFindingNode *leftNode, RouteFindingNode *rightNode) const
    {
        return (leftNode->getCost() < rightNode->getCost());
    }
};

struct ObstacleRelation
{
    ObstacleRelation()
        : vehicle(NULL)
        , diffU(std::numeric_limits<float>::signaling_NaN())
        , diffDu(std::numeric_limits<float>::signaling_NaN())
    {
    }

    ObstacleRelation(Vehicle *veh, double dU, double dDu)
        : vehicle(veh)
        , diffU(dU)
        , diffDu(dDu)
    {
    }

    Vehicle *vehicle;
    double diffU;
    double diffDu;

    bool noOR() const
    {
        //if(vehicle==NULL && diffU!=diffU && diffDu!=diffDu) {
        if (vehicle == NULL)
        {
            return true;
        }
        else
        {
            return false;
        }
    }
};

#endif
