/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */


#include "VehicleUtils.h"

#include <cover/coVRMSController.h>

/** Visitor that calculates the Boundingbox. */

/** Constructor.
*/
calculateBoundingBoxVisitor::calculateBoundingBoxVisitor()
    : osg::NodeVisitor(TRAVERSE_ALL_CHILDREN)
{
    boundingbox_.init();
    transformationMatrix_.makeIdentity();
}

/** Traverse Geode.
 * First a boundingbox of the drawables will be calucated. Then
 * the boundingbox's corner coordinates will be transformed to
 * global coordinates and the total boundingbox will be updated.
 * Note that this is not 100% accurate if the local boundingbox
 * is not optimal!!!
 * Geodes with names containing "coShadow" are ignored!
 * TO IMPLEMENT: other nodes (LOD,...)
*/
void
calculateBoundingBoxVisitor::apply(osg::Geode &geode)
{
    // 	std::cout << geode.getName() << std::endl;
    if (geode.getName().find("coShadow") == std::string::npos)
    {
        // local boundingbox
        osg::BoundingBox bbox_local = geode.getBoundingBox();

        // global boundingbox
        osg::BoundingBox bbox_global;
        bbox_global.init();
        for (int i = 0; i < 8; ++i)
        {
            osg::Vec3 corner_global = transformationMatrix_.preMult(bbox_local.corner(i));
            bbox_global.expandBy(corner_global);
        }

        // update total boundingbox
        boundingbox_.expandBy(bbox_global);
    }
    traverse(geode); // continue
}

/** Traverse other nodes.
*/
// not tested yet:
// void
// 	calculateBoundingBoxVisitor
// 	::apply(osg::Node &node)
// {
// 	traverse(node); // continue
// }

/** Traverse MatrixTransform.
*/
void
calculateBoundingBoxVisitor::apply(osg::MatrixTransform &node)
{
    osg::Matrix temp = transformationMatrix_;
    transformationMatrix_.preMult(node.getMatrix());
    traverse(node); // continue
    transformationMatrix_ = temp;
}

/** End: calculateBoundingBoxVisitor */

/** Visitor to find nodes with a specific name.
*/

/** Constructor */
findNodesByNameVisitor::findNodesByNameVisitor()
    : osg::NodeVisitor(TRAVERSE_ALL_CHILDREN)
    , searchForName()
    , lengthNodeList(0)
{
}

findNodesByNameVisitor::findNodesByNameVisitor(const std::string &searchName)
    : osg::NodeVisitor(TRAVERSE_ALL_CHILDREN)
    , searchForName(searchName)
    , lengthNodeList(0)
{
}

/** Traverse Node.
 * Traverses the nodes. If a node with a matching name is found
 * the node will be appended to the foundNodeList.
*/
void
findNodesByNameVisitor::apply(osg::Node &searchNode)
{
    //	if (searchNode.getName() == searchForName)									// exact match
    //	if (searchNode.getName().find(searchForName) != std::string::npos)	// contains string
    if (searchNode.getName().find(searchForName) == 0) // starts with string
    {
        // 		std::cout << searchNode.getName() << " contains " << searchForName << " at " << searchNode.getName().find(searchForName) << std::endl;
        foundNodeList.push_back(&searchNode);
        lengthNodeList++;
    }

    traverse(searchNode);
}

/** Set the searchForName to user-defined string (+reset).
 */
void
findNodesByNameVisitor::setNameToFind(const std::string &searchName)
{
    searchForName = searchName;
    foundNodeList.clear();
    lengthNodeList = 0;
}

/** Return first node.
*/
osg::Node *
findNodesByNameVisitor::getFirst()
{
    if (lengthNodeList)
        return *(foundNodeList.begin());
    else
        return NULL;
}

/** End: findNodesByNameVisitor */

double RoadTransitionList::getLength()
{
    double l = 0;
    for (RoadTransitionList::iterator listIt = begin(); listIt != end(); ++listIt)
    {
        l += listIt->road->getLength();
    }
    return l;
}

double RoadTransitionList::getLength(RoadTransitionList::iterator fromIt, RoadTransitionList::iterator toIt)
{
    double l = 0;
    for (; fromIt != toIt; ++fromIt)
    {
        l += fromIt->road->getLength();
    }
    return l;
}

RouteFindingNode::RouteFindingNode(RoadTransition trans, const RoadTransition &goalTrans, const Vector2D &goalPos, RouteFindingNode *p)
    : endJunction(NULL)
    , sj(0.0)
    , sh(0.0)
    , endNode(false)
    , goalFound(false)
    , parent(p)
{
    if (p)
    {
        sj = p->getDistanceToNode();
    }

    transList.push_back(trans);
    if (trans == goalTrans)
    {
        goalFound = true;
        return;
    }
    trans.junction = NULL;

    while (!endJunction)
    {
        TarmacConnection *conn = (trans.direction < 0) ? trans.road->getPredecessorConnection() : trans.road->getSuccessorConnection();
        Tarmac *tarmac = NULL;
        if (conn)
        {
            tarmac = conn->getConnectingTarmac();
        }

        Road *nextRoad = dynamic_cast<Road *>(tarmac);
        if (nextRoad)
        {
            trans.road = nextRoad;
            trans.direction = conn->getConnectingTarmacDirection();
            transList.push_back(trans);
            if (trans == goalTrans)
            {
                goalFound = true;
                break;
            }
            sj += trans.road->getLength();
        }
        else if ((endJunction = dynamic_cast<Junction *>(tarmac)))
        {
            double endS = (trans.direction < 0) ? 0.0 : trans.road->getLength();
            Vector3D endPos3D = trans.road->getChordLinePlanViewPoint(endS);
            Vector2D endPos(endPos3D.x(), endPos3D.y());
            sh = (goalPos - endPos).length();

            endNode = false;
            break;
        }
        else
        {
            double endS = (trans.direction < 0) ? 0.0 : trans.road->getLength();
            Vector3D endPos3D = trans.road->getChordLinePlanViewPoint(endS);
            Vector2D endPos(endPos3D.x(), endPos3D.y());
            sh = (goalPos - endPos).length();

            endNode = true;
            break;
        }
    }
}
RouteFindingNode::~RouteFindingNode()
{
    for (std::list<RouteFindingNode *>::iterator childIt = childrenList.begin(); childIt != childrenList.end(); ++childIt)
    {
        delete (*childIt);
    }
}
void RouteFindingNode::addChild(RouteFindingNode *n)
{
    childrenList.push_back(n);
}
Junction *RouteFindingNode::getEndJunction()
{
    return endJunction;
}
const RoadTransition &RouteFindingNode::getLastTransition()
{
    return transList.back();
}
RouteFindingNode *RouteFindingNode::getParent()
{
    return parent;
}
bool RouteFindingNode::isEndNode()
{
    return endNode;
}
bool RouteFindingNode::foundGoal()
{
    return goalFound;
}
double RouteFindingNode::getDistanceToNode()
{
    return sj;
}
double RouteFindingNode::getHeuristicNodeToGoal()
{
    return sh;
}
double RouteFindingNode::getCost()
{
    return sj + sh;
}
RoadTransitionList &RouteFindingNode::backchainTransitionList()
{
    if (parent)
    {
        transList.splice(transList.begin(), parent->backchainTransitionList());
    }
    else
    {
        transList.erase(transList.begin());
    }

    return transList;
}
