/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "Road.h"
#include "Junction.h"

#include <algorithm>
#include <map>

#include <osg/Geometry>
#include <osgViewer/Viewer>
#include <osg/StateAttribute>
#include <osg/StateSet>
#include <osg/PolygonMode>

Road::Road(std::string id, std::string name, double l, Junction *junc)
    : Tarmac(id, name)
{
    length = l;
    junction = junc;

    xyMap[0.0] = (new PlaneStraightLine(0.0));
    zMap[0.0] = (new Polynom(0.0));
    laneSectionMap[0.0] = (new LaneSection(0.0));
    lateralProfileMap[0.0] = (new SuperelevationPolynom(0.0));
    roadTypeMap[0.0] = UNKNOWN;

    predecessor = NULL;
    successor = NULL;
    leftNeighbour = NULL;
    leftNeighbourDirection = -1;
    rightNeighbour = NULL;
    rightNeighbourDirection = 1;

    roadGeode = NULL;
}

void Road::addRoadType(double s, RoadType rt)
{
    roadTypeMap[s] = rt;
}

void Road::addRoadType(double s, std::string rt)
{
    if (rt == "rural")
        roadTypeMap[s] = RURAL;
    else if (rt == "motorway")
        roadTypeMap[s] = MOTORWAY;
    else if (rt == "town")
        roadTypeMap[s] = TOWN;
    else if (rt == "lowspeed")
        roadTypeMap[s] = LOWSPEED;
    else if (rt == "pedestrian")
        roadTypeMap[s] = PEDESTRIAN;
    else
        roadTypeMap[s] = UNKNOWN;
}

void Road::setPredecessorConnection(TarmacConnection *tarmacCon)
{
    predecessor = tarmacCon;
}

void Road::setSuccessorConnection(TarmacConnection *tarmacCon)
{
    successor = tarmacCon;
}

void Road::setLeftNeighbour(Road *road, int dir)
{
    leftNeighbour = road;
    leftNeighbourDirection = dir;
    std::cerr << "Use of road neighbours records in OpenDRIVE is obsolete, please don't." << std::endl;
}

void Road::setRightNeighbour(Road *road, int dir)
{
    rightNeighbour = road;
    rightNeighbourDirection = dir;
    std::cerr << "Use of road neighbours records in OpenDRIVE is obsolete, please don't." << std::endl;
}

void Road::setJunction(Junction *junc)
{
    junction = junc;
}

void Road::addPlanViewGeometryLine(double s, double l, double x, double y, double hdg)
{
    xyMap[s] = (new PlaneStraightLine(s, l, x, y, hdg));
}

void Road::addPlanViewGeometrySpiral(double s, double l, double x, double y, double hdg, double curvStart, double curvEnd)
{
    xyMap[s] = (new PlaneClothoid(s, l, x, y, hdg, curvStart, curvEnd));
}

void Road::addPlanViewGeometryArc(double s, double l, double x, double y, double hdg, double curv)
{
    xyMap[s] = (new PlaneArc(s, l, x, y, hdg, 1 / curv));
}

void Road::addPlanViewGeometryPolynom(double s, double l, double x, double y, double hdg, double a, double b, double c, double d)
{
    xyMap[s] = (new PlanePolynom(s, l, x, y, hdg, a, b, c, d));
}

void Road::addElevationPolynom(double s, double a, double b, double c, double d)
{
    zMap[s] = (new Polynom(s, a, b, c, d));
}

void Road::addSuperelevationPolynom(double s, double a, double b, double c, double d)
{
    lateralProfileMap[s] = (new SuperelevationPolynom(s, a, b, c, d));
}

void Road::addCrossfallPolynom(double s, double a, double b, double c, double d, std::string side)
{
    if (side == "left")
    {
        lateralProfileMap[s] = (new CrossfallPolynom(s, a, b, c, d, -1.0, 0));
    }
    else if (side == "right")
    {
        lateralProfileMap[s] = (new CrossfallPolynom(s, a, b, c, d, 0, 1.0));
    }
    else
    {
        lateralProfileMap[s] = (new CrossfallPolynom(s, a, b, c, d));
    }
}

void Road::addLaneSection(LaneSection *section)
{
    laneSectionMap[section->getStart()] = section;
}

LaneSection *Road::getLaneSection(double s)
{
    /*std::cout << id << ": Lane sections: ";
   for(std::map<double, LaneSection*>::iterator lsIt = laneSectionMap.begin(); lsIt!=laneSectionMap.end(); ++lsIt) {
      std::cout << "\t " << lsIt->second << "(" << lsIt->second->getStart() << ")";
   }
   std::cout << std::endl;*/

    return (--laneSectionMap.upper_bound(s))->second;
}

double Road::getLength()
{
    return length;
}

double Road::getHeading(double s)
{
    return ((--xyMap.upper_bound(s))->second)->getOrientation(s);
}

RoadPoint Road::getChordLinePoint(double s)
{
    Vector3D xyPoint(((--xyMap.upper_bound(s))->second)->getPoint(s));
    double zValue = (((--zMap.upper_bound(s))->second)->getValue(s));
    return RoadPoint(xyPoint.x(), xyPoint.y(), zValue);
}

RoadPoint Road::getRoadPoint(double s, double t)
{
    //RoadPoint center = getChordLinePoint(s);
    Vector3D xyPoint = ((--xyMap.upper_bound(s))->second)->getPoint(s);
    Vector2D zPoint = ((--zMap.upper_bound(s))->second)->getPoint(s);

    //std::cerr << "Center: x: " << center.x() << ", y: " << center.y() << ", z: " << center.z() << std::endl;

    double alpha = ((--lateralProfileMap.upper_bound(s))->second)->getAngle(s, t);
    double beta = zPoint[1];
    double gamma = xyPoint[2];
    //std::cerr << "s: " << s << "Alpha: " << alpha << ", Beta: " << beta << ", Gamma: " << gamma << std::endl;
    double sinalpha = sin(alpha);
    double cosalpha = cos(alpha);
    double sinbeta = sin(beta);
    double cosbeta = cos(beta);
    double singamma = sin(gamma);
    double cosgamma = cos(gamma);

    return RoadPoint(xyPoint.x() + (sinalpha * sinbeta * cosgamma - cosalpha * singamma) * t,
                     xyPoint.y() + (sinalpha * sinbeta * singamma + cosalpha * cosgamma) * t,
                     zPoint[0] + (sinalpha * cosbeta) * t,
                     sinalpha * singamma + cosalpha * sinbeta * cosgamma,
                     cosalpha * sinbeta * singamma - sinalpha * cosgamma,
                     cosalpha * cosbeta);
}

void Road::getLaneRoadPoints(double s, int i, RoadPoint &pointIn, RoadPoint &pointOut, double &disIn, double &disOut)
{
    LaneSection *section = (--laneSectionMap.upper_bound(s))->second;
    disIn = section->getDistanceToLane(s, i);
    disOut = disIn + section->getLaneWidth(s, i);
    //std::cerr << "s: " << s << ", i: " << i << ", distance: " << disIn << ", width: " << disOut-disIn << std::endl;

    Vector3D xyPoint = ((--xyMap.upper_bound(s))->second)->getPoint(s);
    Vector2D zPoint = ((--zMap.upper_bound(s))->second)->getPoint(s);

    double alpha = ((--lateralProfileMap.upper_bound(s))->second)->getAngle(s, (disOut + disIn) * 0.5);
    double beta = zPoint[1];
    double gamma = xyPoint[2];
    double Tx = (sin(alpha) * sin(beta) * cos(gamma) - cos(alpha) * sin(gamma));
    double Ty = (sin(alpha) * sin(beta) * sin(gamma) + cos(alpha) * cos(gamma));
    double Tz = (sin(alpha) * cos(beta));
    //std::cout << "s: " << s << ", i: " << i << ", alpha: " << alpha << ", beta: " << beta << ", gamma: " << gamma << std::endl;

    double nx = sin(alpha) * sin(gamma) + cos(alpha) * sin(beta) * cos(gamma);
    double ny = cos(alpha) * sin(beta) * sin(gamma) - sin(alpha) * cos(gamma);
    double nz = cos(alpha) * cos(beta);

    pointIn = RoadPoint(xyPoint.x() + Tx * disIn, xyPoint.y() + Ty * disIn, zPoint[0] + Tz * disIn, nx, ny, nz);
    pointOut = RoadPoint(xyPoint.x() + Tx * disOut, xyPoint.y() + Ty * disOut, zPoint[0] + Tz * disOut, nx, ny, nz);
    //std::cout << "s: " << s << ", i: " << i << ", inner: x: " << pointIn.x() << ", y: " << pointIn.y() << ", z: " << pointIn.z() << std::endl;
    //std::cout << "s: " << s << ", i: " << i << ", outer: x: " << pointOut.x() << ", y: " << pointOut.y() << ", z: " << pointOut.z() << std::endl << std::endl;
}

Transform Road::getRoadTransform(const double &s, const double &t)
{
    Vector3D xyPoint = ((--xyMap.upper_bound(s))->second)->getPoint(s);
    Vector2D zPoint = ((--zMap.upper_bound(s))->second)->getPoint(s);

    //std::cerr << "Center: x: " << center.x() << ", y: " << center.y() << ", z: " << center.z() << std::endl;

    double alpha = ((--lateralProfileMap.upper_bound(s))->second)->getAngle(s, t);
    //std::cerr << "s: " << s << "Alpha: " << alpha << ", Beta: " << beta << ", Gamma: " << gamma << std::endl;
    //double sinalpha = sin(alpha); double cosalpha=cos(alpha);
    //double sinbeta = sin(beta); double cosbeta=cos(beta);
    //double singamma = sin(gamma); double cosgamma=cos(gamma);

    Quaternion qz(xyPoint[2], Vector3D(0, 0, 1));
    Quaternion qya(zPoint[1], Vector3D(0, 1, 0));
    Quaternion qxaa(alpha, Vector3D(1, 0, 0));

    Quaternion q = qz * qya * qxaa;
    /*
	return RoadTransform(	xyPoint.x() + (sinalpha*sinbeta*cosgamma-cosalpha*singamma)*t,
							xyPoint.y() + (sinalpha*sinbeta*singamma+cosalpha*cosgamma)*t,
							zPoint.z() + (sinalpha*cosbeta)*t,
                     alpha, beta, gamma);
   */

    return Transform(Vector3D(xyPoint.x(), xyPoint.y(), zPoint[0]) + (q * Vector3D(0, t, 0) * q.T()).getVector(), q);
}

Vector2D Road::getLaneCenter(const int &l, const double &s)
{
    if (l != Lane::NOLANE)
    {
        LaneSection *section = (--laneSectionMap.upper_bound(s))->second;
        //std::cout << "getLaneCenter: Road: " << id << ", section: " << section << std::endl;
        return Vector2D(section->getDistanceToLane(s, l) + 0.5 * section->getLaneWidth(s, l), atan(0.5 * section->getLaneWidthSlope(s, l)));
    }
    else
    {
        return Vector2D(0.0, 0.0);
    }
}

Junction *Road::getJunction()
{
    return junction;
}
bool Road::isJunctionPath()
{
    return (junction == NULL) ? false : true;
}

TarmacConnection *Road::getPredecessorConnection()
{
    return predecessor;
}

TarmacConnection *Road::getSuccessorConnection()
{
    return successor;
}

osg::Geode *Road::getRoadGeode()
{
    return (roadGeode == NULL) ? createRoadGeode() : roadGeode;
}

osg::Geode *Road::createRoadGeode()
{
    roadGeode = new osg::Geode();

    //int n = 100;
    //double h = length/n;
    double h = 0.5;
    double texlength = 10.0;
    double texwidth = 10.0;

    std::map<double, LaneSection *>::iterator lsIt;
    LaneSection *ls;
    for (lsIt = laneSectionMap.begin(); lsIt != laneSectionMap.end(); ++lsIt)
    {
        double lsStart;
        double lsEnd;
        std::map<double, LaneSection *>::iterator nextLsIt = lsIt;
        ++nextLsIt;
        ls = (*lsIt).second;
        lsStart = ls->getStart();

        if (lsStart > length)
        {
            std::cerr << "Inconsistency: Lane Section defined with start position " << lsStart << " not in road boundaries 0 - " << lsEnd << "!" << std::endl;
            break;
        }
        else if (nextLsIt == laneSectionMap.end())
        {
            lsEnd = length;
        }
        else
        {
            lsEnd = (*(nextLsIt)).second->getStart();
            if (lsEnd > length)
                lsEnd = length;
        }

        osg::Geometry *roadGeometry;
        roadGeometry = new osg::Geometry();
        roadGeode->addDrawable(roadGeometry);

        osg::Vec3Array *roadVertices;
        roadVertices = new osg::Vec3Array;
        roadGeometry->setVertexArray(roadVertices);

        osg::Vec3Array *roadNormals;
        roadNormals = new osg::Vec3Array;
        roadGeometry->setNormalArray(roadNormals);
        roadGeometry->setNormalBinding(osg::Geometry::BIND_PER_VERTEX);

        osg::Vec2Array *roadTexCoords;
        roadTexCoords = new osg::Vec2Array;
        roadGeometry->setTexCoordArray(0, roadTexCoords);

        osg::Vec4Array *roadMarks;
        roadMarks = new osg::Vec4Array;
        roadGeometry->setVertexAttribArray(6, roadMarks);
        roadGeometry->setVertexAttribBinding(6, osg::Geometry::BIND_PER_VERTEX);

        osg::DrawArrays *roadBase;

        int firstVertLane = 0;
        int numVertsLane = 0;
        for (int i = (ls->getNumLanesLeft()); i >= -(ls->getNumLanesRight()); --i)
        {
            if (i == 0)
                continue;
            bool lastRound = false;
            //std::cout << "Lane: " << i << ", lsStart: " << lsStart << ", lsEnd: " << lsEnd << std::endl;
            for (double s = lsStart; s < (lsEnd + h); s += h)
            {
                if (s > lsEnd)
                {
                    //std::cout << "Last round: s = " << s << ", lsEnd: " << lsEnd << std::endl;
                    s = lsEnd;
                    lastRound = true;
                }
                RoadPoint pointLeft;
                RoadPoint pointRight;
                double disLeft, disRight;
                RoadMark *markLeft;
                RoadMark *markRight;
                if (i < 0)
                {
                    getLaneRoadPoints(s, i, pointLeft, pointRight, disLeft, disRight);
                    markLeft = ls->getLaneRoadMark(s, i + 1);
                    markRight = ls->getLaneRoadMark(s, i);
                }
                else
                {
                    getLaneRoadPoints(s, i, pointRight, pointLeft, disRight, disLeft);
                    markLeft = ls->getLaneRoadMark(s, i);
                    markRight = ls->getLaneRoadMark(s, i - 1);
                }
                double width = fabs(disLeft - disRight);
                //std::cerr << "s: " << s << ", i: " << "Outer mark width: " << markOut->getWidth() << ", inner mark width: " << markIn->getWidth() << std::endl;
                //std::cerr << "s: " << s << ", i: " << i << ", width: " << width << std::endl;
                //std::cout << "s: " << s << ", i: " << i << ", left Point: x: " << pointLeft.x() << ", y: " << pointLeft.y() << ", z: " << pointLeft.z() << std::endl;
                //std::cerr << ", outer Point: x: " << pointOut.x() << ", y: " << pointOut.y() << ", z: " << pointOut.z() << std::endl;
                //std::cout << "s: " << s << ", i: " << i << ", inner Point: x: " << pointIn.nx() << ", y: " << pointIn.ny() << ", z: " << pointIn.nz();
                //std::cout << ", outer Point: x: " << pointOut.nx() << ", y: " << pointOut.ny() << ", z: " << pointOut.nz() << std::endl;

                roadVertices->push_back(osg::Vec3(pointLeft.x(), pointLeft.y(), pointLeft.z()));
                roadNormals->push_back(osg::Vec3(pointLeft.nx(), pointLeft.ny(), pointLeft.nz()));
                roadTexCoords->push_back(osg::Vec2(s / texlength, 0.5 + disLeft / texwidth));
                roadMarks->push_back(osg::Vec4(-markLeft->getWidth() / width, -markRight->getWidth() / width, -markLeft->getType(), -markRight->getType()));

                roadVertices->push_back(osg::Vec3(pointRight.x(), pointRight.y(), pointRight.z()));
                roadNormals->push_back(osg::Vec3(pointRight.nx(), pointRight.ny(), pointRight.nz()));
                roadTexCoords->push_back(osg::Vec2(s / texlength, 0.5 + disRight / texwidth));
                roadMarks->push_back(osg::Vec4(markLeft->getWidth() / width, markRight->getWidth() / width, markLeft->getType(), markRight->getType()));

                numVertsLane += 2;

                if (lastRound)
                    s = lsEnd + h;
            }
            roadBase = new osg::DrawArrays(osg::PrimitiveSet::TRIANGLE_STRIP, firstVertLane, numVertsLane);
            roadGeometry->addPrimitiveSet(roadBase);
            firstVertLane += numVertsLane;
            numVertsLane = 0;
        }
    }

    //osg::StateSet* roadStateSet = roadGeode->getOrCreateStateSet();
    //osg::PolygonMode* polygonMode = new osg::PolygonMode(osg::PolygonMode::FRONT_AND_BACK, osg::PolygonMode::LINE);
    //roadStateSet->setAttributeAndModes(polygonMode);

    roadGeode->setName(this->id);
    return roadGeode;
}
