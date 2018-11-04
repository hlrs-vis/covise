/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "RoadObject.h"
#include <stdio.h>

#include <osg/Vec2>
#include <osg/Group>
#include <osg/Geometry>
#include <osg/CullFace>
#include <osg/AlphaFunc>
#include <cover/coVRFileManager.h>
#include "Road.h"
#include <osgDB/ReadFile>
#include <cover/coVRConfig.h>

using namespace opencover;

RoadObject::RoadObject(const std::string &setId, const std::string &setFile, const std::string &setTextureFile, const std::string &setName, const std::string &setType, const double &setS,
                       const double &setT, const double &setZOffset, const double &setValidLength, OrientationType setOrientation, const double &setLength,
                       const double &setWidth, const double &setRadius, const double &setHeight, const double &setHdg, const double &setPitch, const double &setRoll, Road *roadP)
    : Element(setId)
    , fileName(setFile)
    , textureFileName(setTextureFile)
    , name(setName)
    , type(setType)
    , s(setS)
    , t(setT)
    , zOffset(setZOffset)
    , validLength(setValidLength)
    , orientation(setOrientation)
    , length(setLength)
    , width(setWidth)
    , radius(setRadius)
    , height(setHeight)
    , hdg(setHdg)
    , pitch(setPitch)
    , roll(setRoll)
    , objectNode(NULL)
    , objectRepeat(false)
    , repeatLength(0.0)
    , repeatDistance(0.0)
    , road(roadP)
    , outline(NULL)
{

    absolute = false;
}

void RoadObject::setObjectRepeat(const double &setS, const double &setRepeatLength, const double &setRepeatDistance)
{
    s = setS;
    repeatLength = setRepeatLength;
    repeatDistance = setRepeatDistance;
    if (setRepeatLength > 0.0)
    {
        objectRepeat = true;
    }
}

osg::Node *RoadObject::getObjectNode()
{
    objectNode = loadObjectGeometry(fileName);
    return objectNode;
}

bool RoadObject::fileExist(const char *fileName)
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

std::map<std::string, osg::Node *> fileMap;

osg::Geode *RoadObject::reflectorPostGeode = NULL;
osg::Geode *RoadObject::reflectorPostYellowGeode = NULL;

osg::Node *RoadObject::loadObjectGeometry(std::string file)
{
    if (outline != NULL)
    {
        absolute = true;
        return createOutlineGeode();
    }
    else if (type == "guardRail")
    {
        absolute = true;
        return createGuardRailGeode();
    }
    /*else if(type == "STEPBarrier")
	{
		absolute = true;
		return createSTEPBarrierGeode();
	}
	else if(type == "JerseyBarrier")
	{
		absolute = true;
		return createJerseyBarrierGeode();
	}*/
    else if (type == "reflectorPost")
    {
        absolute = false;
        if (reflectorPostGeode == NULL)
        {
            reflectorPostGeode = createReflectorPostGeode(type);
            reflectorPostGeode->ref(); // never delete
        }
        return reflectorPostGeode;
    }
    else if (type == "reflectorPostYellow")
    {
        absolute = false;
        if (reflectorPostYellowGeode == NULL)
        {
            reflectorPostYellowGeode = createReflectorPostGeode(type);
            reflectorPostYellowGeode->ref(); // never delete
        }
        return reflectorPostYellowGeode;
    }
    else
    {
        std::string findFile = file;
        for (unsigned int i = 0; i < findFile.length(); i++)
        {
            if (findFile[i] == '/')
                findFile[i] = '_';
            if (findFile[i] == '.')
                findFile[i] = '_';
            if (findFile[i] == '\\')
                findFile[i] = '_';
        }
        osg::Node *localObjectNode = NULL;
        osg::Group *objectGroup = new osg::Group(); //objectGroup is argument for coVRFileManager::instance()->loadFile() so that the loaded osg::Node isn't hooked to the opencover root node automatically...

        std::map<std::string, osg::Node *>::iterator fileId = fileMap.find(findFile);

        if (fileId != fileMap.end())
        {

            //return ( (--fileMap.upper_bound(file))->second );
            return fileId->second;
        }

        else
        {

            if (fileExist(file.c_str()))
            {
                localObjectNode = coVRFileManager::instance()->loadFile(file.c_str(), NULL, objectGroup);
            }
            else
            {
                std::cerr << "RoadObject::getObjectGeometry(): Couldn't load file: " << file << "..." << std::endl;
            }

            if (!localObjectNode)
            {
                localObjectNode = objectGroup;
            }

            localObjectNode->setName(findFile);

            fileMap[findFile] = localObjectNode;
            return localObjectNode;
        }
    }
    return NULL;
}

osg::Geode *RoadObject::createReflectorPostGeode(std::string &textureName)
{
    osg::Geode *postGeode = new osg::Geode();
    postGeode->setName(name.c_str());
    osg::StateSet *postStateSet = postGeode->getOrCreateStateSet();

    postStateSet->setMode(GL_LIGHTING, osg::StateAttribute::ON);
    postStateSet->setMode(GL_BLEND, osg::StateAttribute::OFF);
    osg::CullFace *cullFace = new osg::CullFace();
    cullFace->setMode(osg::CullFace::BACK);
    postStateSet->setAttributeAndModes(cullFace, osg::StateAttribute::ON);

    std::string fn = "share/covise/signals/Germany/" + textureName + ".png";

    const char *fileName = opencover::coVRFileManager::instance()->getName(fn.c_str());
    if (fileName)
    {
        osg::Image *postTexImage = osgDB::readImageFile(fileName);
        osg::Texture2D *postTex = new osg::Texture2D;
        postTex->setWrap(osg::Texture2D::WRAP_S, osg::Texture::CLAMP_TO_EDGE);
        postTex->setWrap(osg::Texture2D::WRAP_T, osg::Texture::CLAMP_TO_EDGE);
        if (postTexImage)
            postTex->setImage(postTexImage);
        postStateSet->setTextureAttributeAndModes(3, postTex, osg::StateAttribute::ON);
    }
    else
    {
        std::cerr << "ERROR: no texture found named: " << fn;
    }

    float hsize = 0.6;
    float vsize = 0.6;

    float front = 0.015;
    float frontHeight = 1.0;
    float backHeight = 1.07;
    float back = 0.04;
    float depth = 0.1;
    osg::Vec3 v[8];
    v[0].set(-front, 0, 0);
    v[1].set(-back, depth, 0);
    v[2].set(back, depth, 0);
    v[3].set(front, 0, 0);
    v[4].set(-front, 0, frontHeight);
    v[5].set(-back, depth, backHeight);
    v[6].set(back, depth, backHeight);
    v[7].set(front, 0, frontHeight);

    osg::Vec3 np[5];
    np[0].set(0, -1, 0);
    np[1].set(0.8, -0.2, 0);
    np[2].set(0, 1, 0);
    np[3].set(-0.8, -0.2, 0);
    np[4].set(0, -0.2, 0.8);

    osg::Vec2 tc[8];
    tc[0].set(0.4, 0);
    tc[1].set(0, 0);
    tc[2].set(1, 0);
    tc[3].set(0.6, 0);
    tc[4].set(0.4, frontHeight);
    tc[5].set(0, backHeight);
    tc[6].set(1, backHeight);
    tc[7].set(0.6, frontHeight);

    osg::Geometry *postGeometry;
    postGeometry = new osg::Geometry();
    postGeometry->setUseDisplayList(coVRConfig::instance()->useDisplayLists());
    postGeometry->setUseVertexBufferObjects(coVRConfig::instance()->useVBOs());
    postGeode->addDrawable(postGeometry);

    postGeometry->setUseDisplayList(true);
    postGeometry->setUseVertexBufferObjects(false);

    osg::Vec3Array *postVertices;
    postVertices = new osg::Vec3Array;
    postGeometry->setVertexArray(postVertices);

    osg::Vec3Array *postNormals;
    postNormals = new osg::Vec3Array;
    postGeometry->setNormalArray(postNormals);
    postGeometry->setNormalBinding(osg::Geometry::BIND_PER_VERTEX);

    osg::Vec2Array *postTexCoords;
    postTexCoords = new osg::Vec2Array;
    postGeometry->setTexCoordArray(3, postTexCoords);

    postVertices->push_back(v[3]);
    postTexCoords->push_back(tc[3]);
    postNormals->push_back(np[0]);
    postVertices->push_back(v[7]);
    postTexCoords->push_back(tc[7]);
    postNormals->push_back(np[0]);
    postVertices->push_back(v[4]);
    postTexCoords->push_back(tc[4]);
    postNormals->push_back(np[0]);
    postVertices->push_back(v[0]);
    postTexCoords->push_back(tc[0]);
    postNormals->push_back(np[0]);

    postVertices->push_back(v[2]);
    postTexCoords->push_back(tc[2]);
    postNormals->push_back(np[1]);
    postVertices->push_back(v[6]);
    postTexCoords->push_back(tc[6]);
    postNormals->push_back(np[1]);
    postVertices->push_back(v[7]);
    postTexCoords->push_back(tc[7]);
    postNormals->push_back(np[1]);
    postVertices->push_back(v[3]);
    postTexCoords->push_back(tc[3]);
    postNormals->push_back(np[1]);

    postVertices->push_back(v[1]);
    postTexCoords->push_back(tc[1]);
    postNormals->push_back(np[2]);
    postVertices->push_back(v[5]);
    postTexCoords->push_back(tc[5]);
    postNormals->push_back(np[2]);
    postVertices->push_back(v[6]);
    postTexCoords->push_back(tc[5]);
    postNormals->push_back(np[2]);
    postVertices->push_back(v[2]);
    postTexCoords->push_back(tc[1]);
    postNormals->push_back(np[2]);

    postVertices->push_back(v[0]);
    postTexCoords->push_back(tc[0]);
    postNormals->push_back(np[3]);
    postVertices->push_back(v[4]);
    postTexCoords->push_back(tc[4]);
    postNormals->push_back(np[3]);
    postVertices->push_back(v[5]);
    postTexCoords->push_back(tc[5]);
    postNormals->push_back(np[3]);
    postVertices->push_back(v[1]);
    postTexCoords->push_back(tc[1]);
    postNormals->push_back(np[3]);

    postVertices->push_back(v[4]);
    postTexCoords->push_back(tc[4]);
    postNormals->push_back(np[4]);
    postVertices->push_back(v[7]);
    postTexCoords->push_back(tc[7]);
    postNormals->push_back(np[4]);
    postVertices->push_back(v[6]);
    postTexCoords->push_back(tc[6]);
    postNormals->push_back(np[4]);
    postVertices->push_back(v[5]);
    postTexCoords->push_back(tc[5]);
    postNormals->push_back(np[4]);

    osg::DrawArrays *post = new osg::DrawArrays(osg::PrimitiveSet::QUADS, 0, postVertices->size());
    postGeometry->addPrimitiveSet(post);

    return postGeode;
}

osg::Geode *RoadObject::createGuardRailGeode()
{
    osg::Geode *guardRailGeode = new osg::Geode();
    guardRailGeode->setName(name.c_str());

    double h = 4.5;
    double texlength = 4.5;
    double texwidth = 0.33;
    double railLength = length;
    if (repeatLength > 0)
        railLength = repeatLength;
    if (s + railLength > road->getLength())
    {
        railLength = road->getLength() - s;
    }
    if (repeatDistance > 0)
    {
        h = repeatDistance;
        texlength = repeatDistance;
    }

    osg::Geometry *guardRailGeometry;
    guardRailGeometry = new osg::Geometry();
    guardRailGeometry->setUseDisplayList(coVRConfig::instance()->useDisplayLists());
    guardRailGeometry->setUseVertexBufferObjects(coVRConfig::instance()->useVBOs());
    guardRailGeode->addDrawable(guardRailGeometry);

    guardRailGeometry->setUseDisplayList(true);
    guardRailGeometry->setUseVertexBufferObjects(false);

    osg::Vec3Array *guardRailVertices;
    guardRailVertices = new osg::Vec3Array;
    guardRailGeometry->setVertexArray(guardRailVertices);

    osg::Vec3Array *guardRailNormals;
    guardRailNormals = new osg::Vec3Array;
    guardRailGeometry->setNormalArray(guardRailNormals);
    guardRailGeometry->setNormalBinding(osg::Geometry::BIND_PER_VERTEX);

    osg::Vec2Array *guardRailTexCoords;
    guardRailTexCoords = new osg::Vec2Array;
    guardRailGeometry->setTexCoordArray(3, guardRailTexCoords);

    osg::Geometry *guardRailPostGeometry;
    guardRailPostGeometry = new osg::Geometry();
    guardRailPostGeometry->setUseDisplayList(coVRConfig::instance()->useDisplayLists());
    guardRailPostGeometry->setUseVertexBufferObjects(coVRConfig::instance()->useVBOs());

    guardRailGeode->addDrawable(guardRailPostGeometry);

    osg::Vec3Array *guardRailPostVertices;
    guardRailPostVertices = new osg::Vec3Array;
    guardRailPostGeometry->setVertexArray(guardRailPostVertices);

    osg::Vec2Array *guardRailPostTexCoords;
    guardRailPostTexCoords = new osg::Vec2Array;
    guardRailPostGeometry->setTexCoordArray(3, guardRailPostTexCoords);

    osg::Vec3Array *guardRailPostNormals;
    guardRailPostNormals = new osg::Vec3Array;
    guardRailPostGeometry->setNormalArray(guardRailPostNormals);
    guardRailPostGeometry->setNormalBinding(osg::Geometry::BIND_PER_VERTEX);

    bool up = true;
    if (s ==  0)
        up = false; // do not go up if we start right at the bginning of a road, then we continue from the last road
    bool down = false;

    RoadPoint railPoint;
    RoadPoint railPoint2;

    osg::Vec3 n;

    LaneSection * currentLaneSection = NULL;
    int currentLaneId = 0;
    double sSection = s;
    double d = 0.0;
    double currentT = t;

    for (double currentS = s; currentS <= s + railLength; currentS += h)
    {
        if (currentS > (s + railLength - (h)))
        {
            currentS = (s + railLength);
            if (fabs(s + railLength - road->getLength()) > 1.0e-6) // only go down if the rail does not extend to the end of the road.
            {
                down = true;
            }
        }

        if (road->getLaneSection(currentS) != currentLaneSection)
        {
            LaneSection * newLaneSection = road->getLaneSection(currentS);
            while (currentLaneSection && (currentLaneSection != newLaneSection))
            {
                if (t < 0)
                {
                    currentT = -currentLaneSection->getLaneSpanWidth(0, currentLaneId, road->getLaneSectionEnd(currentLaneSection->getStart())) + d;
                }
                else if (t > 0)
                {
                    currentT = currentLaneSection->getLaneSpanWidth(0, currentLaneId, road->getLaneSectionEnd(currentLaneSection->getStart())) + d;
                }

                currentLaneSection = road->getLaneSectionNext(currentLaneSection->getStart() + 1.0e-3);
                currentLaneId = road->getLaneNumber(currentLaneSection->getStart(), currentT);
                sSection = currentLaneSection->getStart();
            }

            currentLaneSection = newLaneSection;
            currentLaneId = road->getLaneNumber(sSection, currentT);

            if (t < 0)
            {
                if (fabs(currentT) < currentLaneSection->getLaneSpanWidth(0, currentLaneId + 1, currentS) + currentLaneSection->getLaneWidth(currentS, currentLaneId)/2)
                {
                    currentLaneId++;
                }
                d = currentLaneSection->getLaneSpanWidth(0, currentLaneId, currentS) + currentT;
            }
            else if (t >  0)
            {
                if (currentT < currentLaneSection->getLaneSpanWidth(0, currentLaneId - 1,  currentS) + currentLaneSection->getLaneWidth(currentS, currentLaneId)/2)
                {
                    currentLaneId--;
                }
                d = currentT -  currentLaneSection->getLaneSpanWidth(0, currentLaneId, currentS);
            }
        }


        if (t < 0)
        {
            currentT = -currentLaneSection->getLaneSpanWidth(0, currentLaneId, currentS) + d;
        }
        else if (t > 0)
        {
            currentT = currentLaneSection->getLaneSpanWidth(0, currentLaneId, currentS) + d;
        }

        railPoint = road->getRoadPoint(currentS, currentT);
        if (t > 0)
            railPoint2 = road->getRoadPoint(currentS, currentT + 1.0);
        else
            railPoint2 = road->getRoadPoint(currentS, currentT - 1.0);
        n.set(railPoint2.x() - railPoint.x(), railPoint2.y() - railPoint.y(), railPoint2.z() - railPoint.z());
        n.normalize();
        osg::Vec3 negn = -n;

        if (up || down)
        {

            guardRailVertices->push_back(osg::Vec3(railPoint.x(), railPoint.y(), railPoint.z() - 0.31));
            guardRailTexCoords->push_back(osg::Vec2((currentS / texlength), 0));

            guardRailVertices->push_back(osg::Vec3(railPoint.x(), railPoint.y(), railPoint.z()));
            guardRailTexCoords->push_back(osg::Vec2((currentS / texlength), 0.33 / texwidth));
        }
        else
        {
            guardRailVertices->push_back(osg::Vec3(railPoint.x(), railPoint.y(), railPoint.z() + 0.44));
            guardRailTexCoords->push_back(osg::Vec2((currentS / texlength), 0));

            guardRailVertices->push_back(osg::Vec3(railPoint.x(), railPoint.y(), railPoint.z() + 0.75));
            guardRailTexCoords->push_back(osg::Vec2((currentS / texlength), 0.33 / texwidth));

            // post

            osg::Vec3 p1, p2, p3, p4;
            osg::Vec3 n2;
            n2.set(railPoint.nx(), railPoint.ny(), railPoint.nz());
            n2.normalize();
            osg::Vec3 n3 = n2 ^ n;
            n3.normalize();

            p1.set(railPoint.x(), railPoint.y(), railPoint.z());
            p1 += n * 0.01;
            p2 = p1 + (n3 * 0.08);
            p3 = p2 + (n * 0.06);
            p4 = p1 + (n * 0.06);

            osg::Vec3 p5, p6, p7, p8;
            p5.set(p1.x(), p1.y(), p1.z() + 0.44);
            p6.set(p2.x(), p2.y(), p2.z() + 0.44);

            guardRailPostVertices->push_back(p1);
            guardRailPostTexCoords->push_back(osg::Vec2(0.02, 0.02));
            guardRailPostNormals->push_back(n);
            guardRailPostVertices->push_back(p2);
            guardRailPostTexCoords->push_back(osg::Vec2(0.02, 0.1));
            guardRailPostNormals->push_back(n);
            guardRailPostVertices->push_back(p6);
            guardRailPostTexCoords->push_back(osg::Vec2(0.1, 0.1));
            guardRailPostNormals->push_back(n);
            guardRailPostVertices->push_back(p5);
            guardRailPostTexCoords->push_back(osg::Vec2(0.1, 0.02));
            guardRailPostNormals->push_back(n);

            p5.set(p1.x(), p1.y(), p1.z() + 0.6);
            p6.set(p2.x(), p2.y(), p2.z() + 0.6);
            p7.set(p3.x(), p3.y(), p3.z() + 0.6);
            p8.set(p4.x(), p4.y(), p4.z() + 0.6);

            osg::Vec3 negn3 = -n3;

            guardRailPostVertices->push_back(p2);
            guardRailPostTexCoords->push_back(osg::Vec2(0.02, 0.02));
            guardRailPostNormals->push_back(negn3);
            guardRailPostVertices->push_back(p3);
            guardRailPostTexCoords->push_back(osg::Vec2(0.02, 0.1));
            guardRailPostNormals->push_back(negn3);
            guardRailPostVertices->push_back(p7);
            guardRailPostTexCoords->push_back(osg::Vec2(0.1, 0.1));
            guardRailPostNormals->push_back(negn3);
            guardRailPostVertices->push_back(p6);
            guardRailPostTexCoords->push_back(osg::Vec2(0.1, 0.02));
            guardRailPostNormals->push_back(negn3);

            guardRailPostVertices->push_back(p1);
            guardRailPostTexCoords->push_back(osg::Vec2(0.02, 0.02));
            guardRailPostNormals->push_back(negn3);
            guardRailPostVertices->push_back(p4);
            guardRailPostTexCoords->push_back(osg::Vec2(0.02, 0.1));
            guardRailPostNormals->push_back(negn3);
            guardRailPostVertices->push_back(p8);
            guardRailPostTexCoords->push_back(osg::Vec2(0.1, 0.1));
            guardRailPostNormals->push_back(negn3);
            guardRailPostVertices->push_back(p5);
            guardRailPostTexCoords->push_back(osg::Vec2(0.1, 0.02));
            guardRailPostNormals->push_back(negn3);
        }

        if (t > 0)
        {
            guardRailNormals->push_back(n);
            guardRailNormals->push_back(n);
        }
        else
        {
            guardRailNormals->push_back(negn);
            guardRailNormals->push_back(negn);
        }

        up = false;
    }

    osg::DrawArrays *guardRailPosts = new osg::DrawArrays(osg::PrimitiveSet::QUADS, 0, guardRailPostVertices->size());
    guardRailPostGeometry->addPrimitiveSet(guardRailPosts);

    osg::DrawArrays *guardRail = new osg::DrawArrays(osg::PrimitiveSet::TRIANGLE_STRIP, 0, guardRailVertices->size());
    guardRailGeometry->addPrimitiveSet(guardRail);

    osg::StateSet *guardRailStateSet = guardRailGeode->getOrCreateStateSet();

    guardRailStateSet->setMode(GL_LIGHTING, osg::StateAttribute::ON);
    //guardRailStateSet->setMode ( GL_LIGHT0, osg::StateAttribute::ON );
    //guardRailStateSet->setMode ( GL_LIGHT1, osg::StateAttribute::ON);

    const char *fileName = coVRFileManager::instance()->getName("share/covise/materials/guardRailTex.jpg");
    if (fileName)
    {
        osg::Image *guardRailTexImage = osgDB::readImageFile(fileName);
        osg::Texture2D *guardRailTex = new osg::Texture2D;
        guardRailTex->setWrap(osg::Texture2D::WRAP_S, osg::Texture::REPEAT);
        guardRailTex->setWrap(osg::Texture2D::WRAP_T, osg::Texture::REPEAT);
        if (guardRailTexImage)
            guardRailTex->setImage(guardRailTexImage);
        guardRailStateSet->setTextureAttributeAndModes(3, guardRailTex, osg::StateAttribute::ON);
    }
    else
    {
        std::cerr << "ERROR: no texture found named: share/covise/materials/guardRailTex.jpg";
    }

    return guardRailGeode;
}

osg::Geode *RoadObject::createSTEPBarrierGeode()
{
    osg::Geode *STEPBarrierGeode = new osg::Geode();
    STEPBarrierGeode->setName(name.c_str());

    double h = 4.5;
    double texlength = 4.5;
    double texwidth = 0.33;
    double railLength = length;
    if (repeatLength > 0)
        railLength = repeatLength;
    if (s + railLength > road->getLength())
    {
        railLength = road->getLength() - s;
    }
    if (repeatDistance > 0)
    {
        h = repeatDistance;
        texlength = repeatDistance;
    }

    osg::Geometry *STEPBarrierGeometry;
    STEPBarrierGeometry = new osg::Geometry();
    STEPBarrierGeometry->setUseDisplayList(coVRConfig::instance()->useDisplayLists());
    STEPBarrierGeometry->setUseVertexBufferObjects(coVRConfig::instance()->useVBOs());
    STEPBarrierGeode->addDrawable(STEPBarrierGeometry);

    STEPBarrierGeometry->setUseDisplayList(true);
    STEPBarrierGeometry->setUseVertexBufferObjects(false);

    osg::Vec3Array *STEPBarrierVertices;
    STEPBarrierVertices = new osg::Vec3Array;
    STEPBarrierGeometry->setVertexArray(STEPBarrierVertices);

    osg::Vec3Array *STEPBarrierNormals;
    STEPBarrierNormals = new osg::Vec3Array;
    STEPBarrierGeometry->setNormalArray(STEPBarrierNormals);
    STEPBarrierGeometry->setNormalBinding(osg::Geometry::BIND_PER_VERTEX);

    osg::Vec2Array *STEPBarrierTexCoords;
    STEPBarrierTexCoords = new osg::Vec2Array;
    STEPBarrierGeometry->setTexCoordArray(3, STEPBarrierTexCoords);

    RoadPoint railPoint;
    RoadPoint railPoint2;

    osg::Vec3 n;
    double wr, wl;
    double startWidth;
    road->getRoadSideWidths(s, wr, wl);
    if (t > 0)
        startWidth = wl;
    else
        startWidth = wr;

    int numSegments = 7;
    osg::Vec2 profile[8];
    profile[0] = osg::Vec2(-0.271, 0);
    profile[1] = osg::Vec2(-0.249, 0.15);
    profile[2] = osg::Vec2(-0.189, 0.2);
    profile[3] = osg::Vec2(-0.091, 0.9);
    profile[4] = osg::Vec2(0.091, 0.9);
    profile[5] = osg::Vec2(0.189, 0.2);
    profile[6] = osg::Vec2(0.249, 0.15);
    profile[7] = osg::Vec2(0.271, 0);

    for (int pseg = 0; pseg < numSegments; pseg++)
    {

        for (double currentS = s; currentS <= s + railLength; currentS += h)
        {
            if (currentS > (s + railLength - (h)))
            {
                currentS = (s + railLength);
            }
            double currentT;
            road->getRoadSideWidths(currentS, wr, wl);
            if (t > 0)
                currentT = t - startWidth + wl;
            else
                currentT = t - startWidth + wr;

            railPoint = road->getRoadPoint(currentS, currentT);
            if (t > 0)
                railPoint2 = road->getRoadPoint(currentS, currentT + 1.0);
            else
                railPoint2 = road->getRoadPoint(currentS, currentT - 1.0);
            n.set(railPoint2.x() - railPoint.x(), railPoint2.y() - railPoint.y(), railPoint2.z() - railPoint.z());
            n.normalize();
            osg::Vec3 negn = -n;

            STEPBarrierVertices->push_back(osg::Vec3(railPoint.x(), railPoint.y(), railPoint.z() + 0.44));
            STEPBarrierTexCoords->push_back(osg::Vec2((currentS / texlength), 0));

            STEPBarrierVertices->push_back(osg::Vec3(railPoint.x(), railPoint.y(), railPoint.z() + 0.75));
            STEPBarrierTexCoords->push_back(osg::Vec2((currentS / texlength), 0.33 / texwidth));

            if (t > 0)
            {
                STEPBarrierNormals->push_back(n);
                STEPBarrierNormals->push_back(n);
            }
            else
            {
                STEPBarrierNormals->push_back(negn);
                STEPBarrierNormals->push_back(negn);
            }
        }
    }
    osg::DrawArrays *STEPBarrier = new osg::DrawArrays(osg::PrimitiveSet::TRIANGLE_STRIP, 0, STEPBarrierVertices->size());
    STEPBarrierGeometry->addPrimitiveSet(STEPBarrier);

    osg::StateSet *STEPBarrierStateSet = STEPBarrierGeode->getOrCreateStateSet();

    STEPBarrierStateSet->setMode(GL_LIGHTING, osg::StateAttribute::ON);

    const char *fileName = coVRFileManager::instance()->getName("share/covise/materials/STEPBarrierTex.jpg");
    if (fileName)
    {
        osg::Image *STEPBarrierTexImage = osgDB::readImageFile(fileName);
        osg::Texture2D *STEPBarrierTex = new osg::Texture2D;
        STEPBarrierTex->setWrap(osg::Texture2D::WRAP_S, osg::Texture::REPEAT);
        STEPBarrierTex->setWrap(osg::Texture2D::WRAP_T, osg::Texture::REPEAT);
        if (STEPBarrierTexImage)
            STEPBarrierTex->setImage(STEPBarrierTexImage);
        STEPBarrierStateSet->setTextureAttributeAndModes(3, STEPBarrierTex, osg::StateAttribute::ON);
    }
    else
    {
        std::cerr << "ERROR: no texture found named: share/covise/materials/STEPBarrierTex.jpg";
    }

    return STEPBarrierGeode;
}

osg::Geode *RoadObject::createOutlineGeode()
{
    osg::Geode *OutlineGeode = new osg::Geode();
    OutlineGeode->setName(name.c_str());

    double h = 4.5;
    double texlength = 4.5;
    double texwidth = 0.33;
    double railLength = length;
    if (repeatLength > 0)
        railLength = repeatLength;
    if (s + railLength > road->getLength())
    {
        railLength = road->getLength() - s;
    }
    if (repeatDistance > 0)
    {
        h = repeatDistance;
        texlength = repeatDistance;
    }

    osg::Geometry *OutlineGeometry;
    OutlineGeometry = new osg::Geometry();
    OutlineGeometry->setUseDisplayList(coVRConfig::instance()->useDisplayLists());
    OutlineGeometry->setUseVertexBufferObjects(coVRConfig::instance()->useVBOs());
    OutlineGeode->addDrawable(OutlineGeometry);

    OutlineGeometry->setUseDisplayList(true);
    OutlineGeometry->setUseVertexBufferObjects(false);

    osg::Vec3Array *OutlineVertices;
    OutlineVertices = new osg::Vec3Array;
    OutlineGeometry->setVertexArray(OutlineVertices);

    osg::Vec3Array *OutlineNormals;
    OutlineNormals = new osg::Vec3Array;
    OutlineGeometry->setNormalArray(OutlineNormals);
    OutlineGeometry->setNormalBinding(osg::Geometry::BIND_PER_VERTEX);

    osg::Vec2Array *OutlineTexCoords;
    OutlineTexCoords = new osg::Vec2Array;
    OutlineGeometry->setTexCoordArray(3, OutlineTexCoords);

    RoadPoint railPoint;
    RoadPoint railPoint2;
    RoadPoint nextRailPoint;

    osg::Vec3 n;

    int numVert = outline->size();
    int numSegments = numVert - 1;
    osg::Vec2 *profile = new osg::Vec2[numVert];
    float len=0;
    float *tcv = new float[numVert];
    for (int i = 0; i < numVert; i++)
    {
        profile[i] = osg::Vec2(outline->at(i).v, outline->at(i).z);
    } 
    tcv[0]=0;
    for (int i = 1; i < numVert; i++)
    {
        len += (profile[i] - profile[i-1]).length();
        tcv[i]=len;
    }
    for (int i = 1; i < numVert; i++)
    {
        tcv[i]/= len;
    }


    LaneSection * startLaneSection = NULL;
    int startLaneId = 0;
    double sSection = s;
    double d = 0.0;
    int lengthSegments;
    double currentT;
    LaneSection * currentLaneSection;
    int currentLaneId;

    for (int pseg = 0; pseg < numSegments; pseg++)
    {

        lengthSegments = 0;
        currentT = t;
        currentLaneSection = startLaneSection;
        currentLaneId = startLaneId;

        for (double currentS = s; currentS <= s + railLength; currentS += h)
        {
            if (currentS > (s + railLength - (h)))
            {
                currentS = (s + railLength);
            }

	    if (road->getLaneSection(currentS) != currentLaneSection)
            {
                LaneSection * newLaneSection = road->getLaneSection(currentS);
                while (currentLaneSection && (currentLaneSection != newLaneSection))
                {
                    if (t < 0)
                    {
                        currentT = -currentLaneSection->getLaneSpanWidth(0, currentLaneId, road->getLaneSectionEnd(currentLaneSection->getStart())) + d;
                    }
                    else if (t > 0)
                    {
                        currentT = currentLaneSection->getLaneSpanWidth(0, currentLaneId, road->getLaneSectionEnd(currentLaneSection->getStart())) + d;
                    }

                    currentLaneSection = road->getLaneSectionNext(currentLaneSection->getStart() + 1.0e-3);
                    currentLaneId = road->getLaneNumber(currentLaneSection->getStart(), currentT);
                    sSection = currentLaneSection->getStart();
                }

                currentLaneSection = newLaneSection;
                currentLaneId = road->getLaneNumber(sSection, currentT);
                if (t < 0)
                {
                    if (fabs(currentT) < currentLaneSection->getLaneSpanWidth(0, currentLaneId + 1, currentS) + currentLaneSection->getLaneWidth(currentS, currentLaneId)/2)
                    {
                        currentLaneId++;
                    }
                    d = currentLaneSection->getLaneSpanWidth(0, currentLaneId, currentS) + currentT;
                }
                else if (t >  0)
                {
                    if (currentT < currentLaneSection->getLaneSpanWidth(0, currentLaneId - 1,  currentS) + currentLaneSection->getLaneWidth(currentS, currentLaneId)/2)
                    {
                        currentLaneId--;
                    }
                    d = currentT -  currentLaneSection->getLaneSpanWidth(0, currentLaneId, currentS);
                }
            }



	    if (t < 0)
            {
                currentT = -currentLaneSection->getLaneSpanWidth(0, currentLaneId, currentS) + d;
            }
            else if (t > 0)
            {
                currentT = currentLaneSection->getLaneSpanWidth(0, currentLaneId, currentS) + d;
            }



            nextRailPoint = road->getRoadPoint(currentS + h, currentT);
            osg::Vec3 p3(nextRailPoint.x(), nextRailPoint.y(), nextRailPoint.z());

            railPoint = road->getRoadPoint(currentS, currentT);
            if (t > 0)
                railPoint2 = road->getRoadPoint(currentS, currentT + 1.0);
            else
                railPoint2 = road->getRoadPoint(currentS, currentT - 1.0);
            n.set(railPoint2.x() - railPoint.x(), railPoint2.y() - railPoint.y(), railPoint2.z() - railPoint.z());
            n.normalize();
            osg::Vec3 negn = -n;
            osg::Vec3 p1(railPoint.x(), railPoint.y(), railPoint.z());
            p1 += n * profile[pseg].x();
            p1[2] += profile[pseg].y();
            OutlineVertices->push_back(osg::Vec3(p1.x(), p1.y(), p1.z()));
            OutlineTexCoords->push_back(osg::Vec2((currentS/texlength), tcv[pseg]));

            osg::Vec3 p2(railPoint.x(), railPoint.y(), railPoint.z());
            p2 += n * profile[pseg + 1].x();
            p2[2] += profile[pseg + 1].y();
            OutlineVertices->push_back(osg::Vec3(p2.x(), p2.y(), p2.z()));

            OutlineTexCoords->push_back(osg::Vec2((currentS/texlength), tcv[pseg+1]));
            lengthSegments += 2;
            osg::Vec3 up = p2 - p1;
            osg::Vec3 forward = p3 - p1;
            osg::Vec3 norm = up ^ forward;
            /*
			if(p1.z() == p2.z())
			{
				norm.set(0,0,-1);
			}
			else if(p1.z() < p2.z())
			{
				osg::Vec3 up = p2-p1;
				up.normalize();
				osg::Vec3 right;
				if(t>=0)
				{
					right= up ^ negn;
				}
				else
				{
					right= up ^ n;
				}
				right.normalize();
				norm = up ^ right;
				norm.normalize();
			}
			else 
			{
				osg::Vec3 up = p1-p2;
				up.normalize();
				osg::Vec3 right;
				if(t>=0)
				{
					right= up ^ negn;
				}
				else
				{
					right= up ^ n;
				}
				right.normalize();
				norm = up ^ right;
				norm.normalize();
			}
			*/

            OutlineNormals->push_back(norm);
            OutlineNormals->push_back(norm);
        }

        osg::DrawArrays *OutlineBarrier = new osg::DrawArrays(osg::PrimitiveSet::TRIANGLE_STRIP, lengthSegments * pseg, lengthSegments);
        OutlineGeometry->addPrimitiveSet(OutlineBarrier);
    }

    osg::StateSet *OutlineStateSet = OutlineGeode->getOrCreateStateSet();

    OutlineStateSet->setMode(GL_LIGHTING, osg::StateAttribute::ON);
    if(numSegments>2)
    {
        osg::CullFace *cullFace = new osg::CullFace();
        cullFace->setMode(osg::CullFace::BACK);
        OutlineStateSet->setAttributeAndModes(cullFace, osg::StateAttribute::ON);
    }
    if(textureFileName!="")
    {
        const char *fileName = coVRFileManager::instance()->getName(textureFileName.c_str());
        if (fileName)
        {
            osg::Image *OutlineTexImage = osgDB::readImageFile(fileName);
            osg::Texture2D *OutlineTex = new osg::Texture2D;
            OutlineTex->setWrap(osg::Texture2D::WRAP_S, osg::Texture::REPEAT);
            OutlineTex->setWrap(osg::Texture2D::WRAP_T, osg::Texture::CLAMP);
            if (OutlineTexImage)
                OutlineTex->setImage(OutlineTexImage);
            OutlineStateSet->setTextureAttributeAndModes(3, OutlineTex, osg::StateAttribute::ON);
            osg::AlphaFunc *alphaFunc = new osg::AlphaFunc();
            alphaFunc->setFunction(osg::AlphaFunc::GEQUAL, 0.1f);

            OutlineStateSet->setAttributeAndModes(alphaFunc, osg::StateAttribute::ON);
        }
        else
        {
            std::cerr << "ERROR: no texture found named: "<< textureFileName<< std::endl;
        }
    }

    return OutlineGeode;
}

osg::Geode *RoadObject::createJerseyBarrierGeode()
{
    return NULL;
}
