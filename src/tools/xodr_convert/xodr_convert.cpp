/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include <osgDB/WriteFile>
#include <osgDB/ReadFile>
#include <osg/Group>
#include <osg/Texture2D>
#include "RoadSystem.h"

int main(int argc, const char *argv[])
{
    RoadSystem *system = RoadSystem::Instance();

    std::string infile(argv[1]);
    std::string outfile(argv[2]);

    system->parseOpenDrive(infile);
    std::cout << "Information about road system: " << std::endl << system;
    std::cout << system;

    osg::Group *roadGroup = new osg::Group;
    roadGroup->setName("RoadSystem");

    osg::StateSet *roadGroupState = roadGroup->getOrCreateStateSet();
    std::string filename("roadTex.jpg");
    osg::Image *roadTexImage = osgDB::readImageFile(filename);
    osg::Texture2D *roadTex = new osg::Texture2D;
    roadTex->setWrap(osg::Texture2D::WRAP_S, osg::Texture::REPEAT);
    roadTex->setWrap(osg::Texture2D::WRAP_T, osg::Texture::REPEAT);
    if (roadTexImage)
        roadTex->setImage(roadTexImage);
    roadGroupState->setTextureAttributeAndModes(0, roadTex);

    int numRoads = system->getNumRoads();
    for (int i = 0; i < numRoads; ++i)
    {
        Road *road = system->getRoad(i);
        osg::Geode *roadGeode = road->getRoadGeode();
        if (roadGeode)
        {
            roadGroup->addChild(roadGeode);
        }
    }
    osgDB::writeNodeFile(*roadGroup, outfile);
}
