/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "CarGeometry.h"

#include <cover/coVRFileManager.h>
#include <cover/coVRMSController.h>
#include <cover/coVRPluginSupport.h>

#include <boost/algorithm/string/replace.hpp>
#include <sys/stat.h>

#include <osg/Group>
#include <osg/LOD>
#include <osg/MatrixTransform>
#include <osg/Node>
#include <osg/Switch>
#include <osg/Transform>
#include <osgDB/ReadFile>

// #include <math.h>
#define PI 3.141592653589793238

inline bool fileExists(const std::string &name)
{
    struct stat buffer;
    return (stat(name.c_str(), &buffer) == 0);
}

osg::Node *CarGeometry::loadFile(const std::string &file)
{
    if (!fileExists(file))
    {
        std::cerr << "CarGeometry::getCarNode(): File not YET found: " << file << "..." << std::endl;
        return nullptr;
    }

    static std::map<std::string, osg::Node *> cache;

    // Create a dummy group so the loaded content isn't added to the real root node.
    static osg::ref_ptr<osg::Group> dummyParent = new osg::Group();

    if (cache.find(file) == cache.end())
    {
        osg::Node *node = opencover::coVRFileManager::instance()->loadFile(file.c_str(), nullptr, dummyParent);

        // Create a safe(r) node name from the filename
        std::string name(file);
        boost::algorithm::replace_all(name, "/", "_");
        boost::algorithm::replace_all(name, ".", "_");
        boost::algorithm::replace_all(name, "\\", "_");
        node->setName(name);

        cache[file] = node;
    }

    return cache.at(file);
}

CarGeometry::CarGeometry(const std::string &name, const std::string &fileName, osg::Group *parentNode)
{
    transformNode = new osg::MatrixTransform();
    transformNode->setName(name);
    transformNode->setNodeMask(transformNode->getNodeMask() & ~(opencover::Isect::Intersection | opencover::Isect::Collision | opencover::Isect::Walk));

    if (parentNode)
    {
        parentNode->addChild(transformNode);
    }

    lodNode = new osg::LOD();
    transformNode->addChild(lodNode);

    osg::Node *modelNode = loadFile(fileName);
    lodNode->addChild(modelNode, 0, 1000.0);
}

CarGeometry::~CarGeometry()
{
    removeFromSceneGraph();
}

void CarGeometry::setTransform(osg::Matrix transform)
{
    transformNode->setMatrix(transform);
}

void CarGeometry::removeFromSceneGraph()
{
    while (transformNode->getNumParents() > 0)
    {
        transformNode->getParent(0)->removeChild(transformNode);
    }
}
