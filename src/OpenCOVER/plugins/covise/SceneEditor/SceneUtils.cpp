/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "SceneUtils.h"

#include "Behaviors/MountBehavior.h"

#include <QDir>
#include <osgDB/ReadFile>

#include <osg/MatrixTransform>

#include <cover/VRSceneGraph.h>
#include <cover/coVRConfig.h>
#include <cover/VRViewer.h>

SceneUtils::SceneUtils()
{
}

SceneUtils::~SceneUtils()
{
}

int SceneUtils::insertNode(osg::Group *node, SceneObject *so)
{
    if (so == NULL)
    {
        return -1;
    }

    osg::Node *n = so->getGeometryNode();
    if (n == NULL)
    {
        return -1;
    }

    if (n->getNumParents() != 1)
    {
        return -1;
    }

    osg::Group *parent = n->getParent(0);
    parent->removeChild(n);
    node->addChild(n);
    parent->addChild(node);

    return 1;
}

int SceneUtils::removeNode(osg::Group *node)
{
    if (node == NULL)
    {
        return 1;
    }

    if (node->getNumParents() != 1)
    {
        return -1;
    }

    node->ref(); // make sure node wont be deleted when we remove it from its parent
    osg::Group *parent = node->getParent(0);
    parent->removeChild(node);
    for (int i = 0; i < node->getNumChildren(); i++)
    {
        parent->addChild(node->getChild(0));
        node->removeChild(node->getChild(0));
    }
    node->unref();

    return 1;
}

osg::ref_ptr<osg::Node> SceneUtils::createGeometryFromXML(QDomElement *parentElem)
{
    std::vector<osg::ref_ptr<osg::Node> > geometries;
    QDomElement geometryElem = parentElem->firstChildElement("geometry");
    while (!geometryElem.isNull())
    {
        geometries.push_back(createSingleGeometryFromXML(&geometryElem));
        geometryElem = geometryElem.nextSiblingElement("geometry");
    }

    if (geometries.size() == 0)
    {
        std::cerr << "Error: No geometry section found" << std::endl;
        return NULL;
    }

    // always add a group (even if we have just one geometry) to keep things simple
    osg::ref_ptr<osg::Group> group = new osg::Group();
    std::vector<osg::ref_ptr<osg::Node> >::iterator iter;
    for (iter = geometries.begin(); iter != geometries.end(); ++iter)
    {
        group->addChild((*iter).get());
    }
    return group;
}

osg::ref_ptr<osg::Node> SceneUtils::createSingleGeometryFromXML(QDomElement *geometryElem)
{

    QDomElement elem;

    // read geometry

    elem = geometryElem->firstChildElement("filename");
    std::string filename = QDir(elem.attribute("value")).absolutePath().toStdString();
    osg::ref_ptr<osg::Node> geoNode = osgDB::readNodeFile(filename);
    if (!geoNode)
    {
        std::cerr << "Error: Could not read geometry file: " << filename << std::endl;
        return NULL;
    }

    // read transformations

    float x, y, z;
    osg::Matrix m;

    elem = geometryElem->firstChildElement("rotation");
    if (!elem.isNull())
    {
        x = osg::DegreesToRadians(elem.attribute("x", "0.0").toFloat());
        y = osg::DegreesToRadians(elem.attribute("y", "0.0").toFloat());
        z = osg::DegreesToRadians(elem.attribute("z", "0.0").toFloat());
        osg::Matrix rot;
        rot.makeRotate(x, osg::Vec3(1.0f, 0.0f, 0.0f), y, osg::Vec3(0.0f, 1.0f, 0.0f), z, osg::Vec3(0.0f, 0.0f, 1.0f));
        m.postMult(rot);
    }

    elem = geometryElem->firstChildElement("scaling");
    if (!elem.isNull())
    {
        x = elem.attribute("x", "1.0").toFloat();
        y = elem.attribute("y", "1.0").toFloat();
        z = elem.attribute("z", "1.0").toFloat();
        osg::Matrix scale;
        scale.makeScale(x, y, z);
        m.postMult(scale);
    }

    elem = geometryElem->firstChildElement("translation");
    if (!elem.isNull())
    {
        x = elem.attribute("x", "0.0").toFloat();
        y = elem.attribute("y", "0.0").toFloat();
        z = elem.attribute("z", "0.0").toFloat();
        osg::Matrix trans;
        trans.makeTranslate(x, y, z);
        m.postMult(trans);
    }

    if (!m.isIdentity())
    {
        osg::ref_ptr<osg::MatrixTransform> mt = new osg::MatrixTransform();
        mt->setMatrix(m);
        mt->addChild(geoNode.get());
        geoNode = mt;
    }

    elem = geometryElem->firstChildElement("namespace");
    if (!elem.isNull())
    {
        std::string geoNameSpace = elem.attribute("value", "").toStdString();
        if (geoNameSpace != "")
        {
            osg::ref_ptr<osg::Group> nsg = new osg::Group();
            nsg->setName("NAMESPACE:" + geoNameSpace);
            nsg->addChild(geoNode);
            geoNode = nsg;
        }
    }

    return geoNode;
}

SceneObject *SceneUtils::followFixedMountsToParent(SceneObject *so)
{
    while (true)
    {
        MountBehavior *mountBe = dynamic_cast<MountBehavior *>(so->findBehavior(BehaviorTypes::MOUNT_BEHAVIOR));
        if (!mountBe)
        {
            return so;
        }
        Connector *connector = mountBe->getActiveSlaveConnector();
        if (!connector)
        {
            return so;
        }
        if (connector->getCombinedRotation() != Connector::ROTATION_FIXED)
        {
            return so;
        }
        Connector *masterConn = connector->getMasterConnector();
        if (masterConn->getConstraint() != Connector::CONSTRAINT_POINT)
        {
            return so;
        }
        so = masterConn->getMasterObject();
    }
    return so;
}

bool SceneUtils::getPlaneIntersection(opencover::coPlane *plane, osg::Matrix pointerMat, osg::Vec3 &point)
{
    osg::Vec3 pointerPos1, pointerPos2, pointerDir;
    pointerPos1 = pointerMat.getTrans();
    pointerDir[0] = pointerMat(1, 0);
    pointerDir[1] = pointerMat(1, 1);
    pointerDir[2] = pointerMat(1, 2);
    pointerPos2 = pointerPos1 + pointerDir;
    return plane->getLineIntersectionPoint(pointerPos1, pointerPos2, point);
}

// Get the visibility of a plane defined by position and normal in local space (within the covers objects root xform).
//  1  orthogonal on front side
// >0  front side
//  0  viewing position in plane
// <0  back side
// -1  orthogonal on back side
float SceneUtils::getPlaneVisibility(osg::Vec3 position, osg::Vec3 normal)
{
    osg::Vec3 n = normal;
    n = opencover::cover->getXformMat().getRotate() * n;

    osg::Vec3 p = position;
    p *= opencover::VRSceneGraph::instance()->scaleFactor();
    p = opencover::cover->getXformMat().preMult(p);

    if (opencover::coVRConfig::instance()->orthographic())
    {
        return osg::Vec3(0.0f, 1.0f, 0.0f) * n;
    }
    else
    {
        osg::Vec3 comp = p - opencover::VRViewer::instance()->getViewerPos();
        comp.normalize();
        return comp * n;
    }
}
