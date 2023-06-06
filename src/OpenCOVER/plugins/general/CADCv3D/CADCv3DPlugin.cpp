/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/****************************************************************************\
 **                                                            (C)2009 HLRS  **
 **                                                                          **
 ** Description: CADv3D plugin                                               **
 ** Load data from University of Cologne CAD server                          **
 **                                                                          **
 ** Author: Andreas Kopecki                                                  **
 **                                                                          **
 **                                                                          **
\****************************************************************************/

#include "CADCv3DPlugin.h"
#include <cover/coVRPluginSupport.h>
#include <cover/coVRFileManager.h>

#include <iostream>
#include <string>

#include <QFile>

#include "CADCv3DGeoList.h"

#include <osg/Array>
#include <osg/Geode>
#include <osg/Geometry>
#include <osg/Group>
#include <osg/Material>
#include <osg/PrimitiveSet>
#include <osg/StateSet>

#include <config/coConfig.h>

/*
static FileHandler handler =
{
   0,
   CADCv3DPlugin::loadCadHandler,
   CADCv3DPlugin::unloadCadHandler,
   "cgl"
};
*/

CADCv3DPlugin *CADCv3DPlugin::plugin = 0;

CADCv3DPlugin::CADCv3DPlugin()
: coVRPlugin(COVER_PLUGIN_NAME)
{
    //std::cerr.clear();
    //std::cerr << "CADCv3DPlugin::<init>" << std::endl;
    CADCv3DPlugin::plugin = this;

    this->currentFilename = "";
    this->loading = false;
    this->root = 0;
    this->rootNode = 0;
    this->renderWhileLoading = true;

    this->supportedReadFileExtensions.push_back("cgl");

    coVRFileManager::instance()->registerFileHandler(this);
}

// this is called if the plugin is removed at runtime
CADCv3DPlugin::~CADCv3DPlugin()
{
    //std::cerr.clear();
    //std::cerr << "CADCv3DPlugin::<dest>" << std::endl;
    coVRFileManager::instance()->unregisterFileHandler(this);
    CADCv3DPlugin::plugin = 0;
}

bool CADCv3DPlugin::init()
{
    return true;
}

osg::Node *CADCv3DPlugin::load(const std::string &location, osg::Group *)
{
    if (loadCad(location.c_str(), 0) != 0)
        return this->rootNode.get();
    else
        return 0;
}

void CADCv3DPlugin::preFrame()
{
}

bool CADCv3DPlugin::abortIO()
{
    std::cerr << "CADCv3DPlugin::abortIO fixme: stub" << std::endl;
    return false;
}

int CADCv3DPlugin::loadCad(const char *filename, osg::Group *)
{

    this->renderWhileLoading = coConfig::getInstance()->isOn("COVER.Plugin.CADCv3D.RenderWhileLoading", true);

    while (loadPart(filename) == coVRIOReader::Loading)
    {
    }

    return 1;
}

int CADCv3DPlugin::unloadCad(const char *filename)
{
    if (filename == this->currentFilename)
    {
        osg::Group *parent = this->rootNode->getParent(0);
        if (parent != 0)
            parent->removeChild(this->rootNode.get());

        this->currentFilename = "";
        this->rootNode = 0;

        return 1;
    }
    else
    {
        return 0;
    }
}

coVRIOReader::IOStatus CADCv3DPlugin::loadPart(const std::string &location)
{

    std::cerr.clear();

    // First part: open file and read internal data structs
    if (root == 0)
    {
        if (!loadData(location))
        {
            delete this->root;
            this->root = 0;
            return Failed;
        }
        else
        {
            if (!this->root->firstGeo())
            {
                delete this->root;
                this->root = 0;
                return Failed;
            }
            this->rootNode = new osg::Group();
            setIOProgress(0.0f);
            return Loading;
        }
    }

    // Continue to load otherwise
    std::string geoName;
    this->root->getGeoName(geoName);
    std::cerr << "CADCv3DPlugin::loadCad info: processing " << geoName << std::endl;

    int primitive, primitiveIndex;
    double x, y, z;
    int normalBinding, colorBinding;
    float r, g, b, a;

    osg::ref_ptr<osg::Geode> geode = new osg::Geode();

    osg::ref_ptr<osg::Geometry> geometry = new osg::Geometry();
    osg::ref_ptr<osg::StateSet> stateSet = geode->getOrCreateStateSet();
    osg::ref_ptr<osg::Vec3Array> vertices = new osg::Vec3Array();
    osg::ref_ptr<osg::Vec3Array> normals = new osg::Vec3Array();
    osg::ref_ptr<osg::Vec4Array> colors = new osg::Vec4Array();
    osg::ref_ptr<osg::DrawElementsUInt> primitives = 0;

    for (bool continueVertex = this->root->firstGeoVertex(x, y, z); continueVertex; continueVertex = this->root->nextGeoVertex(x, y, z))
    {
        vertices->push_back(osg::Vec3(x, y, z));
    }

    for (bool continueNormal = this->root->firstGeoNormal(x, y, z); continueNormal; continueNormal = this->root->nextGeoNormal(x, y, z))
    {
        normals->push_back(osg::Vec3(x, y, z));
    }

    for (bool continueColor = this->root->firstGeoColor(r, g, b, a); continueColor; continueColor = this->root->nextGeoColor(r, g, b, a))
    {
        colors->push_back(osg::Vec4(r, g, b, a));
    }

    // Don't add empty geometries
    if (vertices->empty())
    {
        if (!this->root->nextGeo())
        {
            this->loading = false;
            return Failed;
        }
        else
        {
            setIOProgress(((float)this->root->getCurrentGeoIndex()) / ((float)this->root->getNumberOfGeometries()) * 100.0f);
            return Loading;
        }
    }

    for (bool continuePrimitive = this->root->firstGeoPrimitive(primitive); continuePrimitive;
         continuePrimitive = this->root->nextGeoPrimitive(primitive))
    {

        GLenum mode = osg::PrimitiveSet::POINTS;

        switch (primitive)
        {
        case CADCv3DPrimitive::TYPE_POINTS:
            mode = osg::PrimitiveSet::POINTS;
            break;

        case CADCv3DPrimitive::TYPE_LINES:
            mode = osg::PrimitiveSet::LINES;
            break;

        case CADCv3DPrimitive::TYPE_LINE_STRIP:
            mode = osg::PrimitiveSet::LINE_STRIP;
            break;

        case CADCv3DPrimitive::TYPE_LINE_LOOP:
            mode = osg::PrimitiveSet::LINE_LOOP;
            break;

        case CADCv3DPrimitive::TYPE_TRIANGLES:
            mode = osg::PrimitiveSet::TRIANGLES;
            break;

        case CADCv3DPrimitive::TYPE_TRIANGLE_STRIP:
            mode = osg::PrimitiveSet::TRIANGLE_STRIP;
            break;

        case CADCv3DPrimitive::TYPE_TRIANGLE_FAN:
            mode = osg::PrimitiveSet::TRIANGLE_FAN;
            break;

        case CADCv3DPrimitive::TYPE_QUADS:
            mode = osg::PrimitiveSet::QUADS;
            break;

        case CADCv3DPrimitive::TYPE_QUAD_STRIP:
            mode = osg::PrimitiveSet::QUAD_STRIP;
            break;

        case CADCv3DPrimitive::TYPE_POLYGON:
            mode = osg::PrimitiveSet::POLYGON;
            break;

        default:
            assert("Unhandled CADCv3DPrimitive" == NULL);
            break;
        }

        int indexCount;
        this->root->getPrimitiveIndexCount(indexCount);

        uint *index = new uint[indexCount];
        int ctr;

        if (this->root->firstPrimitiveIndex(primitiveIndex))
            index[0] = primitiveIndex;
        else
            continue;

        for (ctr = 1; this->root->nextPrimitiveIndex(primitiveIndex); ++ctr)
        {
            index[ctr] = primitiveIndex;
        }

        primitives = new osg::DrawElementsUInt(mode, indexCount, index);
        geometry->addPrimitiveSet(primitives.get());

        delete[] index;
    }

    geometry->setVertexArray(vertices.get());

    this->root->getGeoNormalBinding(normalBinding);
    this->root->getGeoColorBinding(colorBinding);

    if (!normals->empty())
    {
        geometry->setNormalArray(normals.get());
        switch (normalBinding)
        {
        case CADCv3DGeometry::BIND_PERVERTEX:
            geometry->setNormalBinding(osg::Geometry::BIND_PER_VERTEX);
            break;

        case CADCv3DGeometry::BIND_PERPRIMITIVE:
            geometry->setNormalBinding(osg::Geometry::BIND_PER_PRIMITIVE);
            break;

        case CADCv3DGeometry::BIND_PERGEOMETRY:
            geometry->setNormalBinding(osg::Geometry::BIND_OVERALL);
            break;

        default:
            ;
        }
    }

    if (!colors->empty())
    {

        geometry->setColorArray(colors.get());

        osg::Material *material = new osg::Material();
        material->setColorMode(osg::Material::AMBIENT_AND_DIFFUSE);
        material->setAmbient(osg::Material::FRONT_AND_BACK, osg::Vec4(0.2f, 0.2f, 0.2f, 1.0));
        material->setDiffuse(osg::Material::FRONT_AND_BACK, osg::Vec4(0.9f, 0.9f, 0.9f, 1.0));
        material->setSpecular(osg::Material::FRONT_AND_BACK, osg::Vec4(0.4f, 0.4f, 0.4f, 1.0));
        material->setEmission(osg::Material::FRONT_AND_BACK, osg::Vec4(0.0f, 0.0f, 0.0f, 1.0));
        material->setShininess(osg::Material::FRONT_AND_BACK, 16.0f);
        stateSet->setAttributeAndModes(material, osg::StateAttribute::ON);

        switch (colorBinding)
        {
        case CADCv3DGeometry::BIND_PERVERTEX:
            geometry->setColorBinding(osg::Geometry::BIND_PER_VERTEX);
            break;

        case CADCv3DGeometry::BIND_PERPRIMITIVE:
            geometry->setColorBinding(osg::Geometry::BIND_PER_PRIMITIVE);
            break;

        case CADCv3DGeometry::BIND_PERGEOMETRY:
            geometry->setColorBinding(osg::Geometry::BIND_OVERALL);
            break;

        default:
            ;
        }
    }

    geometry->setStateSet(stateSet.get());

    geode->setName(geoName);
    geode->addDrawable(geometry.get());

    this->rootNode->addChild(geode.get());

    if (!this->root->nextGeo())
    {
        // Finished loading
        this->loading = false;
        delete this->root;
        this->root = 0;
        setIOProgress(100.0f);
        return Finished;
    }
    else
    {
        // Still loading
        setIOProgress(((float)this->root->getCurrentGeoIndex()) / ((float)this->root->getNumberOfGeometries()) * 100.0f);
        return Loading;
    }
}

int CADCv3DPlugin::loadCadHandler(const char *filename, osg::Group *group)
{
    if (CADCv3DPlugin::plugin)
        return plugin->loadCad(filename, group);
    return 0;
}

int CADCv3DPlugin::unloadCadHandler(const char *filename)
{
    if (CADCv3DPlugin::plugin)
        return plugin->unloadCad(filename);
    return 0;
}

bool CADCv3DPlugin::loadData(const std::string &filename)
{

    QFile file(QString::fromStdString(filename));
    if (!file.open(QIODevice::ReadOnly))
    {
        std::cerr << "CADCv3DPlugin::loadData err: cannot open "
                  << filename << std::endl;
        return false;
    }

    std::cerr << "CADCv3DPlugin::loadData info: loading " << filename << std::endl;

    this->loading = true;
    this->currentFilename = QString::fromStdString(filename);

    qint64 fileSize = file.size();
    char *data = new char[fileSize];
    qint64 fileResult = file.read(data, fileSize);
    (void)fileResult;

    this->root = new CADCv3DGeoList();
    this->root->read(data, fileSize);

    return true;
}

COVERPLUGIN(CADCv3DPlugin)
