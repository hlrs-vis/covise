/* This file is part of COVISE.

  You can use it under the terms of the GNU Lesser General Public License
  version 2.1 or later, see lgpl-2.1.txt.

* License: LGPL 2+ */

/****************************************************************************\
 **                                                          (C)2020 HLRS  **
 **                                                                        **
 ** Description: OpenCOVER Plug-In for reading Shapefiles                  **
 **                                                                        **
 **                                                                        **
 ** Author: Thomas Obst / Uwe Woessner                                     **
 **                                                                        **
 ** History:                                                               **
 ** April 2020  v1                                                         **
 **                                                                        **
 **                                                                        **
\****************************************************************************/

#include "Shapefiles.h"
#include <osg/Group>
#include <osg/LineWidth>
#include <osg/Version>
#include <cover/coVRTui.h>
#include <cover/coVRFileManager.h>
#include <cover/coVRAnimationManager.h>
#include <boost/tokenizer.hpp>
#include <boost/filesystem.hpp>

ShapefilesPlugin *ShapefilesPlugin::plugin = NULL;

osg::ref_ptr<osg::Material> ShapefilesPlugin::globalDefaultMaterial;

static const FileHandler handlers[] = {
    { NULL,
      ShapefilesPlugin::SloadSHP,
      ShapefilesPlugin::SunloadSHP,
      "shp" }
};

std::string proj_to ="+proj=utm +zone=32 +ellps=GRS80 +units=m +no_defs ";
std::string proj_from = "+proj=latlong";
float offset [] = {-507048.f,-5398554.9,-450};

ShapefilesPlugin::ShapefilesPlugin()
: coVRPlugin(COVER_PLUGIN_NAME)
, ui::Owner("ShapefilesPlugin", cover->ui)
{
    fprintf(stderr, "Starting Shapefiles Plugin\n");
    plugin = this;
    
    ShapefilesRoot = new osg::MatrixTransform();
    ShapefilesRoot->setName("Shapefiles");
    ShapefilesRoot->setMatrix(osg::Matrix::translate(-410000, -5320000, 40.0));
    cover->getObjectsRoot()->addChild(ShapefilesRoot);
    SHPGroup = new osg::MatrixTransform();
    SHPGroup->setName("SHP");

    ShapefilesRoot->addChild(SHPGroup);
    
    ShapefileTab = new ui::Menu("Shapefiles",ShapefilesPlugin::plugin);
    ShapefileTab->setText("Shapefiles");
    
    ShowShape = new ui::Button(ShapefileTab,"ShowShape");
    ShowShape->setText("Show Shapefiles");
    ShowShape->setCallback([this] (bool PVVisible){
        if (PVVisible)
        {
            if (SHPGroup->getNumParents()==0)
                ShapefilesRoot->addChild(SHPGroup);
        }
        else
        {
            if (SHPGroup->getNumParents() != 0)
                ShapefilesRoot->removeChild(SHPGroup);
        }
    });
}

ShapefilesPlugin::~ShapefilesPlugin()
{
    
}
bool ShapefilesPlugin::init()
{
    //PVL = osgDB::readNodeFile("/data/Weinberge/PVL.ive");
    //PVP = osgDB::readNodeFile("/data/Weinberge/PVP.ive");
    loadPVShp("/mnt/data/HLRS/klima-qgis/08315_Traj_RKLS.shp");
    return true;
}


bool ShapefilesPlugin::loadPVShp(const std::string& filename)
{
    fprintf(stderr, "loadPVShp called\n");

    // Initialize GDAL/OGR library
    GDALAllRegister();


    // Open the shapefile
    GDALDataset* dataset = static_cast<GDALDataset*>(GDALOpenEx(filename.c_str(), GDAL_OF_VECTOR, nullptr, nullptr, nullptr));
    if (dataset == nullptr) {
        std::cerr << "Failed to open shapefile." << std::endl;
        return false;
    }

    // Get the layer (assuming the shapefile has a single layer)
    OGRLayer* layer = dataset->GetLayer(0);
    if (layer == nullptr) {
        std::cerr << "Failed to get layer." << std::endl;
        GDALClose(dataset);
        return false;
    }

    // Loop through features
    OGRFeature* feature;
    layer->ResetReading(); // Reset reading to the beginning of the layer

    size_t featureNumber = 0;
    double spacing = 2.0;
    while ((feature = layer->GetNextFeature()) != nullptr) {
        bool used = false;

        OGRGeometry* geometry = feature->GetGeometryRef();
        if (geometry != nullptr)
        {
            if (wkbFlatten(geometry->getGeometryType()) == wkbLineString)
            {
                OGRLineString* lineString = static_cast<OGRLineString*>(geometry);

                drawTrajectory(lineString);

            }
        }

        // Destroy the feature to avoid memory leaks
        if (!used)
        {
            OGRFeature::DestroyFeature(feature);
        }
    }

    // Clean up
    GDALClose(dataset);

    return true;
}

void ShapefilesPlugin::drawTrajectory(OGRLineString* lineString)
{
    //fprintf(stderr, "drawTrajectory called\n");

    static double AlphaThreshold = 0.5;
    float linewidth = 4.0f;

    geode = new osg::Geode();
    osg::Geometry *geom = new osg::Geometry();

    cover->setRenderStrategy(geom);

    //Setup geometry
    // Loop through the vertices of the linestring

    osg::Vec3Array *vert = new osg::Vec3Array;
    int numPoints = lineString->getNumPoints();
    double x = 0.0;
    double y = 0.0;
    double z = 0.0;
    for (int n = 0; n < numPoints; n++)
    {
        x = lineString->getX(n);
        y = lineString->getY(n);
        z = lineString->getZ(n);
        osg::Vec3d point(x, y, z);
        vert->push_back(osg::Vec3(x, y , z));
    }

    geom->setVertexArray(vert);

    //color
    osg::Vec4Array *colArr = new osg::Vec4Array();
    for (int t = 0; t < lineString->getNumPoints(); ++t)
    {
        colArr->push_back(osg::Vec4(0.2, 0 , 4.0f, 1.0f));
    }
    geom->setColorArray(colArr);
    geom->setColorBinding(osg::Geometry::BIND_PER_VERTEX );

    //primitves
    osg::DrawArrayLengths *primitives = new osg::DrawArrayLengths(osg::PrimitiveSet::LINE_STRIP);
    primitives->push_back(lineString->getNumPoints());
    geom->addPrimitiveSet(primitives);

    //normals
    osg::Vec3Array *normalArray = new osg::Vec3Array();
    osg::Vec3 norm = osg::Vec3(0,0,1);
    norm.normalize();
    normalArray->push_back(norm);
    geom->setNormalArray(normalArray);
    geom->setNormalBinding(osg::Geometry::BIND_OVERALL);

    //geoState
    osg::StateSet *geoState = geode->getOrCreateStateSet();
    if (globalDefaultMaterial.get() == NULL)
    {
        globalDefaultMaterial = new osg::Material;
        globalDefaultMaterial->setColorMode(osg::Material::AMBIENT_AND_DIFFUSE);
        globalDefaultMaterial->setAmbient(osg::Material::FRONT_AND_BACK, osg::Vec4(0.2f, 0.2f, 0.2f, 1.0));
        globalDefaultMaterial->setDiffuse(osg::Material::FRONT_AND_BACK, osg::Vec4(1.0f, 1.0f, 1.0f, 1.0));
        globalDefaultMaterial->setSpecular(osg::Material::FRONT_AND_BACK, osg::Vec4(0.4f, 0.4f, 0.4f, 1.0));
        globalDefaultMaterial->setEmission(osg::Material::FRONT_AND_BACK, osg::Vec4(0.0f, 0.0f, 0.0f, 1.0));
        globalDefaultMaterial->setShininess(osg::Material::FRONT_AND_BACK, 16.0f);
    }
    geoState->setAttributeAndModes(globalDefaultMaterial.get(), osg::StateAttribute::ON);

    geoState->setRenderingHint(osg::StateSet::OPAQUE_BIN);
    geoState->setMode(GL_BLEND, osg::StateAttribute::OFF);
    geoState->setNestRenderBins(false);

    osg::AlphaFunc *alphaFunc = new osg::AlphaFunc();
    alphaFunc->setFunction(osg::AlphaFunc::GREATER, AlphaThreshold);
    geoState->setAttributeAndModes(alphaFunc, osg::StateAttribute::ON);


    geoState->setMode(GL_LIGHTING, osg::StateAttribute::OFF);
    osg::LineWidth *lineWidth = new osg::LineWidth(linewidth);
    geoState->setAttributeAndModes(lineWidth, osg::StateAttribute::ON);

    geode->setName("Trackline");
    geode->addDrawable(geom);
    geode->setStateSet(geoState);
    SHPGroup->addChild(geode);
}




bool ShapefilesPlugin::update()
{
    return false;
}
bool ShapefilesPlugin::destroy()
{
    cover->getObjectsRoot()->removeChild(ShapefilesRoot);
    delete ShowShape;
    delete ShapefileTab;
    return false;
}


//GPS fileHandler
int ShapefilesPlugin::SloadSHP(const char *filename, osg::Group *parent, const char *)
{
    instance()->loadSHP(filename, parent);
    return 0;
}
int ShapefilesPlugin::loadSHP(const char *filename, osg::Group *parent)
{
    /*if(parent == NULL)
        parent =OSGGPSPlugin;
    File *f = new File(filename, parent);
    this->addFile(f);*/
    return 0;
}

int ShapefilesPlugin::SunloadSHP(const char *filename, const char *)
{
    return ShapefilesPlugin::instance()->unloadSHP(filename);
}
int ShapefilesPlugin::unloadSHP(const char *filename)
{
    return 0;
}


COVERPLUGIN(ShapefilesPlugin)
