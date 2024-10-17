/* This file is part of COVISE.

  You can use it under the terms of the GNU Lesser General Public License
  version 2.1 or later, see lgpl-2.1.txt.

* License: LGPL 2+ */

/****************************************************************************\
 **                                                          (C)2020 HLRS  **
 **                                                                        **
 ** Description: OpenCOVER Plug-In for reading Vineyard sensor data       **
 **                                                                        **
 **                                                                        **
 ** Author: Uwe Woessner                                                   **
 **                                                                        **
 ** History:                                                               **
 ** April 2020  v1                                                         **
 **                                                                        **
 **                                                                        **
\****************************************************************************/

#include "Vineyard.h"
#include <osg/LineWidth>
#include <osg/Version>
#include <cover/coVRTui.h>
#include <cover/coVRFileManager.h>
#include <cover/coVRAnimationManager.h>
#include <boost/tokenizer.hpp>
#include <boost/filesystem.hpp>

VineyardPlugin *VineyardPlugin::plugin = NULL;

std::string proj_to ="+proj=utm +zone=32 +ellps=GRS80 +units=m +no_defs ";
std::string proj_from = "+proj=latlong";
float offset [] = {-507048.f,-5398554.9,-450};

VineyardPlugin::VineyardPlugin()
: coVRPlugin(COVER_PLUGIN_NAME)
, ui::Owner("VineyardPlugin", cover->ui)
{
    fprintf(stderr, "Starting Vineyard Plugin\n");
    plugin = this;
    
    VineyardRoot = new osg::MatrixTransform();
    VineyardRoot->setName("Vineyard");
    VineyardRoot->setMatrix(osg::Matrix::translate(-513820, -5426730, -244.0));
    cover->getObjectsRoot()->addChild(VineyardRoot);
    PVGroup = new osg::MatrixTransform();
    PVGroup->setName("PV");

    VineyardRoot->addChild(PVGroup);
    
    VineTab = new ui::Menu("Vineyard",VineyardPlugin::plugin);
    VineTab->setText("Vineyard");
    
    ShowPV = new ui::Button(VineTab,"ShowPV");
    ShowPV->setText("Show PV");
    ShowPV->setCallback([this] (bool PVVisible){
        if (PVVisible)
        {
            if (PVGroup->getNumParents()==0)
                VineyardRoot->addChild(PVGroup);
        }
        else
        {
            if (PVGroup->getNumParents() != 0)
                VineyardRoot->removeChild(PVGroup);
        }
    });
}

VineyardPlugin::~VineyardPlugin()
{
    
}
bool VineyardPlugin::init()
{
    PVL = osgDB::readNodeFile("/data/Weinberge/PVL.ive");
    PVP = osgDB::readNodeFile("/data/Weinberge/PVP.ive");
    loadPVShp("/data/Weinberge/hessigheim-solar-corrected-elevation.shp");
    return true;
}


bool VineyardPlugin::loadPVShp(const std::string& filename)
{

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
        // Get the feature's class/type
        const char* Name = feature->GetFieldAsString("Name");  // Adjust field name
        const char* Spacing = feature->GetFieldAsString("Spacing");  // Adjust field name
        if(Spacing != nullptr)
            sscanf(Spacing, "%lf", &spacing);

        // Check if the feature's class is "e_602_tehnopaigaldis_p"
        if (Name && std::string(Name) == "10")
        {
        }
        else if (Name && std::string(Name) == "20")
        {
            // Print geometry information or any other relevant info
            OGRGeometry* geometry = feature->GetGeometryRef();
            if (geometry != nullptr) {
                if (wkbFlatten(geometry->getGeometryType()) == wkbPoint) {
                    OGRPoint* point = static_cast<OGRPoint*>(geometry);

                    // Output X, Y, and Z coordinates
                    double x = point->getY();
                    double y = point->getX();
                    double z = point->getZ();  // Z coordinate (may be 0 if 2D data)
                    auto PVT = new osg::MatrixTransform();
                    PVT->setName(feature->GetFieldAsString("name"));
                    PVGroup->addChild(PVT);
                }
            }
        }
        else
        {
            // Output the feature's fields
           /* for (int i = 0; i < feature->GetFieldCount(); i++) {
                const char* fieldName = feature->GetFieldDefnRef(i)->GetNameRef();
                const char* fieldValue = feature->GetFieldAsString(i);
                std::cout << fieldName << ": " << fieldValue << std::endl;
            }*/
        }
        auto pvori = O_LANDSCAPE;
        const char* orientation = feature->GetFieldAsString("orientation");
        if (orientation != nullptr && orientation[0] == 'l')
        {

        }

        // Print geometry information or any other relevant info
        OGRGeometry* geometry = feature->GetGeometryRef();
        if (geometry != nullptr)
        {
            if (wkbFlatten(geometry->getGeometryType()) == wkbLineString)
            {
                OGRLineString* lineString = static_cast<OGRLineString*>(geometry);

                // Output the number of points in the linestring
                int numPoints = lineString->getNumPoints();
                double x = lineString->getX(0);
                double y = lineString->getY(0);
                double z = lineString->getZ(0);
                osg::Vec3d pos(x, y, z);
                osg::Vec3d lastpos(x, y, z);
                double lp = 0;
                // Loop through the vertices of the linestring
                for (int n = 1; n < numPoints; n++)
                {
                    x = lineString->getX(n);
                    y = lineString->getY(n);
                    z = lineString->getZ(n);
                    osg::Vec3d p2(x, y, z);
                    osg::Vec3d dir = (p2 - lastpos);
                    osg::Vec3d dirxy = dir;
                    dirxy[2] = 0;
                    double len = dirxy.length();
                    dir.normalize();
                    double angle = atan2(p2.x() - lastpos.x(), p2.y() - lastpos.y()) ;
                    while (lp + spacing < lp+len)
                    {
                        pos = lastpos;
                        osg::MatrixTransform *posMT = new osg::MatrixTransform();
                        posMT->setMatrix(osg::Matrix::rotate(angle, osg::Vec3(0, 0, 1)) * osg::Matrix::translate(pos));
                        if(pvori==O_LANDSCAPE)
                            posMT->addChild(PVL);
                        else
                            posMT->addChild(PVP);
                        PVGroup->addChild(posMT);
                        lp += spacing;
                        len -= spacing;
                        lastpos = lastpos + dir * (spacing);
                    }

                }

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
    /*

    for (size_t i = 0; i < nx; i++)
    {
        std::cerr << i << std::endl;
        for (size_t j = 0; j < ny; j++)
        {
            if (shapeObjects[i * ny + j].size() > 0)
            {
                auto featureEntry = shapeObjects[i * ny + j][0];
                // Write the nodes to files
                // Construct filename
                std::ostringstream filename;
                filename << "objects_" << featureEntry.i * 1000 << "_" << featureEntry.j * 1000 << ".ive";
                //writeNodeToFile(cityObjectMembersSorted[i * ny + j], filename.str());
                osg::ref_ptr<osg::Group> root = new osg::Group;
                for (const auto& featureEntry : shapeObjects[i * ny + j])
                {
                    osg::ProxyNode* p = new osg::ProxyNode();
                    osg::MatrixTransform* m = new osg::MatrixTransform();
                    float scale = 1;
                    float angle = featureEntry.angle;
                    if (featureEntry.type == windturbine)
                    {
                        const char* Height = featureEntry.feature->GetFieldAsString("korgus");  // Adjust field name
                        float height = std::stof(Height);
                        scale = height / 1.053; // our model is 1.053 m high
                        p->setFileName(0, "Windrad.ive");
                    }
                    else if (featureEntry.type == powerline)
                    {
                        const char* Voltage = featureEntry.feature->GetFieldAsString("nimipinge");  // Adjust field name

                        float v = 0;
                        if (Voltage != "")
                        {
                            try {
                                v = std::stof(Voltage);
                            }
                            catch (...) {}

                        }
                        if (v <= 10)
                        {
                            p->setFileName(0, "FreileitungSmall.ive");
                        }
                        else if (v >= 110)
                        {
                            p->setFileName(0, "Freileitung.ive");
                        }
                        else
                        {
                            p->setFileName(0, "Freileitung20.ive");
                        }
                    }
                    osg::Vec3d position(featureEntry.y, featureEntry.x, featureEntry.z);
                    // osg::Vec3 direction;
                     //float angle = atan2(direction.x(), direction.y());
                    m->setMatrix(osg::Matrix::scale(osg::Vec3(scale, scale, scale)) * osg::Matrix::rotate(angle, osg::Vec3(0, 0, 1)) * osg::Matrix::translate(position));

                    m->addChild(p);
                    root->addChild(m);
                }
                osgDB::writeNodeFile(*root.get(), filename.str());
            }

        }
    }*/
    return true;
}
bool VineyardPlugin::update()
{
    return false;
}
bool VineyardPlugin::destroy()
{
    cover->getObjectsRoot()->removeChild(VineyardRoot);
    delete ShowPV;
    delete VineTab;
    return false;
}

COVERPLUGIN(VineyardPlugin)
