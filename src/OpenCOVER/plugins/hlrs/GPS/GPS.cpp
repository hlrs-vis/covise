/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/****************************************************************************\ 
 **                                                            (C)2008 HLRS  **
 **                                                                          **
 ** Description: GPS OpenCOVER Plugin (is polite)                          **
 **                                                                          **
 **                                                                          **
 ** Author: U.Woessner, T.Gudat	                                             **
 **                                                                          **
 ** History:  								     **
 ** June 2008  v1	    				       		     **
 **                                                                          **
 **                                                                          **
\****************************************************************************/

#include "GPS.h"
#include "GPSPoint.h"
#include "Track.h"
#include "File.h"
#include "GPSALLPoints.h"
#include "GPSALLTracks.h"
#include <time.h>


#include <cover/coVRPluginSupport.h>
#include <cover/coVRFileManager.h>
#include <cover/coBillboard.h>
#include <cover/coVRLabel.h>
#include <osg/Group>
#include <osg/Switch>
#include <osg/Material>
#include <osg/Vec3>
#include <osg/Vec4>
#include <osg/ShapeDrawable>
#include <gdal_priv.h>
#include <config/CoviseConfig.h>


#include <xercesc/dom/DOM.hpp>
#if XERCES_VERSION_MAJOR < 3
#include <xercesc/dom/DOMWriter.hpp>
#else
#include <xercesc/dom/DOMLSSerializer.hpp>
#endif
#include <xercesc/framework/LocalFileFormatTarget.hpp>
#include <xercesc/util/XMLUni.hpp>
#include <xercesc/parsers/XercesDOMParser.hpp>


GPSPlugin *GPSPlugin::plugin = NULL;

static const int NUM_HANDLERS = 1;

using namespace opencover;

static const FileHandler handlers[] = {
    { NULL,
      GPSPlugin::SloadGPX,
      GPSPlugin::SloadGPX,
      GPSPlugin::SunloadGPX,
      "gpx" }
};

GPSPlugin::GPSPlugin(): ui::Owner("GPSPlugin", cover->ui)
{
    fprintf(stderr, "------------------\n GPSPlugin started\n------------------\n");
    plugin = this;

    for (int index = 0; index < NUM_HANDLERS; index++)
        coVRFileManager::instance()->registerFileHandler(&handlers[index]);
    GPSTab = NULL;
    GPSTab_create();
    OSGGPSPlugin = new osg::Group();
    OSGGPSPlugin->setName("GPS Plugin");
    cover->getObjectsRoot()->addChild(OSGGPSPlugin);

    //testlabel
    //std::string s  = "what";
    //int size = 300;
    //int len = 300;
    //Label = new coVRLabel(s.c_str(), size,len, osg::Vec4(1,0,0,1), osg::Vec4(0.1,0.1,0.1,1));
    //Label->setPositionInScene(osg::Vec3(1,1,0));


    //mapping of coordinates
#ifdef WIN32
    const char *pValue;
    size_t len;
    errno_t err = _dupenv_s(&((char *)pValue), &len, "ODDLOTDIR");
    if (err || pValue == NULL || strlen(pValue) == 0)
        err = _dupenv_s(&((char *)pValue), &len, "COVISEDIR");
    if (err)
        pValue = "";
#else
    const char *pValue = getenv("ODDLOTDIR");
    if (!pValue || pValue[0] == '\0')
        pValue = getenv("COVISEDIR");
    if (!pValue)
        pValue = "";
#endif
    std::string covisedir = pValue;
    dir = covisedir + "/share/covise/";

    std::string proj_from = "+proj=latlong +datum=WGS84";
    fprintf(stderr, "proj_from: %s\n",proj_from.c_str());
    if (!(pj_from = pj_init_plus(proj_from.c_str())))
    {
            fprintf(stderr, "ERROR: pj_from failed\n");
    }

    std::string proj_to ="+proj=tmerc +lat_0=0 +lon_0=9 +k=1.000000 +x_0=9703.397 +y_0=-5384244.453 +ellps=bessel +datum=potsdam";// +nadgrids=" + dir + std::string("BETA2007.gsb");
    fprintf(stderr, "proj_to: %s\n",proj_to.c_str());

    if (!(pj_to = pj_init_plus(proj_to.c_str())))
    {
            fprintf(stderr, "ERROR: pj_to failed\n");
    }

    GDALAllRegister();
    std::string heightMapFileName = coCoviseConfig::getEntry("heightMap", "COVER.Plugin.GPS", "/data/reallabor/gelaende/Herrenberg10mwgs84.tif");
    openImage(heightMapFileName);

}


bool GPSPlugin::update()
{
    //Label->update();
    return false;
}

GPSPlugin::~GPSPlugin()
{
    //OSGGPSPlugin->removeChildren(0,1);
    for (int index = 0; index < NUM_HANDLERS; index++){
        coVRFileManager::instance()->unregisterFileHandler(&handlers[index]);
    }
    for (auto *x : fileList){
        delete x;
    }

    GPSTab_delete();
    closeImage();
    fprintf(stderr, "------------------\n GPSPlugin stopped\n------------------\n");
}

void GPSPlugin::GPSTab_create(void)
{
    GPSTab = new ui::Menu("GPS", this);
    // infoLabel = new ui::Label("GPS Version 1.0", GPSTab);

    ToggleTracks = new ui::Button(GPSTab, "Toggle Tracks ON/OFF");
    ToggleTracks->setText("Tracks Tracks ON/OFF");
    ToggleTracks->setCallback([this](bool) {
        for (auto *x : fileList){
            if(x->SwitchTracks->getNewChildDefaultValue()){
                x->SwitchTracks->setAllChildrenOff();
            }
            else {
                x->SwitchTracks->setAllChildrenOn();
            }
        }
    });
    TogglePoints = new ui::Button(GPSTab, "Toggle Points ON/OFF");
    TogglePoints->setText("Tracks Points ON/OFF");
    TogglePoints->setCallback([this](bool) {
        for (auto *x : fileList){
            if(x->SwitchPoints->getNewChildDefaultValue()){
                x->SwitchPoints->setAllChildrenOff();
            }
            else {
                x->SwitchPoints->setAllChildrenOn();
            }
        }
    });

}
void GPSPlugin::GPSTab_delete(void)
{
    if (GPSTab)
    {
        //delete infoLabel;
        delete GPSTab;
    }
}
void GPSPlugin::addFile(File *f)
{
    fileList.push_back(f);
    OSGGPSPlugin->addChild(f->FileGroup);
    //fprintf(stderr, "File added to GPSPlugin\n");
}

void GPSPlugin::closeImage()
{
        if (heightDataset)
                GDALClose(heightDataset);
}
void GPSPlugin::openImage(std::string &name)
{

        heightDataset = (GDALDataset *)GDALOpen(name.c_str(), GA_ReadOnly);
    if (heightDataset != NULL)
    {
        int             nBlockXSize, nBlockYSize;
        int             bGotMin, bGotMax;
        double          adfMinMax[2];
        double        adfGeoTransform[6];

        printf("Size is %dx%dx%d\n", heightDataset->GetRasterXSize(), heightDataset->GetRasterYSize(), heightDataset->GetRasterCount());
        if (heightDataset->GetGeoTransform(adfGeoTransform) == CE_None)
        {
            printf("Origin = (%.6f,%.6f)\n",
                adfGeoTransform[0], adfGeoTransform[3]);
            printf("Pixel Size = (%.6f,%.6f)\n",
                adfGeoTransform[1], adfGeoTransform[5]);
        }
        int numRasterBands = heightDataset->GetRasterCount();

        heightBand = heightDataset->GetRasterBand(1);
        heightBand->GetBlockSize(&nBlockXSize, &nBlockYSize);
        cols = heightDataset->GetRasterXSize();
        rows = heightDataset->GetRasterYSize();
        double transform[100];
        heightDataset->GetGeoTransform(transform);

        xOrigin = transform[0];
        yOrigin = transform[3];
        pixelWidth = transform[1];
        pixelHeight = -transform[5];
        delete[] rasterData;
        rasterData = new float[cols*rows];
        float *pafScanline;
        int   nXSize = heightBand->GetXSize();
        pafScanline = (float *)CPLMalloc(sizeof(float)*nXSize);
        for (int i = 0; i < rows; i++)
        {
            if (heightBand->RasterIO(GF_Read, 0, i, nXSize, 1,
                pafScanline, nXSize, 1, GDT_Float32,
                0, 0) == CE_Failure)
            {
                std::cerr << "GPSPlugin::openImage: GDALRasterBand::RasterIO failed" << std::endl;
                break;
            }
            memcpy(&(rasterData[(i*cols)]), pafScanline, nXSize * sizeof(float));
        }

        if (heightBand->ReadBlock(0, 0, rasterData) == CE_Failure)
        {
            std::cerr << "GPSPlugin::openImage: GDALRasterBand::ReadBlock failed" << std::endl;
            return;
        }

                adfMinMax[0] = heightBand->GetMinimum(&bGotMin);
                adfMinMax[1] = heightBand->GetMaximum(&bGotMax);
                if (!(bGotMin && bGotMax))
                        GDALComputeRasterMinMax((GDALRasterBandH)heightBand, TRUE, adfMinMax);

                printf("Min=%.3fd, Max=%.3f\n", adfMinMax[0], adfMinMax[1]);

        }
}

float GPSPlugin::getAlt(double x, double y)
{
    int col = int((x - xOrigin) / pixelWidth);
    int row = int((yOrigin - y) / pixelHeight);
    if (col < 0)
        col = 0;
    if (col >= cols)
        col = cols-1;
    if (row < 0)
        row = 0;
    if (row >= rows)
        row = rows - 1;
    return rasterData[col + (row*cols)];
        float *pafScanline;
        int   nXSize = heightBand->GetXSize();

    delete[] pafScanline;
        pafScanline = new float[nXSize];
        auto err = heightBand->RasterIO(GF_Read, x, y, 1, 1,
                pafScanline, nXSize, 1, GDT_Float32,
                0, 0);
        float height = pafScanline[0];
        delete[] pafScanline;

    if (err != CE_None)
    {
        std::cerr << "MapDrape::getAlt: error" << std::endl;
        return 0.f;
    }

        return height;
}
//GPS fileHandler
int GPSPlugin::SloadGPX(const char *filename, osg::Group *parent, const char *)
{
    instance()->loadGPX(filename, parent);
    return 0;
}
int GPSPlugin::loadGPX(const char *filename, osg::Group *parent)
{
    File *f = new File(filename);
    this->addFile(f);
    return 0;
}

int GPSPlugin::SunloadGPX(const char *filename, const char *)
{
    return GPSPlugin::instance()->unloadGPX(filename);
}
int GPSPlugin::unloadGPX(const char *filename)
{
    return 0;
}


COVERPLUGIN(GPSPlugin)
