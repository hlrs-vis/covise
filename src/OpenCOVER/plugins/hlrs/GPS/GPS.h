/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _GPS_PLUGIN_H
#define _GPS_PLUGIN_H
/****************************************************************************\ 
 **                                                            (C)2008 HLRS  **
 **                                                                          **
 ** Description: GPS OpenCOVER Plugin (is polite)                            **
 **                                                                          **
 **                                                                          **
 ** Author: U.Woessner, T.Gudat		                                     **
 **                                                                          **
 ** History:  								     **
 ** June 2008  v1	    				       		     **
 **                                                                          **
 **                                                                          **
\****************************************************************************/
#include <cover/coVRPlugin.h>
#include <cover/ui/Menu.h>
#include <cover/ui/Button.h>
#include <cover/ui/SelectionList.h>
#include <cover/ui/Label.h>
#include <cover/ui/EditField.h>
#include <cover/ui/Owner.h>
#include <proj_api.h>
#include <gdal_priv.h>
#include <xercesc/dom/DOM.hpp>

using namespace opencover;
using namespace covise;
namespace opencover
{
class coVRLabel;
}
class GPSPoint;
class Track;
class GPSAllPoints;
class GPSALLTracks;

class GPSPlugin : public opencover::coVRPlugin , public ui::Owner
{
public:
    GPSPlugin();
    ~GPSPlugin();
    static GPSPlugin *instance(){return plugin;};

    float getAlt(double x, double y);

    static int SloadGPX(const char *filename, osg::Group *parent, const char *);
    static int SunloadGPX(const char *filename, const char *);

    ui::Menu *GPSTab = nullptr;
    ui::Label *infoLabel = nullptr;
    ui::Button *ToggleLoad = nullptr;
    ui::Button *ToggleTracks = nullptr;
    ui::Button *ToggleEmotions = nullptr;
    ui::Button *TogglePictures = nullptr;
    ui::Button *ToggleAudio = nullptr;
    ui::Button *ToggleText = nullptr;
    osg::ref_ptr<osg::Group> OSGGPSPlugin;
    coVRLabel *Label;
    float zOffset=4.0;

    std::string dir;//Coordinates
    projPJ pj_from, pj_to;//Coordinates


    
private:
    static GPSPlugin *plugin;
    bool update();
    void closeImage();
    void openImage(std::string &name);
    int loadGPX(const char *filename, osg::Group *parent);
    int unloadGPX(const char *filename);
    void GPSTab_create();
    void GPSTab_delete();

    float *rasterData=NULL;
    double xOrigin; // origin of the height map
    double yOrigin;
    double pixelWidth; // size of each pixel
    double pixelHeight;
    int cols; // number of pixel rows and columns
    int rows;
    GDALDataset  *heightDataset;
    GDALRasterBand  *heightBand;

};

#endif
