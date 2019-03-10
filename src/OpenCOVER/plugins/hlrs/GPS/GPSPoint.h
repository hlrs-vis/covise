/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _GPS_PLUGIN_GPSPOINT_H
#define _GPS_PLUGIN_GPSPOINT_H
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
#include <xercesc/dom/DOM.hpp>

#include <osg/MatrixTransform>
#include <osg/ShapeDrawable>
#include <osgText/Text>
#include <cover/coBillboard.h>
#include <osg/Material>

#include <vrml97/vrml/Player.h>



using namespace opencover;
using namespace covise;

namespace vrui
{
    class coNavInteraction;
}
class PointSensor;

//Single Point
class GPSPoint
{
public:
    enum pointType {Good, Medium ,Bad,Angst,Text,Foto,Sprachaufnahme,Barriere, OtherChoice};
    pointType PT;
private:
    double longitude;
    double latitude;
    double altitude;
    double time;
    float speed;
    std::string text;
    std::string filename;
    std::string myDirectory;

public:
    GPSPoint(std::string directory);
    ~GPSPoint();  
    void setPointData (double x, double y, double z, double time, float v, std::string &name);
    void setIndex(int i);
    void draw();
    void createSphere(osg::Vec4 *colVec);
    void createDetail();
    void createSign(osg::Image *img);
    void createBillboard();
    void createText();
    void createPicture();
    void createSound();
    void readFile(xercesc::DOMElement *node);
    osg::ref_ptr<osg::Group> Point;
    osg::ref_ptr<osg::MatrixTransform> geoTrans;
    osg::ref_ptr<osg::MatrixTransform> geoScale;
    osg::ref_ptr<osg::Switch> switchSphere;
    osg::ref_ptr<osg::Switch> switchDetail;
    osg::ref_ptr<coBillboard> BBoard;
    osg::ref_ptr<osg::Geode> PictureGeode;
    osg::ref_ptr<osg::Geode> TextGeode;
    osg::ref_ptr<osg::Image> img;


    osg::ref_ptr<osg::Sphere> sphere;
    osg::ref_ptr<osg::Geode> PointSphere;
    osg::ref_ptr<osg::ShapeDrawable> sphereD;

    osg::ref_ptr<osg::Material> streetmarkMaterial;

    vrml::Audio *audio;
    vrml::Player::Source *source;


    PointSensor *mySensor;
    void activate();
    void disactivate();
    void update();
};


#endif
