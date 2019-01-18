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

#include "GPSPoint.h"
#include "GPS.h"

#include <cover/coBillboard.h>
#include <cover/coVRLabel.h>
#include <osg/Material>
#include <osg/Vec3>
#include <osg/Vec4>
#include <osg/ShapeDrawable>
#include <osg/Geode>

#include <proj_api.h>

#include <xercesc/dom/DOM.hpp>
#if XERCES_VERSION_MAJOR < 3
#include <xercesc/dom/DOMWriter.hpp>
#else
#include <xercesc/dom/DOMLSSerializer.hpp>
#endif
#include <xercesc/framework/LocalFileFormatTarget.hpp>
#include <xercesc/util/XMLUni.hpp>
#include <xercesc/parsers/XercesDOMParser.hpp>


// GPSPoint
GPSPoint::GPSPoint()
{
    Point = new osg::Geode();
}
GPSPoint::~GPSPoint()
{
    fprintf(stderr, "GPSPoint deleted\n");
}
GPSPoint::pointType GPSPoint::gettype (void)
{
    return PT;
}
void GPSPoint::setIndex(int i)
{
    Point->setName("Point Nr.: " + std::to_string(i));
}
float GPSPoint::getSpeed()
{
    return speed;
}

osg::Vec3 GPSPoint::getCoArray()
{
    osg::Vec3 cord(longitude,latitude,altitude);
    return(cord);
}
void GPSPoint::readFile (xercesc::DOMElement *node)
{
    double x;
    double y;
    double z;
    double t;
    float v;
    std::string typetext;
    int type;
    XMLCh *t1 = NULL;

    char *lon = xercesc::XMLString::transcode(node->getAttribute(t1 = xercesc::XMLString::transcode("lon"))); xercesc::XMLString::release(&t1);
    char *lat = xercesc::XMLString::transcode(node->getAttribute(t1 = xercesc::XMLString::transcode("lat"))); xercesc::XMLString::release(&t1);

    sscanf(lon, "%lf", &x);
    sscanf(lat, "%lf", &y);

    //fprintf(stderr, "read from file:   lon: %s\n",lon);
    //fprintf(stderr, "read from file:   lat: %s\n",lat);


    xercesc::DOMNodeList *nodeContentList = node->getChildNodes();
    for (int i = 0; i < nodeContentList->getLength(); ++i)
    {
        xercesc::DOMElement *nodeContent = dynamic_cast<xercesc::DOMElement *>(nodeContentList->item(i));
        if (!nodeContent)
            continue;
        char *tmp = xercesc::XMLString::transcode(nodeContent->getNodeName());
        std::string nodeContentName = tmp;
        xercesc::XMLString::release(&tmp);
        if(nodeContentName == "ele")
        {
            char *alt = xercesc::XMLString::transcode(nodeContent->getTextContent());
            //fprintf(stderr, "read from file:   ele: %s\n",alt);
            sscanf(alt, "%lf", &z);
            xercesc::XMLString::release(&alt);
        }
        else if(nodeContentName == "time")
        {
            char *time = xercesc::XMLString::transcode(nodeContent->getTextContent());
            //fprintf(stderr, "read from file:   time: %s\n",time);
            sscanf(time, "%lf", &t);
            xercesc::XMLString::release(&time);
        }
        else if(nodeContentName == "name")
        {
            char *name = xercesc::XMLString::transcode(nodeContent->getTextContent());
            //fprintf(stderr, "read from file:   name: %s\n",name);
            typetext = name;
            if(typetext.empty())
            {
                typetext = "NO MESSAGE";
            }
            xercesc::XMLString::release(&name);
        }
        else if(nodeContentName == "extensions")
        {
            xercesc::DOMNodeList *extensionsList = nodeContent->getChildNodes();
            for (int k = 0; k < extensionsList->getLength(); ++k)
            {
                xercesc::DOMElement *extensionNode = dynamic_cast<xercesc::DOMElement *>(extensionsList->item(k));
                if (!extensionNode)
                    continue;
                char *tmp = xercesc::XMLString::transcode(extensionNode->getNodeName());
                std::string extensionNodeName = tmp;
                xercesc::XMLString::release(&tmp);
                if(extensionNodeName == "speed")
                {
                    char *speed = xercesc::XMLString::transcode(extensionNode->getTextContent());
                    fprintf(stderr, "read from file:   speed: %s\n",speed);
                    sscanf(speed, "%f", &v);
                    xercesc::XMLString::release(&speed);
                }
                else {
                    fprintf(stderr, "unknown extension node named: %s\n",nodeContentName.c_str() );
                }
            }
        }
        else {
            fprintf(stderr, "unknown content node named: %s\n",nodeContentName.c_str() );
        }

    }
    this->setPointData(x, y, z, t, v, typetext);

    xercesc::XMLString::release(&lat);
    xercesc::XMLString::release(&lon);
}

void GPSPoint::setPointData (double x, double y, double z, double t, float v, std::string &name)
{
    altitude = GPSPlugin::instance()->getAlt(x,y)+GPSPlugin::instance()->zOffset;
    x *= DEG_TO_RAD;
    y *= DEG_TO_RAD;
    longitude = x;
    latitude = y;
    time = time;
    speed = v;

    int error = pj_transform(GPSPlugin::instance()->pj_from, GPSPlugin::instance()->pj_to, 1, 1, &longitude, &latitude, NULL);
    if(error !=0 )
    {
        fprintf(stderr, "------ \nError transforming coordinates, code %d \n", error);
        fprintf (stderr, "%s \n ------ \n", pj_strerrno (error));
    }

    if(name.empty())
    {
        PT = Trackpoint;
    }
    else if(name == "Good")
    {
        PT = Good;
    }
    else if(name == "Medium")
    {
        PT = Medium;
    }
    else if(name == "Bad")
    {
        PT = Bad;
    }
    else if(name == "Angst")
    {
        PT = Angst;
    }
    else if(name == "Foto")
    {
        PT = Foto;
    }
    else if(name == "Sprachaufnahme")
    {
        PT = Sprachaufnahme;
    }
    else if(name == "Barriere" || name == "Fussgaenger" || name == "Fahrrad" || name == "Öpnv" || name == "Miv")
    {
        PT = OtherChoice;
    }
    else {
        PT = Text;
        text = name;
    }
}
void GPSPoint::drawSphere()
{
    float Radius = 10.0f;

    osg::Vec4 *color = new osg::Vec4();

    switch (PT){
    case Trackpoint:
     *color = osg::Vec4(0.0f, 1.0f, 1.0f, 1.0f);
      Radius = 5.0f;
      break;
    case Good:
     *color = osg::Vec4(0.0f, 1.0f, 0.0f, 1.0f);
      break;
    case Medium:
     *color = osg::Vec4(1.0f, 1.0f, 0.0f, 1.0f);
      break;
    case Bad:
     *color = osg::Vec4(1.0f, 0.0f, 0.0f, 1.0f);
      break;
    case Angst:
     *color = osg::Vec4(0.5f, 1.0f, 1.0f, 1.0f);
      break;
    case Text:
     *color = osg::Vec4(0.0f, 0.0f, 1.0f, 1.0f);
      break;
    case Foto:
     *color = osg::Vec4(0.5f, 0.0f, 1.0f, 1.0f);
      break;
    case Sprachaufnahme:
     *color = osg::Vec4(0.0f, 0.5f, 1.0f, 1.0f);
      break;
    case OtherChoice:
     *color = osg::Vec4(0.5f, 0.5f, 1.0f, 1.0f);
      break;
    }


    osg::ref_ptr<osg::Material> material_sphere = new osg::Material();
    material_sphere->setDiffuse(osg::Material::FRONT_AND_BACK, *color);
    material_sphere->setAmbient(osg::Material::FRONT_AND_BACK, *color);
    osg::StateSet *stateSet;

    osg::ref_ptr<osg::Sphere> sphereG = new osg::Sphere(osg::Vec3(longitude, latitude , altitude), Radius);
    osg::ref_ptr<osg::ShapeDrawable> sphereD = new osg::ShapeDrawable(sphereG.get());
    stateSet = sphereD->getOrCreateStateSet();
    stateSet->setAttribute /*AndModes*/ (material_sphere.get(), osg::StateAttribute::PROTECTED);
    stateSet->setMode(GL_LIGHTING, osg::StateAttribute::ON);
    sphereD->setStateSet(stateSet);
    Point->addDrawable(sphereD.get());
    fprintf(stderr, "type %i\n", PT);
}

void GPSPoint::drawDetail()
{

}
