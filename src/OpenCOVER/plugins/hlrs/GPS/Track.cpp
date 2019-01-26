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


#include <osg/Group>
#include <osg/Geode>
#include <osg/Geometry>
#include <osg/BlendFunc>
#include <osg/AlphaFunc>
#include <osg/LineWidth>
#include <cover/RenderObject.h>
#include <cover/coVRPluginSupport.h>

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


osg::ref_ptr<osg::Material> Track::globalDefaultMaterial;

//Track
Track::Track()
{
    SingleTrack = new osg::Group();
    //fprintf(stderr, "Track created\n");
}
Track::~Track()
{
    fprintf(stderr, "Track deleted\n");
}
void Track::setIndex(int i)
{
    SingleTrack->setName("Track Nr.: " + std::to_string(i));
}
void Track::addPoint(double x, double y, double v)
{
    double z = GPSPlugin::instance()->getAlt(x,y)+GPSPlugin::instance()->zOffset;

    x *= DEG_TO_RAD;
    y *= DEG_TO_RAD;

    int error = pj_transform(GPSPlugin::instance()->pj_from, GPSPlugin::instance()->pj_to, 1, 1, &x, &y, NULL);
    if(error !=0 )
    {
        fprintf(stderr, "------ \nError transforming coordinates, code %d \n", error);
        fprintf (stderr, "%s \n ------ \n", pj_strerrno (error));
    }
    std::array<double, 4> arr = {x, y, z, v };
    PointsVec.push_back(arr);
    //fprintf (stderr, "PointsVec size: %i\n", PointsVec.size());
}
void Track::drawBirdView()
{
    //for (std::list<GPSPoint*>::iterator it = TrackPoints.begin(); it != TrackPoints.end(); it++){
    //    (*it)->drawSphere();
    //}
}

void Track::readFile (xercesc::DOMElement *node)
{
    fprintf(stderr, "Trackreading started\n");

    xercesc::DOMNodeList *TrackNodeList = node->getChildNodes();
    int TrackNodeLength = TrackNodeList->getLength();
    for (int i = 0; i < TrackNodeLength; ++i)
    {
        xercesc::DOMElement *TrackNode = dynamic_cast<xercesc::DOMElement *>(TrackNodeList->item(i));
        if (!TrackNode)
            continue;
        char *tmp = xercesc::XMLString::transcode(TrackNode->getNodeName());
        std::string nodeContentName = tmp;
        xercesc::XMLString::release(&tmp);
        if(nodeContentName == "trkseg")
        {
            fprintf(stderr, "trkseg found\n");

            int TrackPointLength = TrackNode->getChildNodes()->getLength();
            PointsVec.reserve(TrackPointLength);
            for (xercesc::DOMNode *currentNode = TrackNode->getFirstChild() ; currentNode != NULL; currentNode = currentNode->getNextSibling())
            {
                xercesc::DOMElement *TrackPoint = dynamic_cast<xercesc::DOMElement *>(currentNode);
                if (!TrackPoint)
                    continue;
                char *tmp2 = xercesc::XMLString::transcode(TrackPoint->getNodeName());
                std::string pointContentName = tmp2;
                xercesc::XMLString::release(&tmp);

                if(pointContentName == "trkpt")           
                {
                    //fprintf(stderr, "trkpt found\n");
                    double x;
                    double y;
                    //double t;
                    double v;
                    XMLCh *t1 = NULL;

                    char *lon = xercesc::XMLString::transcode(TrackPoint->getAttribute(t1 = xercesc::XMLString::transcode("lon"))); xercesc::XMLString::release(&t1);
                    char *lat = xercesc::XMLString::transcode(TrackPoint->getAttribute(t1 = xercesc::XMLString::transcode("lat"))); xercesc::XMLString::release(&t1);

                    sscanf(lon, "%lf", &x);
                    sscanf(lat, "%lf", &y);

                    //fprintf(stderr, "read from file:   lon: %s , lat: %s\n",lon, lat);

                    xercesc::XMLString::release(&lat);
                    xercesc::XMLString::release(&lon);

                    xercesc::DOMNodeList *nodeContentList = TrackPoint->getChildNodes();
    int nodeContentLength = nodeContentList->getLength();
                    for (int i = 0; i < nodeContentLength; ++i)
                    {
                        xercesc::DOMElement *nodeContent = dynamic_cast<xercesc::DOMElement *>(nodeContentList->item(i));
                        if (!nodeContent)
                            continue;
                        char *tmp = xercesc::XMLString::transcode(nodeContent->getNodeName());
                        std::string nodeContentName = tmp;
                        xercesc::XMLString::release(&tmp);

                        //if(nodeContentName == "time")
                        //{
                        //    char *time = xercesc::XMLString::transcode(nodeContent->getTextContent());
                        //    //fprintf(stderr, "read from file:   time: %s\n",time);
                        //    sscanf(time, "%lf", &t);
                        //    xercesc::XMLString::release(&time);
                        //}
                        if(nodeContentName == "extensions")
                        {
                            xercesc::DOMNodeList *extensionsList = nodeContent->getChildNodes();
                            int extensionsLength = extensionsList->getLength();
                            for (int k = 0; k < extensionsLength; ++k)
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
                                    //fprintf(stderr, "read from file:   speed: %s\n",speed);
                                    sscanf(speed, "%lf", &v);
                                    xercesc::XMLString::release(&speed);
                                }
                                else {
                                    fprintf(stderr, "unknown extension node named: %s\n",nodeContentName.c_str() );
                                }
                            }
                        }
                        else {
                            //fprintf(stderr, "unknown content node named: %s\n",nodeContentName.c_str() );
                        }

                    }
                    addPoint(x, y, v);

                }
                else {
                    fprintf(stderr, "unknown content node in trkseg named %s\n",nodeContentName.c_str() );
                }
            }
        }

        else if(nodeContentName == "name")
        {
            fprintf(stderr, "unused node in trkseg called: %s\n",nodeContentName.c_str());
        }
        else {
            fprintf(stderr, "unknown content node in trk named %s\n",nodeContentName.c_str() );
        }
    }
    fprintf(stderr, "Trackreading finished. Trackpoints: %lu\n",(unsigned long)PointsVec.size() );
}

void Track::drawTrack()
{
    //fprintf(stderr, "drawTrack called\n");

    static double AlphaThreshold = 0.5;
    float linewidth = 4.0f;

    osg::Geode *geode = new osg::Geode();
    osg::Geometry *geom = new osg::Geometry();

    cover->setRenderStrategy(geom);

    //Setup geometry
    osg::Vec3Array *vert = new osg::Vec3Array;
    for (int i = 0; i < PointsVec.size(); ++i)
    {
        std::array<double, 4> *a1 = &PointsVec.at(i);
        vert->push_back(osg::Vec3(a1->at(0), a1->at(1) , a1->at(2)));
    }
    geom->setVertexArray(vert);

    //color
    osg::Vec4Array *colArr = new osg::Vec4Array();
    for (int t = 0; t < PointsVec.size(); ++t)
    {
        std::array<double, 4> *a1 = &PointsVec.at(t);
        float s = a1->at(3);
        colArr->push_back(osg::Vec4(1.0f, 1.0f/(1.0f+s), (1.0f+s), 1.0f));
    }
    geom->setColorArray(colArr);
    geom->setColorBinding(osg::Geometry::BIND_PER_VERTEX );

    //primitves
    osg::DrawArrayLengths *primitives = new osg::DrawArrayLengths(osg::PrimitiveSet::LINE_STRIP);
    primitives->push_back(PointsVec.size());
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
    SingleTrack->addChild(geode);
}


