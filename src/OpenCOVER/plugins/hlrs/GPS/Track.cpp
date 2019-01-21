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
    PointsCoVec = new osg::Vec3Array;
    SingleTrack = new osg::Group();
    fprintf(stderr, "Track created\n");
}
Track::~Track()
{
    fprintf(stderr, "Track deleted\n");
}
void Track::setIndex(int i)
{
    SingleTrack->setName("Track Nr.: " + std::to_string(i));
}
void Track::addPoint(GPSPoint *p)
{
    p->setIndex(TrackPoints.size());
    TrackPoints.push_back(p);
    SingleTrack->addChild(p->Point);

    PointsCoVec->push_back(p->getCoArray());
    PointsSpeed.push_back(p->getSpeed());
}
void Track::drawBirdView()
{
    for (std::list<GPSPoint*>::iterator it = TrackPoints.begin(); it != TrackPoints.end(); it++){
        (*it)->drawSphere();
    }
}

void Track::readFile (xercesc::DOMElement *node)
{
    xercesc::DOMNodeList *TrackNodeList = node->getChildNodes();
    for (int i = 0; i < TrackNodeList->getLength(); ++i)
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

            xercesc::DOMNodeList *TrackPointList = TrackNode->getChildNodes();
            for (int k = 0; k < TrackPointList->getLength(); ++k)
            {
                xercesc::DOMElement *TrackPoint = dynamic_cast<xercesc::DOMElement *>(TrackPointList->item(k));
                if (!TrackPoint)
                    continue;
                char *tmp = xercesc::XMLString::transcode(TrackPoint->getNodeName());
                std::string pointContentName = tmp;
                xercesc::XMLString::release(&tmp);

                if(pointContentName == "trkpt")
                {
                    GPSPoint *p = new GPSPoint();
                    p->readFile(TrackPoint);
                    this->addPoint(p);
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

}

void Track::drawTrack()
{
    fprintf(stderr, "drawTrack called\n");

    static double AlphaThreshold = 0.5;
    float linewidth = 4.0f;

    osg::Geode *geode = new osg::Geode();
    osg::Geometry *geom = new osg::Geometry();

    cover->setRenderStrategy(geom);

    //Array of Points
    geom->setVertexArray(PointsCoVec);

    //color
    osg::Vec4Array *colArr = new osg::Vec4Array();

    for (std::list<float>::iterator it = PointsSpeed.begin(); it != PointsSpeed.end(); it++){
        colArr->push_back(osg::Vec4(1.0f, 1.0f/(1.0f+*it), 0.0f, 1.0f));
    }
    geom->setColorArray(colArr);
    geom->setColorBinding(osg::Geometry::BIND_PER_VERTEX );

    //primitves
    osg::DrawArrayLengths *primitives = new osg::DrawArrayLengths(osg::PrimitiveSet::LINE_STRIP);
    primitives->push_back(PointsCoVec->size());
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


