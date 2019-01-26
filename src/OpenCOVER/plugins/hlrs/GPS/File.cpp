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
#include "File.h"
#include "GPSPoint.h"
#include "Track.h"
#include "GPSALLPoints.h"
#include "GPSALLTracks.h"

#include <osg/Group>
#include <osg/Switch>

#include <xercesc/dom/DOM.hpp>
#if XERCES_VERSION_MAJOR < 3
#include <xercesc/dom/DOMWriter.hpp>
#else
#include <xercesc/dom/DOMLSSerializer.hpp>
#endif
#include <xercesc/framework/LocalFileFormatTarget.hpp>
#include <xercesc/util/XMLUni.hpp>
#include <xercesc/parsers/XercesDOMParser.hpp>

//File
File::File()
{
    fprintf(stderr, "File created\n");
}
File::File(const char *filename)
{
    name = filename;
    size_t found = name.find_last_of("/");
    FileGroup = new osg::Group();
    FileGroup->setName("File: " + name.substr(found+1));
    //OSGGPSPlugin->addChild(FileGroup);
    //if (parent)
    //    parent->addChild(FileGroup);
    //else
    //GPSPlugin::instance()->OSGGPSPlugin->addChild(FileGroup);

    SwitchPoints = new osg::Switch();
    SwitchPoints->setName("SwitchNode for Points");
    SwitchTracks = new osg::Switch();
    SwitchTracks->setName("SwitchNode for Tracks");

    FileGroup->addChild(SwitchPoints);
    FileGroup->addChild(SwitchTracks);

    readFile(filename);
    //fprintf(stderr, "File with name %s created\n", name.c_str());
}
File::~File()
{
    for (auto *x : fileAllTracks){
        delete x;
    }
    for (auto *x : fileAllPoints){
        delete x;
    }
    //SwitchTracks.release();
    //SwitchPoints.release();
    //FileGroup.release();

    fprintf(stderr, "File deleted\n");
}
void File::addAllTracks(GPSALLTracks *at)
{
    fileAllTracks.push_back(at);
    SwitchTracks->addChild(at->TrackGroup);
    //fprintf(stderr, "AllTracks added to file\n");
}
void File::addAllPoints(GPSALLPoints *ap)
{
    fileAllPoints.push_back(ap);
    SwitchPoints->addChild(ap->PointGroup);
    //fprintf(stderr, "AllPoints added to file\n");
}

//fileReader
void File::readFile(const std::string &filename)
{
    xercesc::XercesDOMParser *parser = new xercesc::XercesDOMParser();
    parser->setValidationScheme(xercesc::XercesDOMParser::Val_Never);

    try
    {
        parser->parse(filename.c_str());
        fprintf(stderr, "Parsing file: %s\n", filename.c_str());
    }
    catch (...)
    {
        fprintf(stderr, "Error parsing file: %s\n", filename.c_str());
    }

    xercesc::DOMDocument *xmlDoc = parser->getDocument();
    xercesc::DOMElement *rootElement = NULL;

    if (xmlDoc)
    {
        rootElement = xmlDoc->getDocumentElement();
    }
    else{
         fprintf(stderr, "Could not open file: %s\n", filename.c_str());
    }

    if (rootElement)
    {
        fprintf(stderr, "-----------\nStarted reading file: %s \n-----------\n", this->name.c_str());
        //fprintf(stderr, "rootElement %s\n", xercesc::XMLString::transcode(rootElement->getNodeName()));
        xercesc::DOMNodeList *nodeList = rootElement->getChildNodes();

        GPSALLPoints *ap = new GPSALLPoints();
        GPSALLTracks *at = new GPSALLTracks();
        this->addAllTracks(at);
        this->addAllPoints(ap);

        for (int o = 0; o < nodeList->getLength(); ++o)
        {
            xercesc::DOMElement *node = dynamic_cast<xercesc::DOMElement *>(nodeList->item(o));
            if (!node)
                continue;
            char *tmp = xercesc::XMLString::transcode(node->getNodeName());
            std::string nodeName = tmp;
            xercesc::XMLString::release(&tmp);

            if(nodeName == "wpt")
            {
                GPSPoint *p = new GPSPoint();
                ap->addPoint(p);
                p->readFile(node);
            }
            else if(nodeName == "trk")
            {
                Track *t = new Track();
                at->addTrack(t);
                t->readFile(node);
            }
            else
            {
                fprintf(stderr, "unknown node type %s\n",nodeName.c_str() );
            }
        }

        ap->drawBirdView();
        at->drawBirdView();

        fprintf(stderr, "-----------\nFinished reading file: %s \n", this->name.c_str());
        fprintf(stderr, "GPSPoints added: %lu , Tracks added: %lu \n-----------\n",(unsigned long)fileAllPoints.front()->allPoints.size(), (unsigned long)fileAllTracks.front()->allTracks.size() );

    }
}
