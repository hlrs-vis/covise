/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef PLMXML_PARSER_H
#define PLMXML_PARSER_H

#ifndef STANDALONE
#include <util/common.h>
#endif

#include <xercesc/dom/DOM.hpp>
#include <xercesc/dom/DOMDocument.hpp>
#include <xercesc/dom/DOMDocumentType.hpp>
#include <xercesc/dom/DOMElement.hpp>
#include <xercesc/dom/DOMImplementation.hpp>
#include <xercesc/dom/DOMImplementationLS.hpp>
#include <xercesc/dom/DOMNodeIterator.hpp>
#include <xercesc/dom/DOMNodeList.hpp>
#include <xercesc/dom/DOMText.hpp>
#include <xercesc/parsers/XercesDOMParser.hpp>
#include <xercesc/util/XMLUni.hpp>
#include <xercesc/util/XMLString.hpp>

#include <osg/Group>
#include <osg/Node>
#include <osg/MatrixTransform>

#include <string>
#include <iostream>
#include <sstream>
#include <iterator>
#include <vector>

struct ltstr
{
    bool operator()(const char *s1, const char *s2) const
    {
        return strcmp(s1, s2) < 0;
    }
};

using namespace xercesc;

class ElementParser
{

public:
    ElementParser()
    {
        instantiated = false;
    };
    DOMElement *element;
    osg::Group *group;
    bool instantiated;
};

class PLMXMLParser
{

public:
    PLMXMLParser();
    ~PLMXMLParser();
    bool parse(const char *fileName, osg::Group *parent);

    xercesc::DOMDocument *parse(const char *filename, xercesc::DOMDocument *document);

    void addInstance(char *id, osg::Group *parent);
    void addPart(char *id, osg::Group *parent);
    char *filePath;

private:
    void getChildrenPath(DOMElement *node, const char *path, std::vector<DOMNode *> *result);
    osg::MatrixTransform *getTransformNode(const char *id, DOMElement *node);
    const XMLCh *getTransform(DOMElement *node);

    xercesc::XercesDOMParser *m_Parser;
    XMLCh *TAG_location;
    XMLCh *TAG_simulation;
    XMLCh *TAG_format;
    XMLCh *TAG_unit;
    XMLCh *TAG_partRef;
    XMLCh *TAG_id;
    XMLCh *TAG_name;
    XMLCh *TAG_Transform;
    XMLCh *TAG_Representation;
    XMLCh *TAG_CompoundRepresentation;
    XMLCh *TAG_SimRep;
    XMLCh *TAG_InstanceRefs;
    XMLCh *TAG_RootRefs;
    XMLCh *TAG_RootInstanceRef;
    XMLCh *TAG_UserValue;
    XMLCh *TAG_UserData;
    XMLCh *TAG_value;
    XMLCh *TAG_title;

    std::map<char *, osg::Node *, ltstr> files;
    std::map<char *, ElementParser *, ltstr> instances;
    std::map<char *, ElementParser *, ltstr> parts;

    std::vector<std::string> split(const std::string &s)
    {
        std::istringstream iss(s);
        return std::vector<std::string>((std::istream_iterator<std::string>(iss)),
                                        std::istream_iterator<std::string>());
    }
};

#endif
