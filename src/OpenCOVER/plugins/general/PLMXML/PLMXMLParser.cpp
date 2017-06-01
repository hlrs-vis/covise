/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include <PluginUtil/FileReference.h>
#include <PluginUtil/SimReference.h>
#include "PLMXMLParser.h"
#include <cstring>
#include <util/unixcompat.h>

#ifdef STANDALONE
using namespace std;
int main(int argc, char **argv)
{

    if (argc != 2)
    {

        std::cerr << "Usage: " << argv[0] << " filename" << std::endl;
        exit(1);
    }

    PLMXMLParser *parser = new PLMXMLParser();
    parser->parse(argv[1], NULL);
}
#else
#include <cover/coVRFileManager.h>
#endif

PLMXMLParser::PLMXMLParser()
{

    m_Parser = NULL;

    doLoadAll = true;
    doLoadVRML = true;
    doLoadSTL = false;
    doUndoVRMLRotate = true;

    try
    {
        XMLPlatformUtils::Initialize();
    }
    catch (XMLException & /*e*/)
    {
        std::cerr << "failed to initialize xerces" << std::endl;
        return;
    }
    m_Parser = new XercesDOMParser();
    m_Parser->setValidationScheme(XercesDOMParser::Val_Never);
    m_Parser->setDoNamespaces(false);
    m_Parser->setDoSchema(false);
    m_Parser->setLoadExternalDTD(false);

    TAG_location = XMLString::transcode("location");
    TAG_simulation = XMLString::transcode("simulation");
    TAG_format = XMLString::transcode("format");
    TAG_unit = XMLString::transcode("unit");
    TAG_partRef = XMLString::transcode("partRef");
    TAG_id = XMLString::transcode("id");
    TAG_name = XMLString::transcode("name");
    TAG_Transform = XMLString::transcode("Transform");
    TAG_Representation = XMLString::transcode("Representation");
    TAG_CompoundRepresentation = XMLString::transcode("CompoundRep");
    TAG_SimRep = XMLString::transcode("SimRep");
    TAG_InstanceRefs = XMLString::transcode("instanceRefs");
    TAG_RootRefs = XMLString::transcode("rootRefs");
    TAG_RootInstanceRef = XMLString::transcode("rootInstanceRef");
}

PLMXMLParser::~PLMXMLParser()
{
}

bool PLMXMLParser::parse(const char *fileName, osg::Group *group)
{
    //setting the path of the working directory
    std::string tmpPath(fileName);
    std::string tmpPath_2;
    int slash = tmpPath.find_last_of("/");
    if (slash != -1)
    {
        tmpPath_2 = tmpPath.substr(0, tmpPath.find_last_of("/") + 1);
    }
    slash = tmpPath.find_last_of("\\");
    if (slash != -1)
    {
        tmpPath_2 = tmpPath.substr(0, tmpPath.find_last_of("\\") + 1);
    }

    filePath = new char[tmpPath_2.size() + 1];
    strcpy(filePath, tmpPath_2.c_str());

    if (!m_Parser)
        return false;

    m_Parser->parse(fileName);
    xercesc::DOMDocument *xmlDoc = m_Parser->getDocument();
    if (xmlDoc)
    {

        try
        {

            /*
         xercesc::DOMElement* elementRoot = xmlDoc->getDocumentElement();
         std::vector<DOMNode *> productList;
         getChildrenPath(elementRoot, "ProductDef", &productList);

         int numProducts = productList.size();

         for (int index = 0; index < numProducts; index ++) {
            xercesc::DOMElement *product = dynamic_cast<DOMElement *>(productList[index]);
            if (product) {

               std::vector<DOMNode *> instanceGraphList;
               getChildrenPath(product, "InstanceGraph", &instanceGraphList);
               int numIGs = instanceGraphList.size();
               for (int IGindex = 0; IGindex < numIGs; IGindex ++)
               {   
                  xercesc::DOMElement *instanceGraph = dynamic_cast<DOMElement *>(instanceGraphList[IGindex]);
                  if (instanceGraph)
                  {
                      parseInstanceGraph(instanceGraph, fileName, group);
                  }
               }
            }
         }
         */

            xercesc::DOMNodeList *allIGs;
            allIGs = xmlDoc->getElementsByTagName(XMLString::transcode("InstanceGraph"));
            for (int i = 0; i < int(allIGs->getLength()); i++)
            {
                xercesc::DOMElement *instanceGraph = dynamic_cast<DOMElement *>(allIGs->item(i));
                if (instanceGraph)
                {
                    parseInstanceGraph(instanceGraph, fileName, group);
                }
            }
        }
        catch (xercesc::XMLException &e)
        {
            char *message = XMLString::transcode(e.getMessage());
            std::cerr << "Error parsing file: " << message << std::endl;
            xercesc::XMLString::release(&message);
        }
    }
    else
    {
        std::cerr << "Could not open file: " << fileName << std::endl;
        return false;
    }
    return true;
}

void PLMXMLParser::parseInstanceGraph(xercesc::DOMElement *instanceGraph, const char *fileName, osg::Group *group)
{
    std::vector<DOMNode *> instanceList;
    std::vector<DOMNode *> partList;

    getChildrenPath(instanceGraph, "Instance", &instanceList);
    getChildrenPath(instanceGraph, "ProductInstance", &instanceList);
    getChildrenPath(instanceGraph, "Part", &partList);
    getChildrenPath(instanceGraph, "ProductRevisionView", &partList);
    if (instanceList.size() && partList.size())
    {
        int numInstances = instanceList.size();
        int numParts = partList.size();

        for (int index = 0; index < numInstances; index++)
        {
            xercesc::DOMElement *inst = dynamic_cast<DOMElement *>(instanceList[index]);
            if (inst)
            {

                osg::Group *g = new osg::Group();
                char *name = NULL;
                char *id = NULL;

                if (inst->hasAttribute(TAG_name))
                    name = XMLString::transcode(inst->getAttribute(TAG_name));

                Element *instance = new Element();
                instance->group = g;
                instance->element = inst;

                id = XMLString::transcode(inst->getAttribute(TAG_id));
                instances[id] = instance;

                if (name)
                    g->setName(name);
                else
                    g->setName(id);
            }
        }
        for (int index = 0; index < numParts; index++)
        {
            xercesc::DOMElement *part = dynamic_cast<DOMElement *>(partList[index]);
            if (part)
            {

                osg::Group *g = new osg::Group();
                char *name = NULL;
                char *id = NULL;

                if (part->hasAttribute(TAG_name))
                    name = XMLString::transcode(part->getAttribute(TAG_name));

                Element *p = new Element();
                p->group = g;
                p->element = part;

                id = XMLString::transcode(part->getAttribute(TAG_id));
                parts[id] = p;

                if (name)
                    g->setName(name);
                else
                    g->setName(id);
            }
        }

        if (instanceGraph)
        {
            const XMLCh *rootRefs = 0;

            if (instanceGraph->hasAttribute(TAG_RootRefs))
                rootRefs = instanceGraph->getAttribute(TAG_RootRefs);

            if (instanceGraph->hasAttribute(TAG_RootInstanceRef))
                rootRefs = instanceGraph->getAttribute(TAG_RootInstanceRef);

            if (rootRefs != 0)
            {
                char *t = XMLString::transcode(rootRefs);
                std::vector<std::string> in = split(std::string(t));
                std::vector<std::string>::iterator it;
                for (it = in.begin(); it != in.end(); it++)
                {
                    addInstance((char *)(*it).c_str(), group);

                    std::map<char *, Element *, ltstr>::iterator i = instances.find((char *)(*it).c_str());
                    if (i != instances.end())
                    {
                        Element *instance = i->second;
                        osg::Group *g = instance->group;
                        std::string name = fileName + g->getName();
                        g->setName(name);
                    }
                }
                XMLString::release(&t);
            }
        }
    }
}

void PLMXMLParser::addInstance(char *id, osg::Group *parent)
{

    std::map<char *, Element *, ltstr>::iterator it = instances.find(id);
    if (it != instances.end())
    {

        Element *instance = it->second;

        osg::Group *group = instance->group;
        DOMElement *element = instance->element;

        parent->addChild(group);
        coVRSelectionManager::markAsHelperNode(group);

        osg::MatrixTransform *transformNode = getTransformNode(id, element);
        if (transformNode)
        {
            group->addChild(transformNode);
            group = transformNode;
        }
        else
        {
            if (element->hasAttribute(TAG_name))
                group->setName(XMLString::transcode(element->getAttribute(TAG_name)));
            else
                group->setName(id);
        }

        const XMLCh *partRef = element->getAttribute(TAG_partRef);
        char *partName = XMLString::transcode(partRef);
        if (partName[0] == '#')
            addPart(partName + 1, group);
        else
            addPart(partName, group);
        XMLString::release(&partName);
    }
}

void PLMXMLParser::addPart(char *id, osg::Group *parent)
{

    std::map<char *, Element *, ltstr>::iterator it = parts.find(id);
    if (it != parts.end())
    {
        Element *part = it->second;

        osg::Group *group = part->group;
        DOMElement *element = part->element;

        parent->addChild(group);
        if (part->instantiated == false)
        {
            part->instantiated = true;

            xercesc::DOMNodeList *reprList = element->getElementsByTagName(TAG_Representation);
            if (reprList && reprList->getLength() > 0)
            {
                int rep = 0;
                do
                {
                    xercesc::DOMElement *repr = dynamic_cast<xercesc::DOMElement *>(reprList->item(rep));

                    if (repr && (repr->hasAttribute(TAG_location)))
                    {
                        const XMLCh *file = repr->getAttribute(TAG_location);
                        char *fileName = XMLString::transcode(file);
                        const XMLCh *format = repr->getAttribute(TAG_format);
                        const XMLCh *unit = repr->getAttribute(TAG_unit);
                        if (format)
                        {
                            char *formatName = XMLString::transcode(format);
                            float scale = 1;
                            if (unit)
                            {
                                char *unitName = XMLString::transcode(unit);
                                if (strcasecmp(unitName, "m") == 0)
                                {
                                    scale = 1000.0;
                                }
                                else if (strcasecmp(unitName, "cm") == 0)
                                {
                                    scale = 100.0;
                                }
                                else if (strcasecmp(unitName, "mm") == 0)
                                {
                                    scale = 1000.0;
                                }
                            }

                            if (fileName && (strcasecmp(formatName, "STL") != 0 || doLoadSTL) && (strcasecmp(formatName, "VRML") != 0 || doLoadVRML) && /*!(strcasecmp(formatName,"STL")==0 && reprList->getLength() > 0) &&*/ !(strlen(fileName) > 6 && (strcasecmp(fileName + strlen(fileName) - 6, ".model") == 0)) && !(strlen(fileName) > 8 && (strcasecmp(fileName + strlen(fileName) - 8, ".CATPart") == 0)))
                            {

                                osg::MatrixTransform *node = new osg::MatrixTransform();
                                if (scale != 1.0)
                                {
                                    if (strcasecmp(formatName, "VRML") == 0 && doUndoVRMLRotate)
                                    {
                                        node->setMatrix(osg::Matrix::scale(scale, scale, scale) * osg::Matrix::rotate(-M_PI_2, osg::Vec3f(1, 0, 0)));
                                    }
                                    else
                                        node->setMatrix(osg::Matrix::scale(scale, scale, scale));
                                }
                                else
                                {
                                    if (strcasecmp(formatName, "VRML") == 0 && doUndoVRMLRotate)
                                    {
                                        node->setMatrix(osg::Matrix::rotate(M_PI_2, osg::Vec3f(1, 0, 0)));
                                    }
                                }
                                for (char *c = fileName; *c != '\0'; c++)
                                {
                                    if (*c == '\\')
                                        *c = '/';
                                }
                                if (fileName[0] == '/' || fileName[0] == '\\' || fileName[1] == ':')
                                {
                                    if (doLoadAll)
                                    {
                                        coVRFileManager::instance()->loadFile(fileName, NULL, node);
                                    }
                                    else
                                    {
                                        node->setUserData(new FileReference(fileName));
                                    }
                                }
                                else
                                {
                                    char *tmpFile = new char[strlen(filePath) + strlen(fileName) + 1];
                                    sprintf(tmpFile, "%s%s", filePath, fileName);
                                    if (doLoadAll)
                                    {
                                        coVRFileManager::instance()->loadFile(tmpFile, NULL, node);
                                    }
                                    else
                                    {
                                        node->setUserData(new FileReference(tmpFile));
                                    }
                                }

                                /*
						osg::Node *node = NULL;
						std::map<char *, osg::Node *, ltstr>::iterator it = files.find(fileName);
						if (it == files.end()) {
						std::cerr << "loading: " << fileName << std::endl;
						#ifndef STANDALONE
						node = coVRFileManager::instance()->loadFile(fileName, NULL, group);
						#endif
						files[fileName] = node;
						} else {
						node = it->second;
						group->addChild(node);
						}
						*/
                                // TODO: release strings when finished loading
                                //XMLString::release(&fileName);

                                node->setName(fileName);
                                group->addChild(node);
                            }
                        }
                    }
                    xercesc::DOMNodeList *creprList = repr->getElementsByTagName(TAG_CompoundRepresentation);
                    xercesc::DOMNodeList *simList = repr->getElementsByTagName(TAG_SimRep);
                    if (simList && simList->getLength() > 0)
                    {
                        int sim = 0;
                        do
                        {
                            xercesc::DOMElement *repr = dynamic_cast<xercesc::DOMElement *>(simList->item(sim));
                            if (repr && (repr->hasAttribute(TAG_simulation)))
                            {
                                const XMLCh *file = repr->getAttribute(TAG_simulation);
                                const XMLCh *name = repr->getAttribute(TAG_name);
                                char *simfileName = XMLString::transcode(file);
                                char *simName = XMLString::transcode(name);
                                std::string tmpstr(simfileName);
                                tmpstr.append("_Cad");
                                osg::Referenced *data = group->getUserData();
                                SimReference *refdata;
                                if ((refdata = dynamic_cast<SimReference *>(data)))
                                {
                                    refdata->addSim(tmpstr.c_str(), simName); //if group still has an SimReference Object
                                }
                                else
                                {
                                    refdata = new SimReference(tmpstr.c_str(), simName); //new SimReference Object
                                    group->setUserData(refdata);
                                }
                                //cout<<"############Parser found Sim:"<<simfileName<<"|||Name:"<<simName<<"||||Group: "<<group->getName()<<endl;
                            }
                            sim++;
                        } while (sim < int(simList->getLength()));
                    }

                    if (creprList && creprList->getLength() > 0)
                    {

                        int crep = 0;
                        do
                        {
                            osg::Group *parent = group;
                            xercesc::DOMElement *repr = dynamic_cast<xercesc::DOMElement *>(creprList->item(crep));
                            const char *repid = XMLString::transcode(repr->getTagName());

                            osg::MatrixTransform *transformNode = getTransformNode(repid, repr);
                            if (transformNode)
                            {
                                parent->addChild(transformNode);
                                parent = transformNode;
                            }

                            if (repr && (repr->hasAttribute(TAG_location)))
                            {
                                const XMLCh *file = repr->getAttribute(TAG_location);
                                char *fileName = XMLString::transcode(file);
                                const XMLCh *format = repr->getAttribute(TAG_format);
                                const XMLCh *unit = repr->getAttribute(TAG_unit);
                                if (format)
                                {
                                    char *formatName = XMLString::transcode(format);
                                    float scale = 1.0;
                                    if (unit)
                                    {
                                        char *unitName = XMLString::transcode(unit);
                                        if (strcasecmp(unitName, "m") == 0)
                                        {
                                            scale = 1000.0;
                                        }
                                    }

                                    if (fileName && (strcasecmp(formatName, "STL") != 0 || doLoadSTL) && (strcasecmp(formatName, "VRML") != 0 || doLoadVRML) && /*!(strcasecmp(formatName,"STL")==0 && creprList->getLength() > 0) && */ !(strlen(fileName) > 6 && (strcasecmp(fileName + strlen(fileName) - 6, ".model") == 0)) && !(strlen(fileName) > 8 && (strcasecmp(fileName + strlen(fileName) - 8, ".CATPart") == 0)))
                                    {
                                        osg::MatrixTransform *node = new osg::MatrixTransform();
                                        if (scale != 1.0)
                                        {
                                            if (strcasecmp(formatName, "VRML") == 0 && doUndoVRMLRotate)
                                            {
                                                node->setMatrix(osg::Matrix::scale(scale, scale, scale) * osg::Matrix::rotate(-M_PI_2, osg::Vec3f(1, 0, 0)));
                                            }
                                            else
                                                node->setMatrix(osg::Matrix::scale(scale, scale, scale));
                                        }
                                        else
                                        {
                                            if (strcasecmp(formatName, "VRML") == 0 && doUndoVRMLRotate)
                                            {
                                                node->setMatrix(osg::Matrix::rotate(M_PI_2, osg::Vec3f(1, 0, 0)));
                                            }
                                        }

                                        for (char *c = fileName; *c != '\0'; c++)
                                        {
                                            if (*c == '\\')
                                                *c = '/';
                                        }
                                        if (fileName[0] == '/' || fileName[0] == '\\' || fileName[1] == ':')
                                        {
                                            if (doLoadAll)
                                            {
                                                coVRFileManager::instance()->loadFile(fileName, NULL, node);
                                            }
                                            else
                                            {
                                                node->setUserData(new FileReference(fileName));
                                            }
                                        }
                                        else
                                        {
                                            char *tmpFile = new char[strlen(filePath) + strlen(fileName) + 1];
                                            sprintf(tmpFile, "%s%s", filePath, fileName);
                                            if (doLoadAll)
                                            {
                                                coVRFileManager::instance()->loadFile(tmpFile, NULL, node);
                                            }
                                            else
                                            {
                                                node->setUserData(new FileReference(tmpFile));
                                            }
                                        }

                                        node->setName(fileName);
                                        parent->addChild(node);
                                    }
                                }
                                //TODO:  hiding the node

                                /*
                     osg::Node *node = NULL;
                     std::map<char *, osg::Node *, ltstr>::iterator it = files.find(fileName);
                     if (it == files.end()) {
                        std::cerr << "loading: " << fileName << std::endl;
#ifndef STANDALONE
                        node = coVRFileManager::instance()->loadFile(fileName, NULL, parent);
#endif
                        files[fileName] = node;
                     } else {
                        node = it->second;
                        parent->addChild(node);
                     }
		 		      */
                                // TODO: release strings when finished loading
                                //XMLString::release(&fileName);
                            }

                            crep++;
                        } while (crep < int(creprList->getLength()));
                    }

                    rep++;
                } while (rep < int(reprList->getLength()));
            }
        }

        const XMLCh *instanceRefs = element->getAttribute(TAG_InstanceRefs);
        if (instanceRefs)
        {
            char *t = XMLString::transcode(instanceRefs);
            std::vector<std::string> in = split(std::string(t));
            std::vector<std::string>::iterator it;
            for (it = in.begin(); it != in.end(); it++)
                addInstance((char *)(*it).c_str(), group);

            XMLString::release(&t);
        }
    }
}

const XMLCh *PLMXMLParser::getTransform(DOMElement *node)
{

    xercesc::DOMNodeList *list = node->getElementsByTagName(TAG_Transform);
    if (list && list->getLength() > 0)
        return list->item(0)->getTextContent();
    return NULL;
}

osg::MatrixTransform *PLMXMLParser::getTransformNode(const char *id, DOMElement *node)
{

    osg::MatrixTransform *transformNode = NULL;
    const XMLCh *transform = getTransform(node);

    if (transform)
    {
        double m[16];
        char *t = XMLString::transcode(transform);

        transformNode = new osg::MatrixTransform();
        coVRSelectionManager::markAsHelperNode(transformNode);

        if (sscanf(t, "%lg %lg %lg %lg %lg %lg %lg %lg %lg %lg %lg %lg %lg %lg %lg %lg",
                   &m[0], &m[1], &m[2], &m[3], &m[4], &m[5], &m[6], &m[7],
                   &m[8], &m[9], &m[10], &m[11], &m[12], &m[13], &m[14], &m[15]) == 16)
        {
            m[12] *= 1000.0;
            m[13] *= 1000.0;
            m[14] *= 1000.0;

            osg::Matrix mat(m);
            transformNode->setMatrix(mat);

            if (node->hasAttribute(TAG_name))
                transformNode->setName(XMLString::transcode(node->getAttribute(TAG_name)));
            else
                transformNode->setName(id);
        }
        XMLString::release(&t);
    }

    return transformNode;
}

void PLMXMLParser::getChildrenPath(DOMElement *node, const char *path,
                                   std::vector<DOMNode *> *result)
{

    char prefix[128];
    memset(prefix, 0, 128);
    const char *index = strchr(path, '/');

    if (index == NULL)
    {
        XMLCh *tag = XMLString::transcode(path);

        xercesc::DOMNodeList *list = node->getElementsByTagName(tag);
        XMLString::release(&tag);
        int length = list->getLength();
        result->reserve(length);
        for (int i = 0; i < length; i++)
            result->push_back(list->item(i));
        return;
    }

    int length = index - path;

    if (length > 128)
    {
        return;
    }
    memcpy(prefix, path, length);

    XMLCh *tag = XMLString::transcode(prefix);
    xercesc::DOMNodeList *list = node->getElementsByTagName(tag);
    XMLString::release(&tag);
    if (list)
    {
        xercesc::DOMElement *e = dynamic_cast<DOMElement *>(list->item(0));
        if (e)
            getChildrenPath(e, index + 1, result);
    }
}
