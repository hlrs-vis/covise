/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl-2.1.txt.

* License: LGPL 2+ */

#include <OpenScenarioBase.h>
#include <oscVariables.h>
#include <oscHeader.h>
#include <xercesc/parsers/XercesDOMParser.hpp>
#include <iostream>


using namespace OpenScenario;


OpenScenarioBase::OpenScenarioBase():oscObjectBase()
{
    rootElement = NULL;
    parser = NULL;
    oscFactories::instance();

    OSC_ADD_MEMBER(header);
    OSC_ADD_MEMBER(database);
    OSC_ADD_MEMBER(roadNetwork);
    OSC_ADD_MEMBER(environment);
    OSC_ADD_MEMBER(entities);
    OSC_ADD_MEMBER(storyboard);
    OSC_ADD_MEMBER(scenarioEnd);

    base=this;
}

OpenScenarioBase::~OpenScenarioBase()
{
    delete parser;
    try
    {
        xercesc::XMLPlatformUtils::Terminate();
    }
    catch (const xercesc::XMLException &toCatch)
    {
        char *message = xercesc::XMLString::transcode(toCatch.getMessage());
        std::cout << "Error during xerces termination! :\n" << message << std::endl;
        xercesc::XMLString::release(&message);
    }
}


int OpenScenarioBase::loadFile(std::string &fileName)
{
    if(getRootElement(fileName)==NULL)
        return false;
    else
    {
        return parseFromXML(rootElement);
    }
}
int OpenScenarioBase::saveFile(std::string &fileName, bool overwrite/* default false */)
{
    return false;
}

xercesc::DOMElement *OpenScenarioBase::getRootElement(std::string filename)
{
    try
    {
        xercesc::XMLPlatformUtils::Initialize();
    }
    catch (const xercesc::XMLException &toCatch)
    {
        char *message = xercesc::XMLString::transcode(toCatch.getMessage());
        std::cout << "Error during initialization! :\n" << message << std::endl;
        xercesc::XMLString::release(&message);
        return NULL;
    }

    parser = new xercesc::XercesDOMParser();
    parser->setValidationScheme(xercesc::XercesDOMParser::Val_Auto);

    try
    {
        parser->parse(filename.c_str());
    }
    catch (...)
    {
        std::cerr << "Couldn't parse OpenDRIVE XML-file " << filename << "!" << std::endl;
        return NULL;
    }

    xmlDoc = parser->getDocument();
    if (xmlDoc)
    {
        rootElement = xmlDoc->getDocumentElement();
    }

    return rootElement;
}

bool OpenScenarioBase::parseFromXML(xercesc::DOMElement *currentElement)
{
    return oscObjectBase::parseFromXML(currentElement);
}