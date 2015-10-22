/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl-2.1.txt.

* License: LGPL 2+ */

#include <OpenScenarioBase.h>
#include <oscVariables.h>
#include <oscHeader.h>
#include <xercesc/parsers/XercesDOMParser.hpp>
#include <xercesc/dom/DOMImplementation.hpp>
#include <xercesc/dom/DOMLSSerializer.hpp>
#include <xercesc/dom/DOMLSOutput.hpp>

#include <xercesc/framework/LocalFileFormatTarget.hpp>
#include <xercesc/util/XercesVersion.hpp>
#include <iostream>


using namespace OpenScenario;


OpenScenarioBase::OpenScenarioBase():oscObjectBase()
{
    rootElement = NULL;
    parser = NULL;
    xmlDoc = NULL;
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
    xercesc::DOMImplementation *impl;
    impl = xercesc::DOMImplementation::getImplementation();
    xmlDoc = impl->createDocument(0, xercesc::XMLString::transcode("OpenScenario"), 0);
    rootElement = xmlDoc->getDocumentElement();
    writeToDOM(rootElement,xmlDoc);;
#if (XERCES_VERSION_MAJOR < 3)
    xercesc::DOMWriter *writer = impl->createDOMWriter();
#else
    xercesc::DOMLSSerializer *writer = ((xercesc::DOMImplementationLS *)impl)->createLSSerializer();
    // set the format-pretty-print feature
    if (writer->getDomConfig()->canSetParameter(xercesc::XMLUni::fgDOMWRTFormatPrettyPrint, true))
        writer->getDomConfig()->setParameter(xercesc::XMLUni::fgDOMWRTFormatPrettyPrint, true);
#endif

    xercesc::XMLFormatTarget *xmlTarget = new xercesc::LocalFileFormatTarget(fileName.c_str());

#if (XERCES_VERSION_MAJOR < 3)
    if (!writer->writeNode(xmlTarget, *document))
    {
        std::cerr << "OpenScenarioBase::writeXosc: Could not open file for writing!" << std::endl;
        delete xmlTarget;
        delete writer;
        return false;
    }
#else
    xercesc::DOMLSOutput *output = ((xercesc::DOMImplementationLS *)impl)->createLSOutput();
    output->setByteStream(xmlTarget);

    if (!writer->write(xmlDoc, output))
    {
        std::cerr << "OpenScenarioBase::writeXosc: Could not open file for writing!" << std::endl;
        delete xmlTarget;
        delete writer;
        return false;
    }

    delete output;
#endif

    delete xmlTarget;
    delete writer;
    return true;
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
