/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl-2.1.txt.

* License: LGPL 2+ */

#include <OpenScenarioBase.h>
#include <oscVariables.h>
#include "oscSourceFile.h"

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

    OSC_OBJECT_ADD_MEMBER(header,"oscHeader");
    OSC_OBJECT_ADD_MEMBER(catalogs,"oscCatalogs");
    OSC_OBJECT_ADD_MEMBER(roadNetwork,"oscRoadNetwork");
    OSC_OBJECT_ADD_MEMBER(environment,"oscEnvironment");
    OSC_OBJECT_ADD_MEMBER(entities,"oscEntities");
    OSC_OBJECT_ADD_MEMBER(storyboard,"oscStoryboard");
    OSC_OBJECT_ADD_MEMBER(scenarioEnd,"oscScenarioEnd");
	OSC_OBJECT_ADD_MEMBER(test,"oscTest");

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
    xercesc::DOMImplementation *impl = xercesc::DOMImplementation::getImplementation();
    //oscSourceFile for OpenScenarioBase
    oscSourceFile *osbSourceFile;

    //create xmlDocs for every source file
    //
    for (int i = 0; i < srcFileVec.size(); i++)
    {
        std::string srcFileRootElement = srcFileVec[i]->getRootElementName();

        //set filename for main xosc file with root element "OpenScenario" to fileName
        if (srcFileRootElement == "OpenScenario")
        {
            srcFileVec[i]->setVariables(srcFileRootElement, fileName);
            osbSourceFile = srcFileVec[i];
        }

        xercesc::DOMDocument *xmlSrcDoc = impl->createDocument(0, xercesc::XMLString::transcode(srcFileRootElement.c_str()), 0);

        if (!srcFileVec[i]->getXmlDoc())
        {
            srcFileVec[i]->setXmlDoc(xmlSrcDoc);
        }
    }

    //write objects to their own xmlDoc
    //for start use OpenScenarioBase xml document
    //
    writeToDOM(osbSourceFile->getXmlDoc()->getDocumentElement(), osbSourceFile->getXmlDoc());

    //write xmlDocs to disk
    //
    for (int i = 0; i < srcFileVec.size(); i++)
    {
        std::string srcFileName = srcFileVec[i]->getSrcFileName();

        if (srcFileName != fileName)
        {
            srcFileName = "out_" + srcFileName;
        }
        xercesc::DOMDocument* xmlSrcDoc = srcFileVec[i]->getXmlDoc();

#if (XERCES_VERSION_MAJOR < 3)
        xercesc::DOMWriter *writer = impl->createDOMWriter();
#else
        xercesc::DOMLSSerializer *writer = ((xercesc::DOMImplementationLS *)impl)->createLSSerializer();
        // set the format-pretty-print feature
        if (writer->getDomConfig()->canSetParameter(xercesc::XMLUni::fgDOMWRTFormatPrettyPrint, true))
            writer->getDomConfig()->setParameter(xercesc::XMLUni::fgDOMWRTFormatPrettyPrint, true);
#endif

        xercesc::XMLFormatTarget *xmlTarget = new xercesc::LocalFileFormatTarget(srcFileName.c_str());

#if (XERCES_VERSION_MAJOR < 3)
        if (!writer->writeNode(xmlTarget, xmlSrcDoc->getDocumentElement()))
        {
            std::cerr << "OpenScenarioBase::writeXosc: Could not open file for writing!" << std::endl;
            delete xmlTarget;
            delete writer;
            return false;
        }

#else
        xercesc::DOMLSOutput *output = ((xercesc::DOMImplementationLS *)impl)->createLSOutput();
        output->setByteStream(xmlTarget);

        if (!writer->write(xmlSrcDoc, output))
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
    }

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

        //to ensure that the DOM view of a document is the same as if it were saved and re-loaded
        rootElement->normalize();
    }

    return rootElement;
}

bool OpenScenarioBase::parseFromXML(xercesc::DOMElement *currentElement)
{
    source = new oscSourceFile();
    source->initialize(this);
    std::string srcFileName = xercesc::XMLString::transcode(dynamic_cast<xercesc::DOMNode *>(currentElement)->getBaseURI());
    source->setVariables(xercesc::XMLString::transcode(currentElement->getNodeName()), srcFileName);

    this->addToSrcFileVec(source);

    return oscObjectBase::parseFromXML(currentElement, source);
}

void OpenScenarioBase::addToSrcFileVec(oscSourceFile *src)
{
    srcFileVec.push_back(src);
}

std::vector<oscSourceFile *> OpenScenarioBase::getSrcFileVec()
{
    return srcFileVec;
}
