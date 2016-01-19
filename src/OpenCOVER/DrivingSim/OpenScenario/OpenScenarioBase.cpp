/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl-2.1.txt.

* License: LGPL 2+ */

#include <OpenScenarioBase.h>
#include <oscVariables.h>
#include "oscSourceFile.h"

#include <iostream>

#include <xercesc/parsers/XercesDOMParser.hpp>
#include <xercesc/dom/DOMImplementation.hpp>
#include <xercesc/dom/DOMLSSerializer.hpp>
#include <xercesc/dom/DOMLSOutput.hpp>

#include <xercesc/framework/LocalFileFormatTarget.hpp>
#include <xercesc/util/XercesVersion.hpp>


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
    OSC_OBJECT_ADD_MEMBER(environment,"oscEnvironmentRef");
    OSC_OBJECT_ADD_MEMBER(entities,"oscEntities");
    OSC_OBJECT_ADD_MEMBER(storyboard,"oscStoryboard");
    OSC_OBJECT_ADD_MEMBER(scenarioEnd,"oscScenarioEnd");
	OSC_OBJECT_ADD_MEMBER(test,"oscTest");

    base = this;

    ownMem = new oscMember();
    ownMem->setName("OpenScenario");
    ownMem->setTypeName("OpenScenarioBase");
    ownMem->setValue(this);
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


bool OpenScenarioBase::loadFile(const std::string &fileName)
{
    if(getRootElement(fileName)==NULL)
        return false;
    else
    {
        return parseFromXML(rootElement);
    }
}

bool OpenScenarioBase::saveFile(const std::string &fileName, bool overwrite/* default false */)
{
    xercesc::DOMImplementation *impl = xercesc::DOMImplementation::getImplementation();
    //oscSourceFile for OpenScenarioBase
    oscSourceFile *osbSourceFile;

    //create xmlDocs for every source file
    //
    for (int i = 0; i < srcFileVec.size(); i++)
    {
        std::string srcFileRootElement = srcFileVec[i]->getRootElementNameAsStr();

        //set filename for main xosc file with root element "OpenScenario" to fileName
        if (srcFileRootElement == "OpenScenario")
        {
            srcFileVec[i]->setSrcFileName(fileName);
            osbSourceFile = srcFileVec[i];
        }

        if (!srcFileVec[i]->getXmlDoc())
        {
            xercesc::DOMDocument *xmlSrcDoc = impl->createDocument(0, srcFileVec[i]->getRootElementNameAsXmlCh(), 0);
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
        //get the relative path from main to the to write
        std::string relFilePath = srcFileVec[i]->getRelPathFromMainDoc();
        //get the file name to write
        std::string srcFileName = srcFileVec[i]->getSrcFileName();

        //for testing: generate a new filename
        if (srcFileName != fileName)
        {
            srcFileName = srcFileName + "_out.xml";
        }

        //file name and path for writing
        std::string pathFileNameToWrite = relFilePath + srcFileName;

        //xml document to write
        xercesc::DOMDocument* xmlSrcDoc = srcFileVec[i]->getXmlDoc();

#if (XERCES_VERSION_MAJOR < 3)
        xercesc::DOMWriter *writer = impl->createDOMWriter();
#else
        xercesc::DOMLSSerializer *writer = ((xercesc::DOMImplementationLS *)impl)->createLSSerializer();
        // set the format-pretty-print feature
        if (writer->getDomConfig()->canSetParameter(xercesc::XMLUni::fgDOMWRTFormatPrettyPrint, true))
            writer->getDomConfig()->setParameter(xercesc::XMLUni::fgDOMWRTFormatPrettyPrint, true);
#endif

        xercesc::XMLFormatTarget *xmlTarget = new xercesc::LocalFileFormatTarget(pathFileNameToWrite.c_str());

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


xercesc::DOMElement *OpenScenarioBase::getRootElement(const std::string &filename)
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
    //namespaces needed for XInclude
    parser->setDoNamespaces(true);
    //parser will process XInclude nodes
    parser->setDoXInclude(true);

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


bool OpenScenarioBase::parseFromXML(xercesc::DOMElement *rootElement)
{
    source = new oscSourceFile();
    source->setSrcFileHref("");
    std::string fileNamePathStr = xercesc::XMLString::transcode(dynamic_cast<xercesc::DOMNode *>(rootElement)->getBaseURI());
    fileNamePath *fnPath = source->getFileNamePath(fileNamePathStr);
    source->setSrcFileName(fnPath->fileName);
    source->setMainDocPath(fnPath->path);
    source->setRelPathFromMainDoc("");
    source->setRootElementName(rootElement->getNodeName());
    addToSrcFileVec(source);

    return oscObjectBase::parseFromXML(rootElement, source);
}


xercesc::DOMDocument *OpenScenarioBase::getDocument() const
{
    return xmlDoc;
}


void OpenScenarioBase::addToSrcFileVec(oscSourceFile *src)
{
    srcFileVec.push_back(src);
}

std::vector<oscSourceFile *> OpenScenarioBase::getSrcFileVec() const
{
    return srcFileVec;
}
