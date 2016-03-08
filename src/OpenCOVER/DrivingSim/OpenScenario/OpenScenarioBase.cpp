/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl-2.1.txt.

* License: LGPL 2+ */

#include "OpenScenarioBase.h"
#include "oscVariables.h"
#include "oscSourceFile.h"
#include "oscUtilities.h"

#include <iostream>

#include <xercesc/parsers/XercesDOMParser.hpp>
#include <xercesc/validators/common/Grammar.hpp>
#include <xercesc/dom/DOMImplementation.hpp>
#include <xercesc/dom/DOMLSSerializer.hpp>
#include <xercesc/dom/DOMLSOutput.hpp>
#include <xercesc/framework/LocalFileFormatTarget.hpp>
#include <xercesc/framework/MemBufFormatTarget.hpp>
#include <xercesc/framework/MemBufInputSource.hpp>
#include <xercesc/util/XercesVersion.hpp>


using namespace OpenScenario;


/*****
 * constructor
 *****/

OpenScenarioBase::OpenScenarioBase():oscObjectBase()
{
    parser = NULL;
    xmlDoc = NULL;
    oscFactories::instance();

    OSC_OBJECT_ADD_MEMBER(fileHeader,"oscFileHeader");
    OSC_OBJECT_ADD_MEMBER(catalogs,"oscCatalogs");
    OSC_OBJECT_ADD_MEMBER(roadNetwork,"oscRoadNetwork");
    OSC_OBJECT_ADD_MEMBER(environment,"oscEnvironmentReference");
    OSC_OBJECT_ADD_MEMBER(entities,"oscEntities");
    OSC_OBJECT_ADD_MEMBER(storyboard,"oscStoryboard");
    OSC_OBJECT_ADD_MEMBER(scenarioEnd,"oscScenarioEnd");
	OSC_OBJECT_ADD_MEMBER(test,"oscTest");

    base = this;

    ownMem = new oscMember();
    ownMem->setName("OpenSCENARIO");
    ownMem->setTypeName("OpenScenarioBase");
    ownMem->setValue(this);

    try
    {
        xercesc::XMLPlatformUtils::Initialize();
    }
    catch (const xercesc::XMLException &toCatch)
    {
        char *message = xercesc::XMLString::transcode(toCatch.getMessage());
        std::cerr << "Error during initialization! :\n" << message << std::endl;
        xercesc::XMLString::release(&message);
    }
}



/*****
 * initialization static variables
 *****/

bool OpenScenarioBase::m_validate = true;

//
unordered_map<std::string /*XmlFileType*/, std::string /*XsdFileName*/> initFuncFileTypeToXsd()
{
    //set the XSD Schema file name for possible file types
    unordered_map<std::string, std::string> fileTypeToXsd;
    fileTypeToXsd.emplace("OpenSCENARIO", "xml-schema/OpenScenario_XML-Schema_OpenSCENARIO.xsd");
    fileTypeToXsd.emplace("driver", "xml-schema/OpenScenario_XML-Schema_driver.xsd");
    fileTypeToXsd.emplace("entity", "xml-schema/OpenScenario_XML-Schema_entity.xsd");
    fileTypeToXsd.emplace("environment", "xml-schema/OpenScenario_XML-Schema_environment.xsd");
    fileTypeToXsd.emplace("maneuver", "xml-schema/OpenScenario_XML-Schema_maneuver.xsd");
    fileTypeToXsd.emplace("miscObject", "xml-schema/OpenScenario_XML-Schema_miscObject.xsd");
    fileTypeToXsd.emplace("observer", "xml-schema/OpenScenario_XML-Schema_observer.xsd");
    fileTypeToXsd.emplace("pedestrian", "xml-schema/OpenScenario_XML-Schema_pedestrian.xsd");
    fileTypeToXsd.emplace("routing", "xml-schema/OpenScenario_XML-Schema_routing.xsd");
    fileTypeToXsd.emplace("vehicle", "xml-schema/OpenScenario_XML-Schema_vehicle.xsd");

    return fileTypeToXsd;
}

const unordered_map<std::string, std::string> OpenScenarioBase::m_fileTypeToXsdFileName = initFuncFileTypeToXsd();


/*****
 * destructor
 *****/

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



/*****
 * public functions
 *****/

//
void OpenScenarioBase::setValidation(const bool validate)
{
    m_validate = validate;
}


//
bool OpenScenarioBase::loadFile(const std::string &fileName, const std::string &fileType)
{
    xercesc::DOMElement *rootElement = getRootElement(fileName, fileType);
    if(rootElement == NULL)
    {
        return false;
    }
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

        //set filename for main xosc file with root element "OpenSCENARIO" to fileName
        if (srcFileRootElement == "OpenSCENARIO")
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
    //start with the document with root element OpenSCENARIO
    //
    writeToDOM(osbSourceFile->getXmlDoc()->getDocumentElement(), osbSourceFile->getXmlDoc());

    //write xmlDocs to disk
    //
    for (int i = 0; i < srcFileVec.size(); i++)
    {
        //get the relative path from main doc to the file to write
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
        xercesc::DOMDocument *xmlSrcDoc = srcFileVec[i]->getXmlDoc();

        //write xml document to file
        writeFileToDisk(xmlSrcDoc, pathFileNameToWrite.c_str());
    }

    return true;
}


//
bool OpenScenarioBase::writeFileToDisk(xercesc::DOMDocument *xmlDocToWrite, const char *filenameToWrite)
{
    xercesc::DOMImplementation *impl = xercesc::DOMImplementation::getImplementation();

#if (XERCES_VERSION_MAJOR < 3)
    xercesc::DOMWriter *writer = impl->createDOMWriter();
#else
    xercesc::DOMLSSerializer *writer = ((xercesc::DOMImplementationLS *)impl)->createLSSerializer();
    // set the format-pretty-print feature
    if (writer->getDomConfig()->canSetParameter(xercesc::XMLUni::fgDOMWRTFormatPrettyPrint, true))
    {
        writer->getDomConfig()->setParameter(xercesc::XMLUni::fgDOMWRTFormatPrettyPrint, true);
    }
#endif

    xercesc::XMLFormatTarget *xmlTarget = new xercesc::LocalFileFormatTarget(filenameToWrite);

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

    if (!writer->write(xmlDocToWrite, output))
    {
        std::cerr << "OpenScenarioBase::writeXosc: Could not open file for writing!" << std::endl;
        delete output;
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

xercesc::MemBufFormatTarget *OpenScenarioBase::writeFileToMemory(xercesc::DOMDocument *xmlDocToWrite)
{
    xercesc::DOMImplementation *impl = xercesc::DOMImplementation::getImplementation();

#if (XERCES_VERSION_MAJOR < 3)
    xercesc::DOMWriter *writer = impl->createDOMWriter();
#else
    xercesc::DOMLSSerializer *writer = ((xercesc::DOMImplementationLS *)impl)->createLSSerializer();
    // set the format-pretty-print feature
    if (writer->getDomConfig()->canSetParameter(xercesc::XMLUni::fgDOMWRTFormatPrettyPrint, true))
    {
        writer->getDomConfig()->setParameter(xercesc::XMLUni::fgDOMWRTFormatPrettyPrint, true);
    }
#endif

    xercesc::MemBufFormatTarget *xmlMemTarget = new xercesc::MemBufFormatTarget();

#if (XERCES_VERSION_MAJOR < 3)
    if (!writer->writeNode(xmlMemTarget)
    {
        std::cerr << "OpenScenarioBase::writeXosc: Could not open file for writing!" << std::endl;
        delete xmlMemTarget;
        delete writer;
        return NULL;
    }
#else
    xercesc::DOMLSOutput *output = ((xercesc::DOMImplementationLS *)impl)->createLSOutput();
    output->setByteStream(xmlMemTarget);

    if (!writer->write(xmlDocToWrite, output))
    {
        std::cerr << "OpenScenarioBase::writeXosc: Could not open file for writing!" << std::endl;
        delete output;
        delete xmlMemTarget;
        delete writer;
        return NULL;
    }

    delete output;
#endif

    delete writer;

    return xmlMemTarget;
}

xercesc::DOMElement *OpenScenarioBase::getRootElement(const std::string &fileName, const std::string &fileType, const bool validate)
{
    //
    //parsing a file is done in two steps
    // -first step with enabled XInclude and disabled validation
    // -second step with disabled XInclude and enabled validation
    //
    //(if file parsed in one step with enabled XInclude and validation
    // then the validation is done before XInclude. But we want validate the
    // whole document with all elements included in one document
    //

    //new parser, error handler, generic settings
    //
    parser = new xercesc::XercesDOMParser();

    //error handler
    ParserErrorHandler parserErrorHandler;
    parser->setErrorHandler(&parserErrorHandler);

    //namespaces needed for XInclude and validation
    parser->setDoNamespaces(true);


    //parse file with enabled XInclude and disabled validation
    //
    //parser will process XInclude nodes
    parser->setDoXInclude(true);
    //parse without validation
    parser->setDoSchema(false);

    //parse the file
    try
    {
        parser->parse(fileName.c_str());
    }
    catch (...)
    {
        std::cerr << "\nErrors during parse of the document '" << fileName << "'\n" << std::endl;

        return NULL;
    }


    //validate the xosc file in a second run without XInclude
    // (enabled XInclude generate a namespace attribute xmlns:xml for xml:base
    //  and this had to be added to the xml schema)
    //
    xercesc::DOMElement *tmpRootElem = NULL;
    std::string tmpRootElemName;
    xercesc::DOMDocument *tmpXmlDoc = parser->getDocument();
    if (tmpXmlDoc)
    {
        tmpRootElem = tmpXmlDoc->getDocumentElement();
        tmpRootElemName = xercesc::XMLString::transcode(tmpRootElem->getNodeName());
    }

    //only for tmpRootElementName == fileType validation is reasonable
    if (validate && tmpRootElemName == fileType)
    {
        //name of XML Schema
        const char *xsdFileName;
        unordered_map<std::string, std::string>::const_iterator found = m_fileTypeToXsdFileName.find(fileType);
        if (found != m_fileTypeToXsdFileName.end())
        {
            xsdFileName = found->second.c_str();
        }
        else
        {
            std::cerr << "Error! Can't determine a XSD Schema file for fileType " << fileType << ".\n" << std::endl;

            return NULL;
        }

        //set location of schema for elements without namespace
        // (in schema file no global namespace is used)
        parser->setExternalNoNamespaceSchemaLocation(xsdFileName);

        //read the schema grammar file (.xsd) and cache it
        try
        {
            parser->loadGrammar(xsdFileName, xercesc::Grammar::SchemaGrammarType, true);
        }
        catch (...)
        {
            std::cerr << "\nError: Couldn't load XML Schema '" << xsdFileName << "'\n" << std::endl;

            return NULL;
        }


        //settings for validation
        parser->setDoXInclude(false); //disable XInclude for validation
        parser->setValidationScheme(xercesc::XercesDOMParser::Val_Auto);
        parser->setDoSchema(true);
        parser->setValidationConstraintFatal(true);
        parser->setExitOnFirstFatalError(true);
        parser->cacheGrammarFromParse(true);

        //write xml document got from parser to memory buffer
        xercesc::MemBufFormatTarget *tmpXmlMemBufFormat = writeFileToMemory(tmpXmlDoc);

        //raw buffer, length and fake id for memory input source
        const XMLByte *tmpRawBuffer = tmpXmlMemBufFormat->getRawBuffer();
        XMLSize_t tmpRawBufferLength = tmpXmlMemBufFormat->getLen();
        const XMLCh *bufId = xercesc::XMLString::transcode("memBuf");

        //new input source from memory for parser
        xercesc::InputSource *tmpInputSrc = new xercesc::MemBufInputSource(tmpRawBuffer, tmpRawBufferLength, bufId);

        //parse memory buffer
        try
        {
            parser->parse(*tmpInputSrc);
        }
        catch (...)
        {
            std::cerr << "\nErrors during validation of the document " << fileName << "\n" << std::endl;

            delete tmpXmlMemBufFormat;
            delete tmpInputSrc;
            return NULL;
        }

        //delete used objects
        delete tmpXmlMemBufFormat;
        delete tmpInputSrc;
    }


    //set xmlDoc and rootElement
    if (tmpXmlDoc)
    {
        xmlDoc = tmpXmlDoc;
        xercesc::DOMElement *rootElement = tmpRootElem;

        //to ensure that the DOM view of a document is the same as if it were saved and re-loaded
        rootElement->normalize();

        return rootElement;
    }

    return NULL;
}


//
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


//
xercesc::DOMDocument *OpenScenarioBase::getDocument() const
{
    return xmlDoc;
}


//
void OpenScenarioBase::addToSrcFileVec(oscSourceFile *src)
{
    srcFileVec.push_back(src);
}

std::vector<oscSourceFile *> OpenScenarioBase::getSrcFileVec() const
{
    return srcFileVec;
}
