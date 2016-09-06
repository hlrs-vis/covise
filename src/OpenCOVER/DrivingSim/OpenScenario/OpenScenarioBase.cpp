/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl-2.1.txt.

* License: LGPL 2+ */

#include "OpenScenarioBase.h"
#include "oscVariables.h"

#include <iostream>

#include <xercesc/dom/DOMDocument.hpp>
#include <xercesc/dom/DOMElement.hpp>
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

OpenScenarioBase::OpenScenarioBase() :
        oscObjectBase(),
        xmlDoc(NULL),
        m_validate(true),
        m_fullReadCatalogs(false)
{
    oscFactories::instance();

    OSC_OBJECT_ADD_MEMBER(fileHeader, "oscFileHeader");
    OSC_OBJECT_ADD_MEMBER(catalogs, "oscCatalogs");
    OSC_OBJECT_ADD_MEMBER(roadNetwork, "oscRoadNetwork");
    OSC_OBJECT_ADD_MEMBER(environment, "oscEnvironmentReference");
    OSC_OBJECT_ADD_MEMBER(entities, "oscEntities");
    OSC_OBJECT_ADD_MEMBER(storyboard, "oscStoryboard");
    OSC_OBJECT_ADD_MEMBER(scenarioEnd, "oscScenarioEnd");
    OSC_OBJECT_ADD_MEMBER_OPTIONAL(test, "oscTest");

    base = this;

    ownMember = new oscMember();
    ownMember->setName("OpenSCENARIO");
    ownMember->setTypeName("OpenScenarioBase");
    ownMember->setValue(this);

    //in order to work with the Xerces-C++ parser, the XML subsystem must be initialized first
    //every call of XMLPlatformUtils::Initialize() must have a matching call of XMLPlatformUtils::Terminate() (see destructor)
    try
    {
        xercesc::XMLPlatformUtils::Initialize();
    }
    catch (const xercesc::XMLException &toCatch)
    {
        char *message = xercesc::XMLString::transcode(toCatch.getMessage());
        std::cerr << "Error during xerces initialization! :\n" << message << std::endl;
        xercesc::XMLString::release(&message);
    }

    //parser and error handler have to be initialized _after_ xercesc::XMLPlatformUtils::Initialize()
    //can't be done in member initializer list
    parser = new xercesc::XercesDOMParser();
    parserErrorHandler = new ParserErrorHandler();

    //generic settings for parser
    //
    parser->setErrorHandler(parserErrorHandler);
    parser->setExitOnFirstFatalError(true);
    //namespaces needed for XInclude and validation
    parser->setDoNamespaces(true);
    //settings for validation
    parser->setValidationScheme(xercesc::XercesDOMParser::Val_Auto);
    parser->setValidationConstraintFatal(true);
    parser->cacheGrammarFromParse(true);
}



/*****
 * initialization static variables
 *****/

OpenScenarioBase::FileTypeXsdFileNameMap initFuncFileTypeToXsd()
{
    //set the XSD Schema file name for possible file types
    OpenScenarioBase::FileTypeXsdFileNameMap fileTypeToXsd;
    fileTypeToXsd.emplace("OpenSCENARIO", bf::path("OpenScenario_XML-Schema_OpenSCENARIO.xsd"));
    fileTypeToXsd.emplace("driver", bf::path("OpenScenario_XML-Schema_driver.xsd"));
    fileTypeToXsd.emplace("entity", bf::path("OpenScenario_XML-Schema_entity.xsd"));
    fileTypeToXsd.emplace("environment", bf::path("OpenScenario_XML-Schema_environment.xsd"));
    fileTypeToXsd.emplace("maneuver", bf::path("OpenScenario_XML-Schema_maneuver.xsd"));
    fileTypeToXsd.emplace("miscObject", bf::path("OpenScenario_XML-Schema_miscObject.xsd"));
    fileTypeToXsd.emplace("observer", bf::path("OpenScenario_XML-Schema_observer.xsd"));
    fileTypeToXsd.emplace("pedestrian", bf::path("OpenScenario_XML-Schema_pedestrian.xsd"));
    fileTypeToXsd.emplace("routing", bf::path("OpenScenario_XML-Schema_routing.xsd"));
    fileTypeToXsd.emplace("vehicle", bf::path("OpenScenario_XML-Schema_vehicle.xsd"));

    return fileTypeToXsd;
}

const OpenScenarioBase::FileTypeXsdFileNameMap OpenScenarioBase::s_fileTypeToXsdFileName = initFuncFileTypeToXsd();

OpenScenarioBase::DefaultFileTypeNameMap initDefaultFileTypeMap()
{
    //set the XSD Schema file name for possible file types
    OpenScenarioBase::DefaultFileTypeNameMap defaultFileTypeMap;
	defaultFileTypeMap.emplace("vehicle", bf::path("OpenScenario_XML-Default_vehicle.xosc"));
	defaultFileTypeMap.emplace("light", bf::path("OpenScenario_XML-Default_OSCLight.xosc"));
	defaultFileTypeMap.emplace("driver", bf::path("OpenScenario_XML-Default_driver.xosc"));
	defaultFileTypeMap.emplace("entity", bf::path("OpenScenario_XML-Default_entity.xosc"));
	defaultFileTypeMap.emplace("environment", bf::path("OpenScenario_XML-Default_environment.xosc"));
	defaultFileTypeMap.emplace("roadCondition", bf::path("OpenScenario_XML-Default_OSCRoadCondition.xosc"));
	defaultFileTypeMap.emplace("pedestrian", bf::path("OpenScenario_XML-Default_pedestrian.xosc"));
	defaultFileTypeMap.emplace("maneuver", bf::path("OpenScenario_XML-Default_maneuver.xosc"));
	defaultFileTypeMap.emplace("observer", bf::path("OpenScenario_XML-Default_observer.xosc"));
	defaultFileTypeMap.emplace("routing", bf::path("OpenScenario_XML-Default_routing.xosc"));
	defaultFileTypeMap.emplace("miscObject", bf::path("OpenScenario_XML-Default_miscObject.xosc"));
	defaultFileTypeMap.emplace("filter", bf::path("OpenScenario_XML-Default_OSCFilter.xosc"));
	defaultFileTypeMap.emplace("event", bf::path("OpenScenario_XML-Default_OSCEvent.xosc"));
	defaultFileTypeMap.emplace("startCondition", bf::path("OpenScenario_XML-Default_OSCStartCondition.xosc"));
	defaultFileTypeMap.emplace("waypoint", bf::path("OpenScenario_XML-Default_OSCWaypoint.xosc"));
	defaultFileTypeMap.emplace("parameter", bf::path("OpenScenario_XML-Default_OSCParameterTypeA.xosc"));
	defaultFileTypeMap.emplace("action", bf::path("OpenScenario_XML-Default_OSCAction.xosc"));

    return defaultFileTypeMap;
}

const OpenScenarioBase::FileTypeXsdFileNameMap OpenScenarioBase::s_defaultFileTypeMap = initDefaultFileTypeMap();




/*****
 * destructor
 *****/

OpenScenarioBase::~OpenScenarioBase()
{
    delete parserErrorHandler;
    delete parser;
    //match the call of XMLPlatformUtils::Initialize() from constructor
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


//
void OpenScenarioBase::setValidation(const bool validate)
{
    m_validate = validate;
}

bool OpenScenarioBase::getValidation() const
{
    return m_validate;
}

void OpenScenarioBase::setFullReadCatalogs(const bool fullReadCatalogs)
{
    m_fullReadCatalogs = fullReadCatalogs;
}

bool OpenScenarioBase::getFullReadCatalogs() const
{
    return m_fullReadCatalogs;
}


//
void OpenScenarioBase::setPathFromCurrentDirToDoc(const bf::path &path)
{
    m_pathFromCurrentDirToDoc = path.parent_path();
}

void OpenScenarioBase::setPathFromCurrentDirToDoc(const std::string &path)
{
    m_pathFromCurrentDirToDoc = bf::path(path).parent_path();
}

bf::path OpenScenarioBase::getPathFromCurrentDirToDoc() const
{
    return m_pathFromCurrentDirToDoc;
}


//
bool OpenScenarioBase::loadFile(const std::string &fileName, const std::string &fileType)
{
    setPathFromCurrentDirToDoc(fileName);

    xercesc::DOMElement *rootElement = getRootElement(fileName, fileType, m_validate);
	std::string rootElementName = xercesc::XMLString::transcode(rootElement->getNodeName());

	if (rootElementName != fileType)
	{
		return false;
	}

    if(rootElement == NULL) // create new file
    {
		if (createSource(fileName, fileType))
		{
			return true;
		}
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
        //get the file name to write
        bf::path srcFileName = srcFileVec[i]->getSrcFileName();

        //////
        //for testing: generate a new filename
        //
        if (srcFileName != fileName)
        {
            srcFileName += "_out.xml";
        }
        //
        //////

		srcFileVec[i]->writeFileToDisk();
    }

    return true;
}

void OpenScenarioBase::clearDOM()
{
	for (int i = 0; i < srcFileVec.size(); i++)
	{
		srcFileVec[i]->clearXmlDoc();
	}
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
        std::cerr << "\nErrors during parse of the document '" << fileName << "'.\n" << std::endl;

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

    //validation is only reasonable for tmpRootElementName == fileType
    if (validate && tmpRootElemName == fileType)
    {
        //path to covise directory
        bf::path coDir = getEnvVariable("COVISEDIR");
        //relative path from covise directory to OpenSCENARIO directory
        bf::path oscDirRelPath = bf::path("src/OpenCOVER/DrivingSim/OpenScenario");
        //relative path from OpenSCENARIO directory to directory with schema files
        bf::path xsdDirRelPath = bf::path("xml-schema");

        //name of XML Schema
        bf::path xsdFileName;
        FileTypeXsdFileNameMap::const_iterator found = s_fileTypeToXsdFileName.find(fileType);
        if (found != s_fileTypeToXsdFileName.end())
        {
            xsdFileName = found->second;
        }
        else
        {
            std::cerr << "Error! Can't determine a XSD Schema file for fileType '" << fileType << "'.\n" << std::endl;

            return NULL;
        }

        //path and filename of XML Schema *.xsd file
        bf::path xsdPathFileName = coDir;
        xsdPathFileName /= oscDirRelPath;
        xsdPathFileName /= xsdDirRelPath;
        xsdPathFileName /= xsdFileName;
		xsdPathFileName.make_preferred();

        //set schema location and load grammar if filename changed
        if (xsdPathFileName != m_xsdPathFileName)
        {
            m_xsdPathFileName = xsdPathFileName;

            //set location of schema for elements without namespace
            // (in schema file no global namespace is used)
			parser->setExternalNoNamespaceSchemaLocation(m_xsdPathFileName.c_str());

            //read the schema grammar file (.xsd) and cache it
            try
            {
                parser->loadGrammar(m_xsdPathFileName.c_str(), xercesc::Grammar::SchemaGrammarType, true);
            }
            catch (...)
            {
                std::cerr << "\nError! Can't load XML Schema '" << m_xsdPathFileName << "'.\n" << std::endl;

                return NULL;
            }
        }


        //settings for validation
        parser->setDoXInclude(false); //disable XInclude for validation: to prevent generation of a namespace attribute xmlns:xml for xml:base
        parser->setDoSchema(true);

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
            std::cerr << "\nErrors during validation of the document '" << fileName << "'.\n" << std::endl;

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
    else
    {
        return NULL;
    }
}

xercesc::DOMElement *OpenScenarioBase::getDefaultXML(const std::string &fileType)
{
	//path to covise directory
	bf::path coDir = getEnvVariable("COVISEDIR");
	//relative path from covise directory to OpenSCENARIO directory
	bf::path oscDirRelPath = bf::path("src/OpenCOVER/DrivingSim/OpenScenario");
	//relative path from OpenSCENARIO directory to directory with schema files
	bf::path xsdDirRelPath = bf::path("xml-default");

	//name of XML Schema
	bf::path xsdFileName;
	FileTypeXsdFileNameMap::const_iterator found = s_defaultFileTypeMap.find(fileType);
	if (found != s_defaultFileTypeMap.end())
	{
		xsdFileName = found->second;
	}
	else
	{
		std::cerr << "Error! Can't determine a XSD Default file for fileType '" << fileType << "'.\n" << std::endl;

		return NULL;
	}

	//path and filename of XML Schema *.xsd file
	bf::path xsdPathFileName = coDir;
	xsdPathFileName /= oscDirRelPath;
	xsdPathFileName /= xsdDirRelPath;
	xsdPathFileName /= xsdFileName;
	xsdPathFileName.make_preferred();


	//parse file with enabled XInclude and disabled validation
    //
    //parser will process XInclude nodes
    parser->setDoXInclude(true);
    //parse without validation
    parser->setDoSchema(false);

    //parse the file
    try
    {
        parser->parse(xsdPathFileName.c_str());
    }
    catch (...)
    {
        std::cerr << "\nErrors during parse of the document '" << xsdPathFileName << "'.\n" << std::endl;

        return NULL;
    }

    xercesc::DOMElement *tmpRootElem = NULL;
    std::string tmpRootElemName;
    xercesc::DOMDocument *tmpXmlDoc = parser->getDocument();
    if (tmpXmlDoc)
    {
        return tmpXmlDoc->getDocumentElement();
    }

	return NULL;
}


//
bool OpenScenarioBase::parseFromXML(xercesc::DOMElement *rootElement)
{
	std::string fileNamePathStr = xercesc::XMLString::transcode(dynamic_cast<xercesc::DOMNode *>(rootElement)->getBaseURI());
	createSource(fileNamePathStr, xercesc::XMLString::transcode(rootElement->getNodeName()));

    return oscObjectBase::parseFromXML(rootElement, source);
}

//
oscSourceFile *OpenScenarioBase::createSource(const std::string &fileName, const std::string &fileType)
{
    source = new oscSourceFile();
	source->setNameAndPath(fileName, fileType, m_pathFromCurrentDirToDoc);

    addToSrcFileVec(source);

	return source;
}




