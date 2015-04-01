#include "ConfigParser.h"
#include "DefaultValues.h"

#include <xercesc/parsers/XercesDOMParser.hpp>
#include <xercesc/dom/DOM.hpp>
#include <xercesc/dom/DOMDocument.hpp>
#include <xercesc/util/XMLString.hpp>
#include <xercesc/util/PlatformUtils.hpp>

#include <iostream>
#include <fstream>
#include <string>

using namespace std;
using namespace xercesc;
using namespace mui;

// constructor:
ConfigParser::ConfigParser(const std::string xmlAdresse)
{
    XMLPlatformUtils::Initialize();

//    defaultValues.reset(new DefaultValues);

    parser.reset(new XercesDOMParser());                         // create new xercesc-parser
    parser->setValidationScheme(xercesc::XercesDOMParser::Val_Never);

    initializeParser(xmlAdresse);
}

// destructor:
ConfigParser::~ConfigParser()
{
}

// returns true, if the file exists; else returns false
bool ConfigParser::fileExist (std::string File)
{
    FILE *fp = fopen(File.c_str(),"r"); // switched to fopen because of a visual studio 2012 linker issue
    if(fp!=NULL)
    {
        fclose(fp);
        return true;
    }
    else
    {
        return false;
    }
}

std::pair<bool,bool> ConfigParser::getIsVisible(UITypeEnum UI, DeviceTypesEnum Device, string UniqueIdentifier)
{
    std::pair<bool,bool> returnValue;
    std::pair<std::string, bool> gottenAttribute;
    gottenAttribute = getAttributeValue(UI, Device, UniqueIdentifier, mui::VisibleEnum);
    returnValue.second = gottenAttribute.second;
    if (gottenAttribute.second)                                     // matching entry exists
    {
        if (gottenAttribute.first == "true")
        {
            returnValue.first = true;
        }
        else if (gottenAttribute.first == "false")
        {
            returnValue.first = false;
        }
        else
        {
            std::cerr << "ERROR: ConfigParser::getIsVisible(): Visible-value in " << getKeywordUI(UI) << "; " << getKeywordDevice(Device) << "; " << UniqueIdentifier << " is unequal 'true' or 'false'." << std::endl;
            returnValue.first = NULL;
            returnValue.second = false;
        }
    }
    return returnValue;
}

std::pair<std::string, bool> ConfigParser::getParent(UITypeEnum UI, DeviceTypesEnum Device, std::string UniqueIdentifier)
{
    return getAttributeValue(UI, Device, UniqueIdentifier, mui::ParentEnum);
}

std::pair<std::string, bool> ConfigParser::getLabel(UITypeEnum UI, DeviceTypesEnum Device, string UniqueIdentifier)
{
    return getAttributeValue(UI, Device, UniqueIdentifier, mui::LabelEnum);
}

std::pair<std::pair<int,int>, bool> ConfigParser::getPos(UITypeEnum UI, DeviceTypesEnum Device, string UniqueIdentifier)
{
    std::pair<std::string, bool> posx_str = getAttributeValue(UI, Device, UniqueIdentifier, mui::PosXEnum);
    std::pair<std::string, bool> posy_str = getAttributeValue(UI, Device, UniqueIdentifier, mui::PosYEnum);
    std::pair<std::pair<int,int>, bool> returnValue;
    if (posx_str.second && posy_str.second)
    {
        returnValue.second = true;
        sscanf(posx_str.first.c_str(), "%d", &returnValue.first.first);
        sscanf(posy_str.first.c_str(), "%d", &returnValue.first.second);
    }
    else if (posx_str.second || posy_str.second)
    {
        std::cerr << "ERROR: ConfigParser::getPos(): only one position defined in " << getKeywordUI(UI) << "; " << getKeywordDevice(Device) << "; " << UniqueIdentifier << std::endl;
    }
    else
    {
        returnValue.second = false;
    }
    return returnValue;
}

std::pair<DOMNode*,bool> ConfigParser::getDeviceEntryNode(mui::UITypeEnum UI, mui::DeviceTypesEnum Device)
{
    std::pair<DOMNode*, bool> returnValue;
    returnValue.first = NULL;
    returnValue.second= false;
    for (size_t i=0; i<nodeList->getLength(); ++i)
    {
        if (nodeList->item(i)->getNodeType() == DOMNode::ELEMENT_NODE)
        {
            std::string keywordUI = getKeywordUI(UI);
            DOMElement* nodeElement=static_cast<DOMElement*>(nodeList->item(i));               // transform node to element

            if (XMLString::transcode(nodeElement->getTagName()) == keywordUI.c_str())
            {
                std::string keywordDevice = mui::getKeywordAttribute(mui::DeviceEnum);

                std::string AttrVal = XMLString::transcode(nodeElement->getAttribute(XMLString::transcode(keywordDevice.c_str())));
                if (AttrVal == getKeywordDevice(Device))
                {
                    returnValue.first = nodeElement;
                    returnValue.second = true;
                    return returnValue;
                }
            }
        }
    }
    return returnValue;
}

std::pair<DOMElement*, bool> ConfigParser::getElement(string UniqueIdentifier, DOMNodeList *DeviceNodeChilds)
{
    std::pair<DOMElement*, bool> returnValue;
    returnValue.first = NULL;
    returnValue.second= false;
    for (size_t i=0; i< DeviceNodeChilds->getLength(); ++i)                         // loop through all ChildNodes in DeviceNode
    {
        if (DeviceNodeChilds->item(i)->getNodeType() == DOMNode::ELEMENT_NODE)      // Node is ElementNode
        {
            DOMElement* nodeElement = static_cast<DOMElement*>(DeviceNodeChilds->item(i));
            if (XMLString::transcode(nodeElement->getTagName()) == UniqueIdentifier.c_str())    // match in TagName
            {
                returnValue.second = true;
                returnValue.first = nodeElement;
                return returnValue;
            }
        }
    }
    return returnValue;
}

std::pair<std::string,bool> ConfigParser::getAttributeValue(mui::UITypeEnum UI, mui::DeviceTypesEnum Device, string UniqueIdentifier, mui::AttributesEnum Attribute)
{
    std::pair<std::string,bool> returnValue;
    returnValue.first = "";
    returnValue.second= false;
    std::pair<DOMNode*, bool> DeviceEntryNode=getDeviceEntryNode(UI, Device);
    if (DeviceEntryNode.second)                                                       // device-Entry exists
    {
        std::pair<DOMElement*,bool> Element = getElement(UniqueIdentifier, DeviceEntryNode.first->getChildNodes());
        if (Element.second)
        {
            std::string keywordAttribute = getKeywordAttribute(Attribute);
            std::string AttrVal = XMLString::transcode(Element.first->getAttribute(XMLString::transcode(keywordAttribute.c_str())));
            returnValue.first=AttrVal;
            returnValue.second=true;
            return returnValue;
        }
    }
    return returnValue;
}


// initializes parser with adress of configuration file
void ConfigParser::initializeParser(std::string adress)
{
    if (fileExist(adress))                                                                      // check if configuration file exists
    {
        try{
            parser->parse(XMLString::transcode(adress.c_str()));                                // parse configuration file
        }
        catch(...)
        {
            cerr << "ConfigParser.cpp: error parsing XML-Datei" << endl;
        }

        parsedDoc=parser->getDocument();                                                        // save configuration file as DOMDocument

        rootElement=parsedDoc->getDocumentElement();

        nodeList=rootElement->getChildNodes();
    }
    else
    {                                                                                      // File doesn't exist
        cerr << endl << "****************************************************************************************" << endl;
        cerr << "xml-Config-Datei konnte nicht gefunden werden. Bitte Namen und Pfad der Datei überprüfen" << endl;
        cerr << "****************************************************************************************" <<endl << endl;
    }
}

