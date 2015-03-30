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
<<<<<<< HEAD:src/OpenCOVER/cover/mui/support/ConfigParser.cpp
using namespace xercesc_3_1;
using namespace mui;
=======
using namespace xercesc;

>>>>>>> ceec2b1cf1d5366e3aec53c986cd6c847a6476c9:src/OpenCOVER/cover/mui/support/coMUIConfigParser.cpp

// constructor:
ConfigParser::ConfigParser(const std::string xmlAdresse)
{
    XMLPlatformUtils::Initialize();

    defaultValues.reset(new DefaultValues);

    parser.reset(new XercesDOMParser());                         // create new xercesc-parser
    parser->setValidationScheme(xercesc::XercesDOMParser::Val_Never);

    initializeParser(xmlAdresse);
}

// destructor:
ConfigParser::~ConfigParser()
{
}

// returns true, if the element shall be visible; else return false
bool ConfigParser::getIsVisible(const std::string UI, const std::string Klasse, const std::string Instanz)
{
    if (existElement(UI, defaultValues->getKeywordClass(), Klasse, nodeList))                   // element with name UI and class Klasse exists
    {
        UIElementNode = getElementNode(UI, defaultValues->getKeywordClass() ,Klasse, nodeList); // UIElementNode is the elementnode with name UI and class Klasse
        if (existElement(Instanz, defaultValues->getKeywordVisible(), "false", UIElementNode->getChildNodes()))
        {
            return false;
        }
    }
    return true;
}

//! returns the matching parentname
std::pair<std::string, bool> ConfigParser::getParent(const std::string UI, const std::string Klasse, const std::string Instanz)
{
    std::pair<std::string, bool> returnPair;
    if (existElement(UI, defaultValues->getKeywordClass(), Klasse, nodeList))                  // element exists
    {
        UIElementNode=getElementNode(UI, defaultValues->getKeywordClass(), Klasse, nodeList);
        if (existElement(Instanz, defaultValues->getKeywordParent(), UIElementNode->getChildNodes()))
        {
            if (getAttributeValue(Instanz, defaultValues->getKeywordParent(), UIElementNode->getChildNodes()).second)
            {
                return getAttributeValue(Instanz, defaultValues->getKeywordParent(), UIElementNode->getChildNodes());
            }
        }
    }
    returnPair.first="ConfigParser::getParent(): Parent according to ";
    returnPair.first.append(UI);
    returnPair.first.append(" ");
    returnPair.first.append(Klasse);
    returnPair.first.append(" ");
    returnPair.first.append(Instanz);
    returnPair.first.append(" doesn't exist.");
    returnPair.second=false;
    return returnPair;
}

//! returns true if attribute exists in configuration file within UI, Device and Identifier
bool ConfigParser::existAttributeInConfigFile(std::string Attribute, std::string UI, std::string Device, std::string Identifier)
{
    if (existElement(UI, defaultValues->getKeywordClass(), Device, nodeList))
    {
        if (existElement(Identifier, Attribute, getElementNode(UI, defaultValues->getKeywordClass(), Device, nodeList)->getChildNodes()))
        {
            return getAttributeValue(Identifier, Attribute, getElementNode(UI, defaultValues->getKeywordClass(), Device, nodeList)->getChildNodes()).second;
        }
    }
    return false;
}

//! returns the matching label
std::pair<std::string, bool> ConfigParser::getLabel(const std::string UI, const std::string Klasse, const std::string Instanz)
{
    std::pair<std::string, bool> returnPair;
    if (existElement(UI, defaultValues->getKeywordClass(), Klasse, nodeList))
    {
        UIElementNode = getElementNode(UI, defaultValues->getKeywordClass(), Klasse, nodeList);
        if (existElement(Instanz, defaultValues->getKeywordLabel(), UIElementNode->getChildNodes()))
        {
            if (getAttributeValue(Instanz, defaultValues->getKeywordLabel(), UIElementNode->getChildNodes()).second)
            {
                return getAttributeValue(Instanz, defaultValues->getKeywordLabel(), UIElementNode->getChildNodes());
            }
        }
    }
    returnPair.first="ConfigParser::getLabel(): Label according to ";
    returnPair.first.append(UI);
    returnPair.first.append(" ");
    returnPair.first.append(Klasse);
    returnPair.first.append(" ");
    returnPair.first.append(Instanz);
    returnPair.first.append(" doesn't exist.");
    returnPair.second=false;
    return returnPair;
}

// returns the matching position
std::pair<std::pair<int,int>, bool> ConfigParser::getPosition(const std::string UI, const std::string Klasse, const std::string Instanz)
{
    std::pair<std::pair<int,int>, bool> returnParsedPosition;
    if (existElement(UI, defaultValues->getKeywordClass(), Klasse, nodeList))
    {
        UIElementNode=getElementNode(UI, defaultValues->getKeywordClass(), Klasse, nodeList);
        bool existXpos = existElement(Instanz, defaultValues->getKeywordXPosition(), UIElementNode->getChildNodes());
        bool existYpos = existElement(Instanz, defaultValues->getKeywordYPosition(), UIElementNode->getChildNodes());
        if (existXpos && existYpos)
        {
            std::string pos_str = getAttributeValue(Instanz, defaultValues->getKeywordXPosition(), UIElementNode->getChildNodes()).first;
            sscanf(pos_str.c_str(), "%d %d", &returnParsedPosition.first.first, &returnParsedPosition.first.second);
            returnParsedPosition.second = true;
            return returnParsedPosition;
        }
        else if (existXpos || existYpos)
        {
            std::cerr << "ERROR: ConfigParser::getPosition(): Only x- or y- position of " << Instanz << " declared in configuration file! Needed both positions or none." << std::endl;
        }
    }
    returnParsedPosition.first.first = -1;
    returnParsedPosition.first.second = -1;
    returnParsedPosition.second = false;
    return returnParsedPosition;
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

// returns the UI-Type of the element
const std::string ConfigParser::getType(DOMElement* Element)
{
    return std::string(XMLString::transcode(Element->getTagName()));
}

// returns true, if node is elementnode
bool ConfigParser::isNodeElement(DOMNode* Node)
{
    if (Node->getNodeType()==DOMNode::ELEMENT_NODE)
    {
        return true;
    }
    else
    {
        return false;
    }
}

// returns the matching elementnode
DOMNode *ConfigParser::getElementNode(const std::string TagName, const std::string Attribute, const std::string AttributeValue, DOMNodeList* NodeListe)
{
    for (size_t i=0; i<(NodeListe->getLength()); ++i)                                       // loop through all elements in NodeList
    {
        if (NodeListe->item(i)->getNodeType()==DOMNode::ELEMENT_NODE)
        {
            DOMElement* nodeElement=static_cast<DOMElement*>(NodeListe->item(i));
            if (XMLString::transcode(nodeElement->getTagName()) == TagName)                 // match in TagName
            {
                AttrVal= XMLString::transcode(nodeElement->getAttribute(XMLString::transcode(Attribute.c_str())));
                if (!AttrVal.empty())                                                       // entry in Attribute
                {
                    if (AttrVal == AttributeValue)                                          // match
                    {
                        return NodeListe->item(i);
                    }
                }
            }
        }
    }
    return NULL;
}

// returns true, if the element exists; else return false
bool ConfigParser::existElement(const std::string TagName, const std::string Attribute, DOMNodeList* NodeListe)
{
    for (size_t i=0; i<(NodeListe->getLength()); ++i)                                           // loop through all elements in NodeList
    {
        if (NodeListe->item(i)->getNodeType()==DOMNode::ELEMENT_NODE)
        {
            DOMElement* nodeElement=static_cast<DOMElement*>(NodeListe->item(i));               // transform node to element
            if (XMLString::transcode(nodeElement->getTagName()) == TagName)                     // match in TagName
            {
                AttrVal = XMLString::transcode(nodeElement->getAttribute(XMLString::transcode(Attribute.c_str())));
                if (!AttrVal.empty())                                                           // entry in Attribute
                {
                    return true;
                }
            }
        }
    }
    return false;
}

// returns true, if the element exists; else return false
bool ConfigParser::existElement(const std::string TagName, const std::string Attribute, const std::string AttributeValue, DOMNodeList* NodeListe)
{
    for (size_t i=0; i<(NodeListe->getLength()); ++i)                                           // loop through all elements in NodeList
    {
        if (NodeListe->item(i)->getNodeType()==DOMNode::ELEMENT_NODE)
        {
            DOMElement* nodeElement=static_cast<DOMElement*>(NodeListe->item(i));               // transform node to element
            if (XMLString::transcode(nodeElement->getTagName()) == TagName)                     // match in TagName
            {
                AttrVal = XMLString::transcode(nodeElement->getAttribute(XMLString::transcode(Attribute.c_str())));
                if (AttrVal == AttributeValue)                                                  // entry in Attribute
                {
                    return true;
                }
            }
        }
    }
    return false;
}

// retrurns the attributevalue as std::string
std::pair<std::string, bool> ConfigParser::getAttributeValue(const std::string TagName, const std::string Attribute, DOMNodeList* NodeListe)
{
    std::pair<std::string, bool> returnPair;
    for (size_t i=0; i<(NodeListe->getLength()); ++i)                                           // loop through all elements in NodeList
    {
        if (NodeListe->item(i)->getNodeType()==DOMNode::ELEMENT_NODE)
        {
            DOMElement *nodeElement = static_cast<DOMElement*>(NodeListe->item(i));             // transform node to element
            if (XMLString::transcode(nodeElement->getTagName()) == TagName)                     // match in TagName
            {
                AttrVal = XMLString::transcode(nodeElement->getAttribute(XMLString::transcode(Attribute.c_str())));
                if (!AttrVal.empty())
                {                                                                               // entry in Attribute
                    returnPair.first=AttrVal;
                    returnPair.second=true;
                    return returnPair;
                }
            }
        }
    }
    returnPair.first = "Attribute doesen't exist";
    returnPair.second = false;
    return returnPair;
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

// lkets the parser read a new file
void ConfigParser::readNewFile(std::string Filename)
{
    if (fileExist(Filename))
    {                                                               // check if configuration file exists
        try{
            parser->parse(XMLString::transcode(Filename.c_str()));             // parse configuration file
        }
        catch(...)
        {
            cerr << "ConfigParser.cpp: error parsing XML-Datei" << endl;
            cout << "ConfigParser.cpp: Parser konnte XML-Datei nicht parsen" << endl;
        }

        parsedDoc=parser->getDocument();                                             // save configuration file as DOMDocument

        rootElement=parsedDoc->getDocumentElement();

        nodeList=rootElement->getChildNodes();
    }
    else
    {                                                                                  // File doesn't exist
        cerr << endl << "****************************************************************************************" << endl;
        cerr << "xml-Config-Datei konnte nicht gefunden werden. Bitte Namen und Pfad der Datei überprüfen" << endl;
        cerr << "****************************************************************************************" <<endl << endl;
    }
}
