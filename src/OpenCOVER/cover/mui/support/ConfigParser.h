#ifndef MUICONFIGPARSER_H
#define MUICONFIGPARSER_H

#include <boost/smart_ptr.hpp>

#include <xercesc/parsers/XercesDOMParser.hpp>
#include <xercesc/dom/DOMDocument.hpp>
#include <xercesc/dom/DOM.hpp>

namespace mui
{
class DefaultValues;

// begin of class
class ConfigParser
{
public:
    // constructor
    ConfigParser(const std::string xmlAdresse);
    // destructor
    ~ConfigParser();

    // memberfunction:
    const std::string getType(xercesc::DOMElement* Element);         // return the type of the element
    bool isNodeElement(xercesc::DOMNode* Node);
    bool fileExist(std::string File);
    //std::string getValueClassInstanzAttribute(const std::string UI, const std::string Klasse, const std::string Instanz, const std::string Attribute);

    bool getIsVisible(const std::string UI, const std::string Klasse, const std::string Instanz);
    std::pair<std::string, bool> getParent(const std::string UI, const std::string Klasse, const std::string Instanz);
    std::pair<std::pair<int,int>, bool> getPosition(const std::string UI, const std::string Klasse, const std::string Instanz);
    std::pair<std::string, bool> getLabel(const std::string UI, const std::string Klasse, const std::string Instanz);
    void readNewFile(std::string Filename);
    bool existAttributeInConfigFile(std::string Attribute, std::string UI, std::string Device, std::string Identifier);


private:
    // membervariables:
    boost::shared_ptr<xercesc::XercesDOMParser> parser;
    xercesc::DOMDocument* parsedDoc;
    xercesc::DOMElement* rootElement;
    xercesc::DOMNodeList* nodeList;
    xercesc::DOMNode* UIElementNode;

    boost::shared_ptr<DefaultValues> defaultValues;

    std::string AttrVal;                                        // will be overwritten with attriburevalues continously


    // memberfunctions:
    void initializeParser(std::string adress);

    xercesc::DOMNode* getElementNode(const std::string TagName, const std::string Attribute, const std::string AttributeValue, xercesc::DOMNodeList* NodeListe);
    bool existElement(const std::string TagName, const std::string Attribute, xercesc::DOMNodeList* NodeListe);
    bool existElement(const std::string TagName, const std::string Attribute, const std::string AttributeValue, xercesc::DOMNodeList* NodeListe);
    std::pair<std::string, bool> getAttributeValue(const std::string TagName, const std::string Attribute, xercesc::DOMNodeList* NodeListe);

};
} // end namespace
#endif
