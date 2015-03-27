#ifndef MUICONFIGPARSER_H
#define MUICONFIGPARSER_H

#include <iostream>
#include <boost/smart_ptr.hpp>

// forward-declaration
namespace xercesc_3_1
{
class XercesDOMParser;
class DOMDocument;
class DOMNode;
class DOMNodeList;
class DOMElement;
}

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
    const std::string getType(xercesc_3_1::DOMElement* Element);         // return the type of the element
    bool isNodeElement(xercesc_3_1::DOMNode* Node);
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
    boost::shared_ptr<xercesc_3_1::XercesDOMParser> parser;
    xercesc_3_1::DOMDocument* parsedDoc;
    xercesc_3_1::DOMElement* rootElement;
    xercesc_3_1::DOMNodeList* nodeList;
    xercesc_3_1::DOMNode* UIElementNode;

    boost::shared_ptr<DefaultValues> defaultValues;

    std::string AttrVal;                                        // will be overwritten with attriburevalues continously


    // memberfunctions:
    void initializeParser(std::string adress);

    xercesc_3_1::DOMNode* getElementNode(const std::string TagName, const std::string Attribute, const std::string AttributeValue, xercesc_3_1::DOMNodeList* NodeListe);
    bool existElement(const std::string TagName, const std::string Attribute, xercesc_3_1::DOMNodeList* NodeListe);
    bool existElement(const std::string TagName, const std::string Attribute, const std::string AttributeValue, xercesc_3_1::DOMNodeList* NodeListe);
    std::pair<std::string, bool> getAttributeValue(const std::string TagName, const std::string Attribute, xercesc_3_1::DOMNodeList* NodeListe);

};
} // end namespace
#endif
