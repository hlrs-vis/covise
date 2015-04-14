#ifndef MUICONFIGPARSER_H
#define MUICONFIGPARSER_H

#include <boost/smart_ptr.hpp>

#include <xercesc/parsers/XercesDOMParser.hpp>
#include <xercesc/dom/DOMDocument.hpp>
#include <xercesc/dom/DOM.hpp>

#include "DefaultValues.h"

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
    bool fileExist(std::string File);

    std::pair<bool,bool> getIsVisible(mui::UITypeEnum UI, mui::DeviceTypesEnum Device, std::string UniqueIdentifier);
    std::pair<std::string, bool> getParent(mui::UITypeEnum UI, mui::DeviceTypesEnum Device, std::string UniquIdentifier);
    std::pair<std::string, bool> getLabel(mui::UITypeEnum UI, mui::DeviceTypesEnum Device, std::string UniqueIdentifier);
    std::pair<std::pair<int,int>, bool> getPos(mui::UITypeEnum UI, mui::DeviceTypesEnum Device, std::string UniqueIdentifier);
    std::pair<std::string, bool> getAttributeValue(mui::UITypeEnum UI, mui::DeviceTypesEnum Device, std::string UniqueIdentifier, mui::AttributesEnum Attribute);

private:
    // membervariables:
    boost::shared_ptr<xercesc::XercesDOMParser> parser;
    xercesc::DOMDocument* parsedDoc;
    xercesc::DOMElement* rootElement;
    xercesc::DOMNodeList* nodeList;

    boost::shared_ptr<DefaultValues> defaultValues;


    // memberfunctions:
    void initializeParser(std::string adress);

    std::pair<xercesc::DOMNode*,bool> getDeviceEntryNode(mui::UITypeEnum UI, mui::DeviceTypesEnum Device);
    std::pair<xercesc::DOMElement*,bool> getElement(std::string UniqueIdentifier, xercesc::DOMNodeList* DeviceNodeChilds);
    bool existAttributeInConfigFile(UITypeEnum UI, DeviceTypesEnum Device, std::string UniqueIdentifier, AttributesEnum Attribute);

};
} // end namespace
#endif
