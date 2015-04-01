#ifndef MUICONFIGMANAGER_H
#define MUICONFIGMANAGER_H

#include <iostream>
#include <vector>
#include <util/coExport.h>
#include <boost/smart_ptr.hpp>
#include "DefaultValues.h"


namespace mui
{
//forwarddeclaration:
class Container;
class Widget;
class Element;
class DefaultValues;
class ConfigParser;
class ElementManager;
class PositionManager;

// class:
class COVEREXPORT ConfigManager
{
public:
    static ConfigManager *getInstance();
    static void removeInstance();
    bool ConfigFileExists();

    // memberfunctions
    //************************************************************
    // Parser
    const std::string getConfigFile();
    void setAdress(const std::string ConfigAdress);
    std::pair<std::string, bool> getLabel(mui::UITypeEnum UI, mui::DeviceTypesEnum Device, const std::string UniqueIdentifier);
    bool existAttributeInConfigFile(UITypeEnum UI, DeviceTypesEnum Device, std::string UniqueIdentifier, AttributesEnum Attribute);

    //************************************************************
    // PositionManager
    void addPosToPosList(std::string UniqueIdentifier, std::pair<int,int> pos, std::string UniqueIdentifierParent, bool autoassigned);
    void addPosToPosList(std::string UniqueIdentifier, std::pair<int,int> pos, mui::AttributesEnum ParentUniqueIdentifier, bool autoassigned);
    std::pair <int,int> getPosOfElement(std::string UniqueIdentifier);
    std::pair <int,int> getFreePos(std::string UniqueIdentifierParent);
    void deletePosFromPosList(std::string UniqueIdentifier);
    void changePos(std::string UniqueIdentifier, std::pair<int,int> pos);
    bool isPosOccupied(std::pair<int,int> pos, std::string UniqueIdentifierParent);
    bool isPosAutoassigned(std::pair<int,int> pos, std::string UniqueIdentifierParent);
    std::string getIdentifierByPos(std::pair<int,int> pos, std::string UniqueIdentifierParent);
    std::string getPos2Print();


    //************************************************************
    // ElementManager
    void printElementNames();
    Container* getContainerByIdentifier(std::string UniqueIdentifier);
    Widget* getWidgetByIdentifier(std::string UniqueIdentifier);
    bool isElementContainer(std::string UniqueIdentifier);
    void addElement(std::string UniqueIdentifier, Container* Parent);
    void addElement(std::string UniqueIdentifier, Widget* Widget);
    void removeElement(std::string UniqueIdentifier);
    //************************************************************
    // Rest
    std::string getCorrectLabel(std::string Label, mui::UITypeEnum UI, mui::DeviceTypesEnum Device, std::string UniqueIdentifier);
    bool getCorrectVisible(bool visible, mui::UITypeEnum UI, mui::DeviceTypesEnum Device, std::string UniqueIdentifier);
    std::pair<int, int> getCorrectPos(std::pair<int,int> pos, mui::UITypeEnum UI, mui::DeviceTypesEnum Device, std::string UniqueIdentifier);
    std::pair<int, int> getCorrectPos(mui::UITypeEnum UI, mui::DeviceTypesEnum Device, std::string UniqueIdentifier, std::string ParentUniqueIdentifier);
    std::pair<int, int> getCorrectPos(mui::UITypeEnum UI, mui::DeviceTypesEnum Device, std::string UniqueIdentifier, mui::AttributesEnum Attribute);
    std::pair<int,int> getCorrectPosExceptOfPos(std::vector<std::pair<int,int> > exceptPos, mui::UITypeEnum UI, mui::DeviceTypesEnum Device, std::string UniqueIdentifier, std::string ParentUniqueIdentifier);
    Container* getCorrectParent(Container* Parent, mui::UITypeEnum UI, mui::DeviceTypesEnum Device, std::string UniqueIdentifier);
    void preparePos(std::pair<int,int> pos, std::string ParentUniqueIdentifier);


private:
    boost::shared_ptr<ConfigParser> parser;
    boost::shared_ptr<DefaultValues> defaultValues;
    boost::shared_ptr<ElementManager> elementManager;
    boost::shared_ptr<PositionManager> positionManager;

    void setAutoassignedPos(std::pair<int,int> Pos, std::string Identifier, std::string ParentUniqueIdentifier);
    std::pair <int,int> getFreePosExceptOfPos(std::vector<std::pair <int,int> > exceptPos, std::string UniqueIdentifierParent);
    static ConfigManager *Instance;

    bool FileExists;

    //constructor and destructor:
    ConfigManager();
    ~ConfigManager();

    // membervariables
    std::string ConfigFile;

    // not implemented
    ConfigManager(const ConfigManager&);
    ConfigManager& operator=(const ConfigManager&);
};
} // end namespace

#endif
