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
    bool configFileExists();

    // memberfunctions
    //************************************************************
    // Parser
    const std::string getConfigFile();
    //void setAdress(const std::string ConfigAdress);
    std::pair<std::string, bool> getLabel(mui::UITypeEnum UI, mui::DeviceTypesEnum Device, const std::string uniqueIdentifier);
    bool existAttributeInConfigFile(UITypeEnum UI, DeviceTypesEnum Device, std::string uniqueIdentifier, AttributesEnum Attribute);

    //************************************************************
    // PositionManager
    void addPosToPosList(std::string uniqueIdentifier, std::pair<int,int> pos, std::string parentUniqueIdentifier, bool autoassigned);
    std::pair <int,int> getPosOfElement(std::string uniqueIdentifier);
    std::pair <int,int> getFreePos(std::string parentUniqueIdentifier);
    void deletePosFromPosList(std::string uniqueIdentifier);
    void changePos(std::string uniqueIdentifier, std::pair<int,int> pos);
    bool isPosOccupied(std::pair<int,int> pos, std::string parentUniqueIdentifier);
    bool isPosAutoassigned(std::pair<int,int> pos, std::string parentUniqueIdentifier);
    std::string getIdentifierByPos(std::pair<int,int> pos, std::string parentUniqueIdentifier);
    std::string getPos2Print();


    //************************************************************
    // ElementManager
    void printElementNames();
    mui::Element* getElementByIdentifier(std::string uniqueIdentifier);
    bool isElementContainer(std::string uniqueIdentifier);
    void addElement(std::string uniqueIdentifier, mui::Element* parent);
    void removeElement(std::string uniqueIdentifier);
    //************************************************************
    // Rest
    std::string getCorrectLabel(std::string Label, mui::UITypeEnum UI, mui::DeviceTypesEnum Device, std::string uniqueIdentifier);
    bool getCorrectVisible(bool visible, mui::UITypeEnum UI, mui::DeviceTypesEnum Device, std::string uniqueIdentifier);
    std::pair<int, int> getCorrectPos(std::pair<int,int> pos, mui::UITypeEnum UI, mui::DeviceTypesEnum Device, std::string uniqueIdentifier);
    std::pair<int, int> getCorrectPos(mui::UITypeEnum UI, mui::DeviceTypesEnum Device, std::string uniqueIdentifier, std::string parentUniqueIdentifier);
    std::pair<int,int> getCorrectPosExceptOfPos(std::vector<std::pair<int,int> > exceptPos, mui::UITypeEnum UI, mui::DeviceTypesEnum Device, std::string uniqueIdentifier, std::string parentUniqueIdentifier);
    mui::Container* getCorrectParent(Container* Parent, mui::UITypeEnum UI, mui::DeviceTypesEnum Device, std::string uniqueIdentifier);
    void preparePos(std::pair<int,int> pos, std::string parentUniqueIdentifier);


private:
    boost::shared_ptr<ConfigParser> parser;
    boost::shared_ptr<DefaultValues> defaultValues;
    boost::shared_ptr<ElementManager> elementManager;
    boost::shared_ptr<PositionManager> positionManager;

    void setAutoassignedPos(std::pair<int,int> Pos, std::string uniqueIdentifier, std::string parentUniqueIdentifier);
    std::pair <int,int> getFreePosExceptOfPos(std::vector<std::pair <int,int> > exceptPos, std::string parentUniqueIdentifier);
    static ConfigManager *Instance;

    bool fileExists;

    //constructor and destructor:
    ConfigManager();
    ~ConfigManager();

    // membervariables
    std::string configFile;

    // not implemented
    ConfigManager(const ConfigManager&);
    ConfigManager& operator=(const ConfigManager&);
};
} // end namespace

#endif
