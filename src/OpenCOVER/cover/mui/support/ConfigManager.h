#ifndef MUICONFIGMANAGER_H
#define MUICONFIGMANAGER_H

#include <iostream>
#include <vector>
#include <util/coExport.h>
#include "boost/smart_ptr.hpp"


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
    void overwriteAdress(std::string ConfigAdress);
    std::pair<std::string, bool> getLabel(const std::string UI, const std::string Klasse, const std::string UniqueIdentifier);

    //************************************************************
    // PositionManager
    void addPosToPosList(std::string UniqueIdentifier, std::pair<int,int> pos, std::string UniqueIdentifierParent, bool autoassigned);
    std::pair <int,int> getPosOfElement(std::string UniqueIdentifier);
    std::pair <int,int> getFreePos(std::string UniqueIdentifierParent);
    void deletePosFromPosList(std::string UniqueIdentifier);
    void changePos(std::string UniqueIdentifier, std::pair<int,int> pos);
    bool isPosOccupied(std::pair<int,int> pos, std::string UniqueIdentifierParent);
    bool isPosAutoassigned(std::pair<int,int> pos, std::string UniqueIdentifierParent);
    std::string getIdentifierByPos(std::pair<int,int> pos, std::string UniqueIdentifierParent);
    std::string getPos2Print();

    //************************************************************
    // DefaultValues
    std::string keywordCAVE();
    std::string keywordTablet();
    std::string keywordPhone();
    std::string keywordTUI();
    std::string keywordVRUI();
    std::string keywordPowerwall();
    std::string keywordMainWindow();
    std::string keywordVisible();
    std::string keywordParent();
    std::string keywordXPosition();
    std::string keywordYPosition();
    std::string keywordLabel();
    std::string keywordClass();

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
    std::string getCorrectLabel(std::string Label, std::string UI, std::string Device, std::string UniqueIdentifier);
    bool getCorrectVisible(bool visible, std::string UI, std::string Device, std::string UniqueIdentifier);
    std::pair<int, int> getCorrectPos(std::pair<int,int> pos, std::string UI, std::string Device, std::string UniqueIdentifier);
    std::pair<int, int> getCorrectPos(std::string UI, std::string Device, std::string UniqueIdentifier, std::string ParentUniqueIdentifier);
    std::pair<int,int> getCorrectPosExceptOfPos(std::vector<std::pair<int,int> > exceptPos, std::string UI, std::string Device, std::string UniqueIdentifier, std::string ParentUniqueIdentifier);
    Container* getCorrectParent(Container* Parent, std::string UI, std::string Device, std::string UniqueIdentifier);
    void preparePos(std::pair<int,int> pos, std::string ParentUniqueIdentifier);
    bool existAttributeInConfigFile(std::string Attribute, std::string UI, std::string Device, std::string Identifier);


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
