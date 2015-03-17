#ifndef COMUICONFIGMANAGER_H
#define COMUICONFIGMANAGER_H

#include <iostream>
#include <vector>
#include "boost/smart_ptr.hpp"


//forwarddeclaration:
class coMUIContainer;
class coMUIWidget;
class coMUIElement;
class coMUIDefaultValues;
class coMUIConfigParser;
class coMUIElementManager;
class coMUIPositionManager;

// class:
class coMUIConfigManager{
public:
    static coMUIConfigManager *getInstance();
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
    void addPos(std::string UniqueIdentifier, int posx, int posy, std::string UniqueIdentifierParent, bool autoassigned);
    std::vector <int> getPos(std::string UniqueIdentifier);
    std::vector <int> getFreePos(std::string UniqueIdentifierParent);
    void deletePos(std::string UniqueIdentifier);
    void changePos(std::string UniqueIdentifier, int posx, int posy);
    bool isPosOccupied(int posx, int posy, std::string UniqueIdentifierParent);
    bool isAutoassigned(int posx, int posy, std::string UniqueIdentifierParent);
    std::string getIdentifier(int posx, int posy, std::string UniqueIdentifierParent);
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

    //************************************************************
    // ElementManager
    void printElementNames();
    coMUIContainer* getContainerByIdentifier(std::string UniqueIdentifier);
    coMUIWidget* getWidgetByIdentifier(std::string UniqueIdentifier);
    bool isElementContainer(std::string UniqueIdentifier);
    void addElement(std::string UniqueIdentifier, coMUIContainer* Parent);
    void addElement(std::string UniqueIdentifier, coMUIWidget* Widget);
    void removeElement(std::string UniqueIdentifier);
    //************************************************************
    // Rest
    std::string getCorrectLabel(std::string Label, std::string UI, std::string Device, std::string UniqueIdentifier);
    bool getCorrectVisible(bool visible, std::string UI, std::string Device, std::string UniqueIdentifier);
    int getCorrectPosX(int posx, std::string UI, std::string Device, std::string UniqueIdentifier);
    int getCorrectPosX(std::string UI, std::string Device, std::string UniqueIdentifier, std::string ParentUniqueIdentifier);
    int getCorrectPosY(int posy, std::string UI, std::string Device, std::string UniqueIdentifier);
    int getCorrectPosY(std::string UI, std::string Device, std::string UniqueIdentifier, std::string ParentUniqueIdentifier);
    coMUIContainer* getCorrectParent(coMUIContainer* Parent, std::string UI, std::string Device, std::string UniqueIdentifier);
    void preparePos(int posx, int posy, std::string ParentUniqueIdentifier);


private:
    boost::shared_ptr<coMUIConfigParser> parser;
    boost::shared_ptr<coMUIDefaultValues> DefaultValues;
    boost::shared_ptr<coMUIElementManager> ElementManager;
    boost::shared_ptr<coMUIPositionManager> PositionManager;

    coMUIConfigManager(const coMUIConfigManager&){}
    coMUIConfigManager& operator=(const coMUIConfigManager&){}
    static coMUIConfigManager *Instance;

    bool FileExists;

    //constructor and destructor:
    coMUIConfigManager();
    ~coMUIConfigManager();

    // membervariables
    std::string ConfigFile;
};

#endif
