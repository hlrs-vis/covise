#include "coMUIConfigManager.h"
#include "coMUIDefaultValues.h"
#include "../coMUIContainer.h"
#include "coMUISupport.h"
#include "coMUIConfigParser.h"
#include "coMUIElementManager.h"
#include "coMUIPositionManager.h"


coMUIConfigManager* coMUIConfigManager::Instance=NULL;

// access to constructor only by this method
coMUIConfigManager* coMUIConfigManager::getInstance(){
    if (!Instance){
        Instance=new coMUIConfigManager;
    }
    return Instance;
}

bool coMUIConfigManager::ConfigFileExists(){
    return FileExists;
}

//constructor:
coMUIConfigManager::coMUIConfigManager(){
    DefaultValues.reset(new coMUIDefaultValues());
    ElementManager.reset(new coMUIElementManager());
    PositionManager.reset(new coMUIPositionManager());
    FileExists = false;
    ConfigFile="";
}

//destructor:
coMUIConfigManager::~coMUIConfigManager(){
}


// delete instance of coMUIConfigManager
void coMUIConfigManager::removeInstance(){
    //delete Instance;
    //Instance = nullptr;
}

// returns Label from ConfigFile if defined in ConfigFile; otherweise returns Label from input
std::string coMUIConfigManager::getCorrectLabel(std::string Label, std::string UI, std::string Device, std::string UniqueIdentifier){
   if (ConfigFileExists()){                                                            // check if configuration file exists
        std::pair<std::string, bool> parsedLabel=parser->getLabel(UI,Device,UniqueIdentifier);
        if (parsedLabel.second){                                                        // label exists
            return parsedLabel.first;
        }
    }
    return Label;
}

// returns Visible-Value from ConfigFile if defined in ConfigFile; otherwise returns Visible-Value from input
bool coMUIConfigManager::getCorrectVisible(bool visible, std::string UI, std::string Device, std::string UniqueIdentifier){
    if (ConfigFileExists()){                                                            // configuration file existiert
        return parser->getIsVisible(UI, Device, UniqueIdentifier);
    }
    return visible;
}

// returns Parent from ConfigFile if defined; otherwise returns Parent from input
coMUIContainer* coMUIConfigManager::getCorrectParent(coMUIContainer* Parent, std::string UI, std::string Device, std::string UniqueIdentifier){
    if (ConfigFileExists()){
        std::pair<std::string, bool> parentName=parser->getParent(UI,Device, UniqueIdentifier);
        if (parentName.second){                     // no entry for parent in configuration file
            return ElementManager->getContainerByIdentifier(parentName.first);
        }
    }
    return Parent;
}

// returns X-Pos from ConfigFile if defined; otherwise returns X-Pos from input
int coMUIConfigManager::getCorrectPosX(int posx, std::string UI, std::string Device, std::string UniqueIdentifier){
    if (ConfigFileExists()){
        std::pair<std::string, bool> position=parser->getPosition(UI, Device, UniqueIdentifier);
        if (position.second){
            return coMUISupport::readIntFromString(position.first,1);
        }
    }
    return posx;
}

// returns X-Pos from ConfigFile if defined; otherwise returns X-Pos from autoassign
int coMUIConfigManager::getCorrectPosX(std::string UI, std::string Device, std::string UniqueIdentifier, std::string ParentUniqueIdentifier){
    if (ConfigFileExists()){
        std::pair<std::string, bool> position=parser->getPosition(UI, Device, UniqueIdentifier);
        if (position.second){
            return coMUISupport::readIntFromString(position.first,1);                         // return the matchin position set in configuration file
        }
    }
    return getFreePos(ParentUniqueIdentifier)[0];
}

// returns Y-Pos from ConfigFile if defined; otherwise returns Y-Pos from input
int coMUIConfigManager::getCorrectPosY(int posy, std::string UI, std::string Device, std::string UniqueIdentifier){
    if (ConfigFileExists()){
        std::pair<std::string, bool> position=parser->getPosition(UI, Device, UniqueIdentifier);
        if (position.second){
            return coMUISupport::readIntFromString(position.first,2);
        }
    }
    return posy;
}

// returns Y-Pos from ConfigFile if defined; otherwise returns Y-Pos from input
int coMUIConfigManager::getCorrectPosY(std::string UI, std::string Device, std::string UniqueIdentifier, std::string ParentUniqueIdentifier){
    if (ConfigFileExists()){
        std::pair<std::string, bool> position=parser->getPosition(UI, Device, UniqueIdentifier);
        if (position.second){
            return coMUISupport::readIntFromString(position.first,2);
        }
    }
    return getFreePos(ParentUniqueIdentifier)[1];
}


//****************************************************************************************************
// Parser
// returns Label from ConfigFile
std::pair<std::string, bool> coMUIConfigManager::getLabel(const std::string UI, const std::string Class, const std::string UniqueIdentifier){
    return parser->getLabel(UI, Class, UniqueIdentifier);
}

// returns the adress of the configuration file
const std::string coMUIConfigManager::getConfigFile(){
    return ConfigFile;
}

// sets the adress of configuration file for the coMUIConfigParser for the first time and reads the configuration file
void coMUIConfigManager::setAdress(const std::string ConfigAdress){
    if (ConfigFile.empty()){
        ConfigFile = ConfigAdress;
        parser.reset(new coMUIConfigParser(ConfigAdress));
        FileExists = true;
    } else if(ConfigFile != ConfigAdress){
        std::cerr << "coMUIConfigManager.cpp::setAdress: ConfigAdress already exists; to set new one use coMUIConfigManager::overwriteAdress(std::string ConfigAdress)" << std::endl;
    }
}

// sets new adress for- and reads new configuration file
void coMUIConfigManager::overwriteAdress(const std::string ConfigAdress){
    ConfigFile=ConfigAdress;
    parser->readNewFile(ConfigAdress);
}

//****************************************************************************************************
// DefaultValues
// returns keywords
std::string coMUIConfigManager::keywordCAVE(){
    return DefaultValues->getKeywordCAVE();
}

std::string coMUIConfigManager::keywordTablet(){
    return DefaultValues->getKeywordTablet();
}

std::string coMUIConfigManager::keywordPhone(){
    return DefaultValues->getKeywordPhone();
}

std::string coMUIConfigManager::keywordTUI(){
    return DefaultValues->getKeywordTUI();
}

std::string coMUIConfigManager::keywordVRUI(){
    return DefaultValues->getKeywordVRUI();
}

std::string coMUIConfigManager::keywordPowerwall(){
    return DefaultValues->getKeywordPowerwall();
}

std::string coMUIConfigManager::keywordMainWindow(){
    return DefaultValues->getKeywordMainWindow();
}

//****************************************************************************************************
// ElementManager
// prints names of all elements
void coMUIConfigManager::printElementNames(){
    ElementManager->printNames();
}

// add Elements to ElementList
void coMUIConfigManager::addElement(std::string UniqueIdentifier, coMUIContainer* Parent){
    ElementManager->addElement(UniqueIdentifier, Parent);
}

void coMUIConfigManager::addElement(std::string UniqueIdentifier, coMUIWidget* Widget){
    ElementManager->addElement(UniqueIdentifier, Widget);
}

// delete Element from ElementList
void coMUIConfigManager::removeElement(std::string UniqueIdentifier){
    ElementManager->removeElement(UniqueIdentifier);
}

// checks if Element is a container
bool coMUIConfigManager::isElementContainer(std::string UniqueIdentifier){
    return ElementManager->isContainer(UniqueIdentifier);
}

// returns the container with name UniqueIdentifier
coMUIContainer* coMUIConfigManager::getContainerByIdentifier(std::string UniqueIdentifier){
    return ElementManager->getContainerByIdentifier(UniqueIdentifier);
}

// returns the widget with name UniqueIdentifier
coMUIWidget* coMUIConfigManager::getWidgetByIdentifier(std::string UniqueIdentifier){
    return ElementManager->getWidgetByIdentifier(UniqueIdentifier);
}

//***************************************************************************
// PositionManager
// adds position/element to PositionManager
void coMUIConfigManager::addPos(std::string UniqueIdentifier, int posx, int posy, std::string UniqueIdentifierParent, bool autoassigned){
    PositionManager->addPos(UniqueIdentifier, posx, posy, UniqueIdentifierParent, autoassigned);
}

// get position of element
std::vector <int> coMUIConfigManager::getPos(std::string UniqueIdentifier){
    return PositionManager->getPos(UniqueIdentifier);
}

// get next free position
std::vector <int> coMUIConfigManager::getFreePos(std::string UniqueIdentifierParent){
    return PositionManager->getFreePos(UniqueIdentifierParent);
}

// delete position/element from PositionManager
void coMUIConfigManager::deletePos(std::string UniqueIdentifier){
    PositionManager->deletePos(UniqueIdentifier);
}

// changes position of element in PositionManager
void coMUIConfigManager::changePos(std::string UniqueIdentifier, int xPos, int yPos){
    PositionManager->changePos(UniqueIdentifier, xPos, yPos);
}

// returns true, if position is occupied, false if position is free
bool coMUIConfigManager::isPosOccupied(int posx, int posy, std::string UniqueIdentifierParent){
    return PositionManager->isOccupied(posx, posy, UniqueIdentifierParent);
}

// returns true, if position is occupied by autoassigned element
bool coMUIConfigManager::isAutoassigned(int posx, int posy, std::string UniqueIdentifierParent){
    return PositionManager->isAutoassigned(posx, posy, UniqueIdentifierParent);
}

// returns UniqueIdentifier of element
std::string coMUIConfigManager::getIdentifier(int posx, int posy, std::string UniqueIdentifierParent){
    return PositionManager->getIdentifier(posx, posy, UniqueIdentifierParent);
}

// prepares position for new element: sets autoassigned element to new position, if occupied by autoassigned element
void coMUIConfigManager::preparePos(int posx, int posy, std::string ParentUniqueIdentifier){
    if (isPosOccupied(posx, posy, ParentUniqueIdentifier)){                             // Position is occupied
        if (isAutoassigned(posx, posy, ParentUniqueIdentifier)){
            std::string Identifier=getIdentifier(posx,posy,ParentUniqueIdentifier);     // UniqueIdentifier of
            std::vector <int> pos = getFreePos(ParentUniqueIdentifier);                 // next free Position
            changePos(Identifier,pos[0],pos[1]);
            if (isElementContainer(Identifier)){
                getContainerByIdentifier(Identifier)->setPos(pos[0],pos[1]);            // Element neu positionieren
            } else if(!isElementContainer(Identifier)){
                std::cout << "coMUIConfigManager::preparePos(): " << pos[0] << "," << pos[1] << std::endl;
                std::cout << "coMUIConfigManager::preparePos(): " << getWidgetByIdentifier(Identifier)->getUniqueIdentifier() << std::endl;
                getWidgetByIdentifier(Identifier)->setPos(pos[0],pos[1]);               // Element neu positionieren
            }else{
                std::cerr << "ERROR: coMUIConfigManager::preparePos(): " << Identifier << " is no coMUIWidget or coMUIContainer" << std::endl;
            }
        }
    } else if (isPosOccupied(posx, posy, ParentUniqueIdentifier) && !isAutoassigned(posx, posy, ParentUniqueIdentifier)){
        std::cerr << "WARNING: coMUIConfigManager::preparePos(): Position (" << posx << "," << posy << ") in Parent " << ParentUniqueIdentifier << " is occupied." << std::endl;
    }
}

std::string coMUIConfigManager::getPos2Print(){
    return PositionManager->printPos();
}

