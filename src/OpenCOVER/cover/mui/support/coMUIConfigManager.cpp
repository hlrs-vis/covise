#include "coMUIConfigManager.h"
#include "coMUIDefaultValues.h"
#include "../coMUIContainer.h"
#include "coMUIConfigParser.h"
#include "coMUIElementManager.h"
#include "coMUIPositionManager.h"
#include <cover/coTabletUI.h>


coMUIConfigManager* coMUIConfigManager::Instance=NULL;

// access to constructor only by this method
coMUIConfigManager* coMUIConfigManager::getInstance()
{
    if (!Instance)
    {
        Instance=new coMUIConfigManager;
    }
    return Instance;
}

bool coMUIConfigManager::ConfigFileExists()
{
    return FileExists;
}

//constructor:
coMUIConfigManager::coMUIConfigManager()
{
    DefaultValues.reset(new coMUIDefaultValues());
    ElementManager.reset(new coMUIElementManager());
    PositionManager.reset(new coMUIPositionManager());
    FileExists = false;
    ConfigFile="";
}

//destructor:
coMUIConfigManager::~coMUIConfigManager()
{
}


// delete instance of coMUIConfigManager
void coMUIConfigManager::removeInstance()
{
    //delete Instance;
    //Instance = nullptr;
}

// returns Label from ConfigFile if defined in ConfigFile; otherweise returns Label from input
std::string coMUIConfigManager::getCorrectLabel(std::string Label, std::string UI, std::string Device, std::string UniqueIdentifier)
{
   if (ConfigFileExists())                                                            // check if configuration file exists
   {
        std::pair<std::string, bool> parsedLabel=parser->getLabel(UI,Device,UniqueIdentifier);
        if (parsedLabel.second)                                                        // label exists
        {
            return parsedLabel.first;
        }
    }
    return Label;
}

// returns Visible-Value from ConfigFile if defined in ConfigFile; otherwise returns Visible-Value from input
bool coMUIConfigManager::getCorrectVisible(bool visible, std::string UI, std::string Device, std::string UniqueIdentifier)
{
    if (ConfigFileExists())                                                            // configuration file existiert
    {
        return parser->getIsVisible(UI, Device, UniqueIdentifier);
    }
    return visible;
}

// returns Parent from ConfigFile if defined; otherwise returns Parent from input
coMUIContainer* coMUIConfigManager::getCorrectParent(coMUIContainer* Parent, std::string UI, std::string Device, std::string UniqueIdentifier)
{
    if (ConfigFileExists())
    {
        std::pair<std::string, bool> parentName=parser->getParent(UI,Device, UniqueIdentifier);
        if (parentName.second)                     // no entry for parent in configuration file
        {
            return ElementManager->getContainerByIdentifier(parentName.first);
        }
    }
    return Parent;
}

// returns Pos from ConfigFile if defined; otherwise returns Pos from input
std::pair<int,int> coMUIConfigManager::getCorrectPos(std::pair<int,int> pos, std::string UI, std::string Device, std::string UniqueIdentifier)
{
    if (ConfigFileExists())
    {
        std::pair<std::pair<int,int>, bool> ParsedPosition = parser->getPosition(UI, Device, UniqueIdentifier);
        if (ParsedPosition.second)
        {
            return ParsedPosition.first;
        }
    }
    return pos;
}

// returns Pos from ConfigFile if defined; otherwise returns Pos from autoassign
std::pair<int,int> coMUIConfigManager::getCorrectPos(std::string UI, std::string Device, std::string UniqueIdentifier, std::string ParentUniqueIdentifier)
{
    if (ConfigFileExists())
    {
        std::pair<std::pair<int,int>, bool> ParsedPosition = parser->getPosition(UI, Device, UniqueIdentifier);
        if (ParsedPosition.second)
        {
            return ParsedPosition.first;                         // return the matchin position set in configuration file
        }
    }
    return getFreePos(ParentUniqueIdentifier);
}

// returns Pos from ConfigFile if defined; otherwise returns Pos from autoassign after inputPos
std::pair<int,int> coMUIConfigManager::getCorrectPosExceptOfPos(std::vector<std::pair<int,int> > exceptPos, std::string UI, std::string Device, std::string UniqueIdentifier, std::string ParentUniqueIdentifier)
{
    if (ConfigFileExists())
    {
        std::pair<std::pair<int,int>, bool> ParsedPosition = parser->getPosition(UI, Device, UniqueIdentifier);
        if (ParsedPosition.second)
        {
            return ParsedPosition.first;
        }
    }
    else if (!ConfigFileExists())
    {

    }
    return getFreePosExceptOfPos(exceptPos, ParentUniqueIdentifier);
}

//****************************************************************************************************
// Parser
// returns Label from ConfigFile
std::pair<std::string, bool> coMUIConfigManager::getLabel(const std::string UI, const std::string Class, const std::string UniqueIdentifier)
{
    return parser->getLabel(UI, Class, UniqueIdentifier);
}

// returns the adress of the configuration file
const std::string coMUIConfigManager::getConfigFile()
{
    return ConfigFile;
}

// sets the adress of configuration file for the coMUIConfigParser for the first time and reads the configuration file
void coMUIConfigManager::setAdress(const std::string ConfigAdress)
{
    if (ConfigFile.empty())
    {
        ConfigFile = ConfigAdress;
        parser.reset(new coMUIConfigParser(ConfigAdress));
        FileExists = true;
    }
    else if(ConfigFile != ConfigAdress)
    {
        std::cerr << "coMUIConfigManager.cpp::setAdress: ConfigAdress already exists; to set new one use coMUIConfigManager::overwriteAdress(std::string ConfigAdress)" << std::endl;
    }
}

// sets new adress for- and reads new configuration file
void coMUIConfigManager::overwriteAdress(const std::string ConfigAdress)
{
    ConfigFile=ConfigAdress;
    parser->readNewFile(ConfigAdress);
}

//****************************************************************************************************
// DefaultValues
// returns keywords
std::string coMUIConfigManager::keywordCAVE()
{
    return DefaultValues->getKeywordCAVE();
}

std::string coMUIConfigManager::keywordTablet()
{
    return DefaultValues->getKeywordTablet();
}

std::string coMUIConfigManager::keywordPhone()
{
    return DefaultValues->getKeywordPhone();
}

std::string coMUIConfigManager::keywordTUI()
{
    return DefaultValues->getKeywordTUI();
}

std::string coMUIConfigManager::keywordVRUI()
{
    return DefaultValues->getKeywordVRUI();
}

std::string coMUIConfigManager::keywordPowerwall()
{
    return DefaultValues->getKeywordPowerwall();
}

std::string coMUIConfigManager::keywordMainWindow()
{
    return DefaultValues->getKeywordMainWindow();
}

std::string coMUIConfigManager::keywordVisible()
{
    return DefaultValues->getKeywordVisible();
}

std::string coMUIConfigManager::keywordParent()
{
    return DefaultValues->getKeywordParent();
}

std::string coMUIConfigManager::keywordXPosition()
{
    return DefaultValues->getKeywordXPosition();
}

std::string coMUIConfigManager::keywordYPosition()
{
    return DefaultValues->getKeywordYPosition();
}

std::string coMUIConfigManager::keywordLabel()
{
    return DefaultValues->getKeywordLabel();
}

std::string coMUIConfigManager::keywordClass()
{
    return DefaultValues->getKeywordClass();
}
//****************************************************************************************************
// ElementManager
// prints names of all elements
void coMUIConfigManager::printElementNames()
{
    ElementManager->printNames();
}

// add Elements to ElementList
void coMUIConfigManager::addElement(std::string UniqueIdentifier, coMUIContainer* Parent)
{
    ElementManager->addElement(UniqueIdentifier, Parent);
}

void coMUIConfigManager::addElement(std::string UniqueIdentifier, coMUIWidget* Widget)
{
    ElementManager->addElement(UniqueIdentifier, Widget);
}

// delete Element from ElementList
void coMUIConfigManager::removeElement(std::string UniqueIdentifier)
{
    ElementManager->removeElement(UniqueIdentifier);
}

// checks if Element is a container
bool coMUIConfigManager::isElementContainer(std::string UniqueIdentifier)
{
    return ElementManager->isContainer(UniqueIdentifier);
}

// returns the container with name UniqueIdentifier
coMUIContainer* coMUIConfigManager::getContainerByIdentifier(std::string UniqueIdentifier)
{
    return ElementManager->getContainerByIdentifier(UniqueIdentifier);
}

// returns the widget with name UniqueIdentifier
coMUIWidget* coMUIConfigManager::getWidgetByIdentifier(std::string UniqueIdentifier)
{
    return ElementManager->getWidgetByIdentifier(UniqueIdentifier);
}

//***************************************************************************
// PositionManager
// adds position/element to PositionManager
void coMUIConfigManager::addPosToPosList(std::string UniqueIdentifier, std::pair<int,int> pos, std::string UniqueIdentifierParent, bool autoassigned)
{
    PositionManager->addPosToPosList(UniqueIdentifier, pos, UniqueIdentifierParent, autoassigned);
}

std::pair<int,int> coMUIConfigManager::getFreePosExceptOfPos(std::vector<std::pair<int,int> > exceptPos, std::string ParentUniqueIdentifier)
{
    return PositionManager->getFreePosExeptOfPos(exceptPos, ParentUniqueIdentifier);
}

// get position of element
std::pair <int,int> coMUIConfigManager::getPosOfElement(std::string UniqueIdentifier)
{
    return PositionManager->getPosOfElement(UniqueIdentifier);
}

// get next free position
std::pair <int,int> coMUIConfigManager::getFreePos(std::string UniqueIdentifierParent)
{
    return PositionManager->getFreePos(UniqueIdentifierParent);
}

// delete position/element from PositionManager
void coMUIConfigManager::deletePosFromPosList(std::string UniqueIdentifier)
{
    PositionManager->deletePosFromPosList(UniqueIdentifier);
}

// changes position of element in PositionManager
void coMUIConfigManager::changePos(std::string UniqueIdentifier, std::pair<int,int> Pos)
{
    PositionManager->changePosInPosList(UniqueIdentifier, Pos);
}

// returns true, if position is occupied, false if position is free
bool coMUIConfigManager::isPosOccupied(std::pair<int,int> pos, std::string UniqueIdentifierParent)
{
    return PositionManager->isPosOccupied(pos, UniqueIdentifierParent);
}

// returns true, if position is occupied by autoassigned element
bool coMUIConfigManager::isPosAutoassigned(std::pair<int,int> pos, std::string UniqueIdentifierParent)
{
    return PositionManager->isAutoassigned(pos, UniqueIdentifierParent);
}

// returns UniqueIdentifier of element
std::string coMUIConfigManager::getIdentifierByPos(std::pair<int,int> pos, std::string UniqueIdentifierParent)
{
    return PositionManager->getIdentifierByPos(pos, UniqueIdentifierParent);
}

// prepares position for new element: sets autoassigned element to new position, if occupied by autoassigned element
void coMUIConfigManager::preparePos(std::pair<int,int> pos, std::string ParentUniqueIdentifier)
{
    if (isPosOccupied(pos, ParentUniqueIdentifier))                                         // Position is occupied
    {
        if (isPosAutoassigned(pos, ParentUniqueIdentifier))
        {
            std::string Identifier=getIdentifierByPos(pos,ParentUniqueIdentifier);          // UniqueIdentifier of
            std::pair <int,int> Pos = getFreePos(ParentUniqueIdentifier);                   // next free Position
            setAutoassignedPos(Pos, Identifier, ParentUniqueIdentifier);
        }
        else if (!isPosAutoassigned(pos, ParentUniqueIdentifier))
        {
            std::cerr << "ERROR: coMUIConfigManager::preparePos(): Position (" << pos.first << "," << pos.second << ") in Parent " << ParentUniqueIdentifier << " is occupied by " << getIdentifierByPos(pos,ParentUniqueIdentifier) << " and will be deleted and overwritten." << std::endl;
            std::string Identifier=getIdentifierByPos(pos,ParentUniqueIdentifier);          // UniqueIdentifier of
            deletePosFromPosList(Identifier);
        }
    }
}

//! returns true, if Attribute exists in ConfigFile with UI, Device and Identifier; else returns false
bool coMUIConfigManager::existAttributeInConfigFile(std::string Attribute, std::string UI, std::string Device, std::string Identifier)
{
    if (ConfigFileExists())
    {
        return parser->existAttributeInConfigFile(Attribute, UI, Device, Identifier);
    }
    return false;
}

void coMUIConfigManager::setAutoassignedPos(std::pair<int,int> Pos, std::string ElementIdentifier, std::string ParentUniqueIdentifier)
{
    deletePosFromPosList(ElementIdentifier);
    if (isElementContainer(ElementIdentifier))
    {
        getContainerByIdentifier(ElementIdentifier)->getTUI()->setPos(Pos.first,Pos.second);         // Element neu positionieren
    }
    else if(!isElementContainer(ElementIdentifier))
    {
        getWidgetByIdentifier(ElementIdentifier)->getTUI()->setPos(Pos.first,Pos.second);            // Element neu positionieren
    }else
    {
        std::cerr << "ERROR: coMUIConfigManager::setAutoassignedPos(): " << ElementIdentifier << " is no coMUIWidget or coMUIContainer" << std::endl;
    }
    addPosToPosList(ElementIdentifier, Pos, ParentUniqueIdentifier, true);
}

std::string coMUIConfigManager::getPos2Print()
{
    return PositionManager->printPos();
}

