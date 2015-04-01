#include "ConfigManager.h"
#include "DefaultValues.h"
#include "../Container.h"
#include "ConfigParser.h"
#include "ElementManager.h"
#include "PositionManager.h"
#include <cover/coTabletUI.h>
#include <config/CoviseConfig.h>
#include "DefaultValues.h"

using namespace mui;

ConfigManager* ConfigManager::Instance=NULL;

// access to constructor only by this method
ConfigManager* ConfigManager::getInstance()
{
    if (!Instance)
    {
        Instance=new ConfigManager;
    }
    return Instance;
}

bool ConfigManager::ConfigFileExists()
{
    return FileExists;
}

//constructor:
ConfigManager::ConfigManager()
{
    elementManager.reset(new ElementManager());
    positionManager.reset(new PositionManager());
    FileExists = false;
    ConfigFile=covise::coCoviseConfig::getEntry("value", "COVER.UiConfig", "userinterface.xml");
    if (ConfigFile.empty())
    {
        ConfigFile = "userinterface.xml";
    }
    if (ConfigFile[0] != '/')
    {
        char* covisedir = getenv("COVISEDIR");
        if (covisedir != NULL)
        {
            ConfigFile= std::string(covisedir) + "/config/" + ConfigFile;
        }
    }
    std::cout << "ConfigManager::ConfigManager(): ConfigFile: " << ConfigFile << std::endl;
}

//destructor:
ConfigManager::~ConfigManager()
{
}


// delete instance of ConfigManager
void ConfigManager::removeInstance()
{
    //delete Instance;
    //Instance = nullptr;
}

// returns Label from ConfigFile if defined in ConfigFile; otherweise returns Label from input
std::string ConfigManager::getCorrectLabel(std::string Label, mui::UITypeEnum UI, mui::DeviceTypesEnum Device, std::string UniqueIdentifier)
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
bool ConfigManager::getCorrectVisible(bool visible, mui::UITypeEnum UI, mui::DeviceTypesEnum Device, std::string UniqueIdentifier)
{
    if (ConfigFileExists())                                                            // configuration file existiert
    {
        std::pair<bool,bool> isVisible = parser->getIsVisible(UI, Device, UniqueIdentifier);
        if (isVisible.second)
        {
            return isVisible.first;
        }
    }
    return visible;
}

// returns Parent from ConfigFile if defined; otherwise returns Parent from input
Container* ConfigManager::getCorrectParent(Container* Parent, mui::UITypeEnum UI, mui::DeviceTypesEnum Device, std::string UniqueIdentifier)
{
    if (ConfigFileExists())
    {
        std::pair<std::string, bool> parentName=parser->getParent(UI,Device, UniqueIdentifier);
        if (parentName.second)                     // no entry for parent in configuration file
        {
            return elementManager->getContainerByIdentifier(parentName.first);
        }
    }
    return Parent;
}

// returns Pos from ConfigFile if defined; otherwise returns Pos from input
std::pair<int,int> ConfigManager::getCorrectPos(std::pair<int,int> pos, mui::UITypeEnum UI, mui::DeviceTypesEnum Device, std::string UniqueIdentifier)
{
    if (ConfigFileExists())
    {
        std::pair<std::pair<int,int>, bool> ParsedPosition = parser->getPos(UI, Device, UniqueIdentifier);
        if (ParsedPosition.second)
        {
            return ParsedPosition.first;
        }
    }
    return pos;
}

// returns Pos from ConfigFile if defined; otherwise returns Pos from autoassign
std::pair<int,int> ConfigManager::getCorrectPos(mui::UITypeEnum UI, mui::DeviceTypesEnum Device, std::string UniqueIdentifier, std::string ParentUniqueIdentifier)
{
    if (ConfigFileExists())
    {
        std::pair<std::pair<int,int>, bool> ParsedPosition = parser->getPos(UI, Device, UniqueIdentifier);
        if (ParsedPosition.second)
        {
            return ParsedPosition.first;                         // return the matchin position set in configuration file
        }
    }
    return getFreePos(ParentUniqueIdentifier);
}

// returns Pos from ConfigFile if defined; otherwise returns Pos from autoassign
std::pair<int,int> ConfigManager::getCorrectPos(mui::UITypeEnum UI, mui::DeviceTypesEnum Device, std::string UniqueIdentifier, mui::AttributesEnum Attribute)
{
    if (ConfigFileExists())
    {
        std::pair<std::pair<int,int>, bool> ParsedPosition = parser->getPos(UI, Device, UniqueIdentifier);
        if (ParsedPosition.second)
        {
            return ParsedPosition.first;                         // return the matchin position set in configuration file
        }
    }
    return getFreePos(mui::getKeywordAttribute(Attribute));
}

// returns Pos from ConfigFile if defined; otherwise returns Pos from autoassign after inputPos
std::pair<int,int> ConfigManager::getCorrectPosExceptOfPos(std::vector<std::pair<int,int> > exceptPos, mui::UITypeEnum UI, mui::DeviceTypesEnum Device, std::string UniqueIdentifier, std::string ParentUniqueIdentifier)
{
    if (ConfigFileExists())
    {
        std::pair<std::pair<int,int>, bool> ParsedPosition = parser->getPos(UI, Device, UniqueIdentifier);
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

bool ConfigManager::existAttributeInConfigFile(UITypeEnum UI, DeviceTypesEnum Device, std::string UniqueIdentifier, AttributesEnum Attribute)
{
    if (ConfigFileExists())
    {
        return parser->getAttributeValue(UI, Device, UniqueIdentifier, Attribute).second;
    }
    return false;
}

// returns the adress of the configuration file
const std::string ConfigManager::getConfigFile()
{
    return ConfigFile;
}

//****************************************************************************************************
// ElementManager
// prints names of all elements
void ConfigManager::printElementNames()
{
    elementManager->printNames();
}

// add Elements to ElementList
void ConfigManager::addElement(std::string UniqueIdentifier, Container* Parent)
{
    elementManager->addElement(UniqueIdentifier, Parent);
}

void ConfigManager::addElement(std::string UniqueIdentifier, Widget* Widget)
{
    elementManager->addElement(UniqueIdentifier, Widget);
}

// delete Element from ElementList
void ConfigManager::removeElement(std::string UniqueIdentifier)
{
    elementManager->removeElement(UniqueIdentifier);
}

// checks if Element is a container
bool ConfigManager::isElementContainer(std::string UniqueIdentifier)
{
    return elementManager->isContainer(UniqueIdentifier);
}

// returns the container with name UniqueIdentifier
Container* ConfigManager::getContainerByIdentifier(std::string UniqueIdentifier)
{
    return elementManager->getContainerByIdentifier(UniqueIdentifier);
}

// returns the widget with name UniqueIdentifier
Widget* ConfigManager::getWidgetByIdentifier(std::string UniqueIdentifier)
{
    return elementManager->getWidgetByIdentifier(UniqueIdentifier);
}

//***************************************************************************
// PositionManager
// adds position/element to PositionManager
void ConfigManager::addPosToPosList(std::string UniqueIdentifier, std::pair<int,int> pos, std::string ParentUniqueIdentifier, bool autoassigned)
{
    positionManager->addPosToPosList(UniqueIdentifier, pos, ParentUniqueIdentifier, autoassigned);
}

void ConfigManager::addPosToPosList(std::string UniqueIdentifier, std::pair<int,int> pos, mui::AttributesEnum ParentUniqueIdentifier, bool autoassigned)
{
    positionManager->addPosToPosList(UniqueIdentifier, pos, mui::getKeywordAttribute(ParentUniqueIdentifier), autoassigned);
}

std::pair<int,int> ConfigManager::getFreePosExceptOfPos(std::vector<std::pair<int,int> > exceptPos, std::string ParentUniqueIdentifier)
{
    return positionManager->getFreePosExeptOfPos(exceptPos, ParentUniqueIdentifier);
}

// get position of element
std::pair <int,int> ConfigManager::getPosOfElement(std::string UniqueIdentifier)
{
    return positionManager->getPosOfElement(UniqueIdentifier);
}

// get next free position
std::pair <int,int> ConfigManager::getFreePos(std::string UniqueIdentifierParent)
{
    return positionManager->getFreePos(UniqueIdentifierParent);
}

// delete position/element from PositionManager
void ConfigManager::deletePosFromPosList(std::string UniqueIdentifier)
{
    positionManager->deletePosFromPosList(UniqueIdentifier);
}

// changes position of element in PositionManager
void ConfigManager::changePos(std::string UniqueIdentifier, std::pair<int,int> Pos)
{
    positionManager->changePosInPosList(UniqueIdentifier, Pos);
}

// returns true, if position is occupied, false if position is free
bool ConfigManager::isPosOccupied(std::pair<int,int> pos, std::string UniqueIdentifierParent)
{
    return positionManager->isPosOccupied(pos, UniqueIdentifierParent);
}

// returns true, if position is occupied by autoassigned element
bool ConfigManager::isPosAutoassigned(std::pair<int,int> pos, std::string UniqueIdentifierParent)
{
    return positionManager->isAutoassigned(pos, UniqueIdentifierParent);
}

// returns UniqueIdentifier of element
std::string ConfigManager::getIdentifierByPos(std::pair<int,int> pos, std::string UniqueIdentifierParent)
{
    return positionManager->getIdentifierByPos(pos, UniqueIdentifierParent);
}

// prepares position for new element: sets autoassigned element to new position, if occupied by autoassigned element
void ConfigManager::preparePos(std::pair<int,int> pos, std::string ParentUniqueIdentifier)
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
            std::cerr << "ERROR: ConfigManager::preparePos(): Position (" << pos.first << "," << pos.second << ") in Parent " << ParentUniqueIdentifier << " is occupied by " << getIdentifierByPos(pos,ParentUniqueIdentifier) << " and will be deleted and overwritten." << std::endl;
            std::string Identifier=getIdentifierByPos(pos,ParentUniqueIdentifier);          // UniqueIdentifier of
            deletePosFromPosList(Identifier);
        }
    }
}

void ConfigManager::setAutoassignedPos(std::pair<int,int> Pos, std::string ElementIdentifier, std::string ParentUniqueIdentifier)
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
        std::cerr << "ERROR: ConfigManager::setAutoassignedPos(): " << ElementIdentifier << " is no Widget or Container" << std::endl;
    }
    addPosToPosList(ElementIdentifier, Pos, ParentUniqueIdentifier, true);
}

std::string ConfigManager::getPos2Print()
{
    return positionManager->printPos();
}

