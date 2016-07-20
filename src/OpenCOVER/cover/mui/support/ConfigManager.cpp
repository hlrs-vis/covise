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

bool ConfigManager::configFileExists()
{
    return fileExists;
}

//constructor:
ConfigManager::ConfigManager()
{
    fileExists = false;
    elementManager.reset(new ElementManager());
    positionManager.reset(new PositionManager());
    configFile=covise::coCoviseConfig::getEntry("value", "COVER.UiConfig", "userinterface.xml");

    if (configFile.empty())
    {
        configFile = "userinterface.xml";
    }
    if (configFile[0] != '/')
    {
        char* covisedir = getenv("COVISEDIR");
        if (covisedir != NULL)
        {
            configFile= std::string(covisedir) + "/config/" + configFile;
        }
    }

    FILE *file=fopen(configFile.c_str(), "r");

    if (file != NULL)
    {
        parser.reset(new ConfigParser(configFile));
        fileExists = true;
    }
}

//destructor:
ConfigManager::~ConfigManager()
{
}


// delete instance of ConfigManager
void ConfigManager::removeInstance()
{
}

// returns Label from ConfigFile if defined in ConfigFile; otherweise returns Label from input
std::string ConfigManager::getCorrectLabel(std::string label, mui::UITypeEnum UI, mui::DeviceTypesEnum device, std::string uniqueIdentifier)
{
   if (configFileExists())                                                            // check if configuration file exists
   {
        std::pair<std::string, bool> parsedLabel=parser->getLabel(UI,device,uniqueIdentifier);
        if (parsedLabel.second)                                                        // label exists
        {
            return parsedLabel.first;
        }
    }
    return label;
}

// returns Visible-Value from ConfigFile if defined in ConfigFile; otherwise returns Visible-Value from input
bool ConfigManager::getCorrectVisible(bool visible, mui::UITypeEnum UI, mui::DeviceTypesEnum device, std::string uniqueIdentifier)
{
    if (configFileExists())                                                            // configuration file existiert
    {
        std::pair<bool,bool> isVisible = parser->getIsVisible(UI, device, uniqueIdentifier);
        if (isVisible.second)
        {
            return isVisible.first;
        }
    }
    return visible;
}

// returns Parent from ConfigFile if defined; otherwise returns Parent from input
mui::Container* ConfigManager::getCorrectParent(mui::Container* parent, mui::UITypeEnum UI, mui::DeviceTypesEnum device, std::string uniqueIdentifier)
{
    if (configFileExists())
    {
        std::pair<std::string, bool> parentName=parser->getParent(UI, device, uniqueIdentifier);
        if (parentName.second)                     // no entry for parent in configuration file
        {
            if (elementManager->isContainer(parentName.first))
            {
                return dynamic_cast<mui::Container *>(elementManager->getElementByIdentifier(parentName.first));
            }
        }
    }
    return parent;
}

// returns Pos from ConfigFile if defined; otherwise returns Pos from input
std::pair<int,int> ConfigManager::getCorrectPos(std::pair<int,int> pos, mui::UITypeEnum UI, mui::DeviceTypesEnum device, std::string uniqueIdentifier)
{
    if (configFileExists())
    {
        std::pair<std::pair<int,int>, bool> parsedPosition = parser->getPos(UI, device, uniqueIdentifier);
        if (parsedPosition.second)
        {
            return parsedPosition.first;
        }
    }
    return pos;
}

// returns Pos from ConfigFile if defined; otherwise returns Pos from autoassign
std::pair<int,int> ConfigManager::getCorrectPos(mui::UITypeEnum UI, mui::DeviceTypesEnum device, std::string uniqueIdentifier, std::string parentUniqueIdentifier)
{
    if (configFileExists())
    {
        std::pair<std::pair<int,int>, bool> parsedPosition = parser->getPos(UI, device, uniqueIdentifier);
        if (parsedPosition.second)
        {
            return parsedPosition.first;                         // return the matchin position set in configuration file
        }
    }
    return getFreePos(parentUniqueIdentifier);
}

// returns Pos from ConfigFile if defined; otherwise returns Pos from autoassign after inputPos
std::pair<int,int> ConfigManager::getCorrectPosExceptOfPos(std::vector<std::pair<int,int> > exceptPos, mui::UITypeEnum UI, mui::DeviceTypesEnum device, std::string uniqueIdentifier, std::string parentUniqueIdentifier)
{
    if (configFileExists())
    {
        std::pair<std::pair<int,int>, bool> parsedPosition = parser->getPos(UI, device, uniqueIdentifier);
        if (parsedPosition.second)
        {
            return parsedPosition.first;
        }
    }
    else if (!configFileExists())
    {

    }
    return getFreePosExceptOfPos(exceptPos, parentUniqueIdentifier);
}

bool ConfigManager::existAttributeInConfigFile(UITypeEnum UI, DeviceTypesEnum device, std::string uniqueIdentifier, AttributesEnum attribute)
{
    if (configFileExists())
    {
        return parser->getAttributeValue(UI, device, uniqueIdentifier, attribute).second;
    }
    return false;
}

// returns the adress of the configuration file
const std::string ConfigManager::getConfigFile()
{
    return configFile;
}

//****************************************************************************************************
// ElementManager
// prints names of all elements
void ConfigManager::printElementNames()
{
    elementManager->printNames();
}

// add Elements to ElementList
void ConfigManager::addElement(std::string uniqueIdentifier, mui::Element* parent)
{
    elementManager->addElement(uniqueIdentifier, parent);
}

// delete Element from ElementList
void ConfigManager::removeElement(std::string uniqueIdentifier)
{
    elementManager->removeElement(uniqueIdentifier);
}

// checks if Element is a container
bool ConfigManager::isElementContainer(std::string uniqueIdentifier)
{
    return elementManager->isContainer(uniqueIdentifier);
}

// returns the container with name UniqueIdentifier
mui::Element* ConfigManager::getElementByIdentifier(std::string uniqueIdentifier)
{
    return elementManager->getElementByIdentifier(uniqueIdentifier);
}

//***************************************************************************
// PositionManager
// adds position/element to PositionManager
void ConfigManager::addPosToPosList(std::string uniqueIdentifier, std::pair<int,int> pos, std::string parentUniqueIdentifier, bool autoassigned)
{
    positionManager->addPosToPosList(uniqueIdentifier, pos, parentUniqueIdentifier, autoassigned);
}

std::pair<int,int> ConfigManager::getFreePosExceptOfPos(std::vector<std::pair<int,int> > exceptPos, std::string parentUniqueIdentifier)
{
    return positionManager->getFreePosExeptOfPos(exceptPos, parentUniqueIdentifier);
}

// get position of element
std::pair <int,int> ConfigManager::getPosOfElement(std::string uniqueIdentifier)
{
    return positionManager->getPosOfElement(uniqueIdentifier);
}

// get next free position
std::pair <int,int> ConfigManager::getFreePos(std::string uniqueIdentifierParent)
{
    return positionManager->getFreePos(uniqueIdentifierParent);
}

// delete position/element from PositionManager
void ConfigManager::deletePosFromPosList(std::string uniqueIdentifier)
{
    positionManager->deletePosFromPosList(uniqueIdentifier);
}

// changes position of element in PositionManager
void ConfigManager::changePos(std::string uniqueIdentifier, std::pair<int,int> pos)
{
    positionManager->changePosInPosList(uniqueIdentifier, pos);
}

// returns true, if position is occupied, false if position is free
bool ConfigManager::isPosOccupied(std::pair<int,int> pos, std::string uniqueIdentifierParent)
{
    return positionManager->isPosOccupied(pos, uniqueIdentifierParent);
}

// returns true, if position is occupied by autoassigned element
bool ConfigManager::isPosAutoassigned(std::pair<int,int> pos, std::string uniqueIdentifierParent)
{
    return positionManager->isAutoassigned(pos, uniqueIdentifierParent);
}

// returns UniqueIdentifier of element
std::string ConfigManager::getIdentifierByPos(std::pair<int,int> pos, std::string uniqueIdentifierParent)
{
    return positionManager->getIdentifierByPos(pos, uniqueIdentifierParent);
}

// prepares position for new element: sets autoassigned element to new position, if occupied by autoassigned element
void ConfigManager::preparePos(std::pair<int,int> pos, std::string parentUniqueIdentifier)
{
    if (isPosOccupied(pos, parentUniqueIdentifier))                                         // Position is occupied
    {
        if (isPosAutoassigned(pos, parentUniqueIdentifier))
        {
            std::string uniqueIdentifier=getIdentifierByPos(pos,parentUniqueIdentifier);          // UniqueIdentifier of
            std::pair <int,int> Pos = getFreePos(parentUniqueIdentifier);                   // next free Position
            setAutoassignedPos(Pos, uniqueIdentifier, parentUniqueIdentifier);
        }
        else if (!isPosAutoassigned(pos, parentUniqueIdentifier))
        {
            std::cerr << "ERROR: ConfigManager::preparePos(): Position (" << pos.first << "," << pos.second << ") in Parent " << parentUniqueIdentifier << " is occupied by " << getIdentifierByPos(pos,parentUniqueIdentifier) << " and will be deleted and overwritten." << std::endl;
            std::string uniqueIdentifier=getIdentifierByPos(pos,parentUniqueIdentifier);          // UniqueIdentifier of
            deletePosFromPosList(uniqueIdentifier);
        }
    }
}

void ConfigManager::setAutoassignedPos(std::pair<int,int> pos, std::string elementIdentifier, std::string parentUniqueIdentifier)
{
    deletePosFromPosList(elementIdentifier);
    if (!getElementByIdentifier(elementIdentifier))
    {
        std::cerr << "ERROR: ConfigManager::setAutoassignedPos(): " << elementIdentifier << " does not exist" << std::endl;
    }
    else if (isElementContainer(elementIdentifier))
    {
        getElementByIdentifier(elementIdentifier)->getTUI()->setPos(pos.first,pos.second);         // Element neu positionieren
    }
    else if(!isElementContainer(elementIdentifier))
    {
        getElementByIdentifier(elementIdentifier)->getTUI()->setPos(pos.first,pos.second);            // Element neu positionieren
    }
    else
    {
        std::cerr << "ERROR: ConfigManager::setAutoassignedPos(): " << elementIdentifier << " is no Widget or Container" << std::endl;
    }
    addPosToPosList(elementIdentifier, pos, parentUniqueIdentifier, true);
}

std::string ConfigManager::getPos2Print()
{
    return positionManager->printPos();
}

