#include "DefaultValues.h"

using namespace mui;

// constructor:
DefaultValues::DefaultValues()
{
    keywordTUI = "TUI";                           // keyword for TUI in configuration file
    keywordVRUI = "VRUI";                         // keyword for VRUI in configuration file
    keywordTablet = "Tablet";                     // keyword for Tablet in configuration file
    keywordCAVE = "CAVE";                         // keyword for CAVE in configuration file
    keywordPhone = "Phone";                       // keyword for Phone in configuration file
    keywordPowerwall = "Powerwall";               // keyword for Powerwall in configuration file
    keywordMainWindow = "HauptfensterMainWindow"; // keyword for MainWindow in configuration file
    keywordVisible = "visible";
    keywordParent = "parent";
    keywordXPosition = "posx";
    keywordYPosition = "posy";
    keywordLabel = "label";
    keywordClass = "class";

    visible = true;
    PositionFirstCall = true;
}

// destructor:
DefaultValues::~DefaultValues()
{

}

std::string DefaultValues::getKeywordCAVE()
{
    return keywordCAVE;
}

std::string DefaultValues::getKeywordTablet()
{
    return keywordTablet;
}

std::string DefaultValues::getKeywordTUI()
{
    return keywordTUI;
}

std::string DefaultValues::getKeywordVRUI()
{
    return keywordVRUI;
}

std::string DefaultValues::getKeywordPowerwall()
{
    return keywordPowerwall;
}

std::string DefaultValues::getKeywordPhone()
{
    return keywordPhone;
}

std::string DefaultValues::getKeywordMainWindow()
{
    return keywordMainWindow;
}

std::string DefaultValues::getKeywordVisible()
{
    return keywordVisible;
}

std::string DefaultValues::getKeywordParent()
{
    return keywordParent;
}

std::string DefaultValues::getKeywordXPosition()
{
    return keywordXPosition;
}

std::string DefaultValues::getKeywordYPosition()
{
    return keywordYPosition;
}

std::string DefaultValues::getKeywordLabel()
{
    return keywordLabel;
}

std::string DefaultValues::getKeywordClass()
{
    return keywordClass;
}
