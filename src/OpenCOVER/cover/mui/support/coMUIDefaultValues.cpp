#include "coMUIDefaultValues.h"

// constructor:
coMUIDefaultValues::coMUIDefaultValues(){
    keywordTUI = "TUI";                           // keyword for TUI in configuration file
    keywordVRUI = "VRUI";                         // keyword for VRUI in configuration file
    keywordTablet = "Tablet";                     // keyword for Tablet in configuration file
    keywordCAVE = "CAVE";                         // keyword for CAVE in configuration file
    keywordPhone = "Phone";                       // keyword for Phone in configuration file
    keywordPowerwall = "Powerwall";               // keyword for Powerwall in configuration file
    keywordMainWindow = "HauptfensterMainWindow"; // keyword for MainWindow in configuration file
    visible = true;
    PositionFirstCall = true;
}

// destructor:
coMUIDefaultValues::~coMUIDefaultValues(){

}

std::string coMUIDefaultValues::getKeywordCAVE(){
    return keywordCAVE;
}

std::string coMUIDefaultValues::getKeywordTablet(){
    return keywordTablet;
}

std::string coMUIDefaultValues::getKeywordTUI(){
    return keywordTUI;
}

std::string coMUIDefaultValues::getKeywordVRUI(){
    return keywordVRUI;
}

std::string coMUIDefaultValues::getKeywordPowerwall(){
    return keywordPowerwall;
}

std::string coMUIDefaultValues::getKeywordPhone(){
    return keywordPhone;
}

std::string coMUIDefaultValues::getKeywordMainWindow(){
    return keywordMainWindow;
}
