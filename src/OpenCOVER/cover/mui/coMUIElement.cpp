#include "coMUIElement.h"
#include <util/common.h>
#include <util/unixcompat.h>
#include <util/coTabletUIMessages.h>
#include <cover/coTabletUI.h>
#include <net/covise_connect.h>
#include <net/covise_host.h>
#include <net/message.h>
#include <net/message_types.h>
#include <config/CoviseConfig.h>
#include <cover/coVRPluginSupport.h>
#include <cover/coVRSelectionManager.h>
#include <cover/coVRMSController.h>
#include <cover/coVRCommunication.h>
#include <cover/coVRFileManager.h>
#include <cover/coVRPluginList.h>
#include <cover/coTUIFileBrowser/VRBData.h>
#include <cover/coTUIFileBrowser/LocalData.h>
#include <cover/coTUIFileBrowser/IRemoteData.h>
#include <cover/coTUIFileBrowser/NetHelp.h>
#include <cover/OpenCOVER.h>
#include <fstream>

#include "support/coMUIConfigManager.h"
#include <iostream>

using namespace opencover;
using namespace covise;

// constructor:
coMUIElement::coMUIElement(){

}

// constructor:
coMUIElement::coMUIElement(const std::string &n_str, int pID)
{
    name_str = n_str;
    label_str = n_str;
    ID = coTabletUI::instance()->getID();

}

// destructor:
coMUIElement::~coMUIElement()
{
    TokenBuffer tb;
    tb << TABLET_REMOVE;
    tb << ID;
    coTabletUI::instance()->send(tb);
}

// check, if file exists:
bool coMUIElement::fileExist (std::string File){
    ifstream f(File.c_str());
    if (f.good()){
        f.close();
        return true;
    }
    else {
        f.close();
        return false;
    }
}

// checks for Label
std::string coMUIElement::findLabel(const std::string Instanz, std::string label, std::string keywordUI, std::string keywordDevice){
    std::string Label;
    ConfigManager=coMUIConfigManager::getInstance();        // needs access to ConfigManager
    if (ConfigManager->ConfigFileExists()){            // configuration file exists and shall be used, if there is a relevant entry
        if (ConfigManager->getLabel(keywordUI, keywordDevice, Instanz).second){   // relevant entry exists
            Label=ConfigManager->getLabel(keywordUI, keywordDevice, Instanz).first;       // read label from configuration file
        }
        else{                                               // there is no relevant entry in the configuration file
            Label=label;                                    // use commitet or default-value
        }
    }
    else{                                                   // there is no relevant entry in the configuration file
        Label=label;                                        // use commitet or default-value
    }
    return Label;
}

// needs to be overwritten by inheritor
void coMUIElement::setPos(int posx, int posy){
    std::cerr << "ERROR: coMUIElement::setPos(int, int) was called and should have been overwritten by derived class" << std::endl;
}

// needs to be overwritten by inheritor
std::string coMUIElement::getUniqueIdentifier(){
    std::cerr << "ERROR: coMUIElement::getUniqueIdentifier() was called and should have been overwritten by derived class" << std::endl;
}

