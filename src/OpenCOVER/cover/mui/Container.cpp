#include "Container.h"
#include <cover/coTabletUI.h>
#include <util/coTabletUIMessages.h>
#include <net/covise_connect.h>
#include <net/covise_host.h>
#include <net/message.h>
#include <net/message_types.h>

using namespace mui;

// constructor
Container::Container()
{
    menuItem = new vrui::coRowMenu("menuItem", 0, 0, false);
}

// destructor
Container::~Container()
{
}

// get ID
int Container::getTUIID()
{
    return ID;
}

// get Pointer to VR-Parent
vrui::coMenu* Container::getVRUI()
{
    return menuItem;
}

bool Container::existVRUI()
{
    return false;
}

bool Container::existTUI()
{
    return false;
}

// needs to be overwritten by inherited class
void Container::setPos(int posx, int posy)
{
    std::cerr << "ERROR: Container::setPos(int, int): Was called and should have been overwritten by inherited class." << std::endl;
}
