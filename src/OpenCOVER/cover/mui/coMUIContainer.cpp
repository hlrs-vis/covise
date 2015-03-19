#include "coMUIContainer.h"
#include <cover/coTabletUI.h>
#include <util/coTabletUIMessages.h>
#include <net/covise_connect.h>
#include <net/covise_host.h>
#include <net/message.h>
#include <net/message_types.h>

// constructor
coMUIContainer::coMUIContainer()
{
    menuItem = new vrui::coRowMenu("menuItem", 0, 0, false);
}

// destructor
coMUIContainer::~coMUIContainer()
{
}

// get ID
int coMUIContainer::getTUIID()
{
    return ID;
}

// get Pointer to VR-Parent
vrui::coMenu* coMUIContainer::getVRUI()
{
    return menuItem;
}

bool coMUIContainer::existVRUI()
{
    return false;
}

bool coMUIContainer::existTUI()
{
    return false;
}

// needs to be overwritten by inherited class
void coMUIContainer::setPos(int posx, int posy)
{
    std::cerr << "ERROR: coMUIContainer::setPos(int, int): Was called and should have been overwritten by inherited class." << std::endl;
}
