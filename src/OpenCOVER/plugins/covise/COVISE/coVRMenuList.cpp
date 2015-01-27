/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

// by Lars Frenzel
// 28.10.1997

#include <util/common.h>
#include "coVRMenuList.h"
#include <cover/VRSceneGraph.h>
#include <cover/coVRPluginSupport.h>
#include <cover/coVRMSController.h>
#include <cover/RenderObject.h>
#include <osg/Texture>
#include <osg/StateSet>
#include <osg/Image>
#include <cover/RenderObject.h>

#include <OpenVRUI/coToolboxMenu.h>
#include <OpenVRUI/coRowMenu.h>
#include <cover/VRPinboard.h>

#include <OpenVRUI/coButtonMenuItem.h>
#include <OpenVRUI/coSubMenuItem.h>
#include <OpenVRUI/coCheckboxMenuItem.h>
#include <OpenVRUI/coPopupHandle.h>
#include <OpenVRUI/coTexturedBackground.h>

using namespace opencover;
using namespace vrui;

coVRMenuImage::coVRMenuImage(std::string name, osg::Node *n)
{
    imageName = name;
    node = n;
    nodeName = n->getName();
    osg::StateSet *geoState = node->getStateSet();
    if (geoState)
    {
        osg::Texture *tex = static_cast<osg::Texture *>(geoState->getTextureAttribute(0, osg::StateAttribute::TEXTURE));
        osg::Image *image = tex->getImage(0);
        unsigned char *buf = image->data();

        imageBackground = new coTexturedBackground((uint *)buf, NULL, NULL, image->getPixelSizeInBits() / 8, image->s(), image->t(), 0);
        imageBackground->setSize(500, 500, 0);
        imageBackground->setTexSize(500, -500);
        imageBackground->setMinWidth(500);
        imageBackground->setMinHeight(500);

        popupHandle = new coPopupHandle(name);
        popupHandle->setScale(2 * cover->getSceneSize() / 2500);
        popupHandle->setPos(-500 * cover->getSceneSize() / 2500, 0, -500 * cover->getSceneSize() / 2500);
        popupHandle->addElement(imageBackground);

        menuItem = new coCheckboxMenuItem(name, true);
        coVRMenuList::instance()->plotMenu->add(menuItem);
        menuItem->setMenuListener(coVRMenuList::instance());
    }
    show();
}

void coVRMenuImage::updateImage(osg::Node *n)
{
    node = n;
    nodeName = n->getName();
    osg::StateSet *geoState = node->getStateSet();
    if (geoState)
    {
        osg::Texture *tex = static_cast<osg::Texture *>(geoState->getTextureAttribute(0, osg::StateAttribute::TEXTURE));
        osg::Image *image = tex->getImage(0);
        unsigned char *buf = image->data();
        imageBackground->setImage((uint *)buf, NULL, NULL, image->getPixelSizeInBits() / 8, image->s(), image->t(), 0);
    }
}

void coVRMenuImage::update()
{
    popupHandle->update();
}

coVRMenuImage::~coVRMenuImage()
{
    delete imageBackground;
    delete popupHandle;
    delete menuItem;
}

void coVRMenuImage::show()
{
    if (popupHandle)
        popupHandle->setVisible(true);
}

void coVRMenuImage::hide()
{
    if (popupHandle)
        popupHandle->setVisible(false);
}

coVRMenuList::coVRMenuList()
{
    pinboardEntry = NULL;
}

coVRMenuList::~coVRMenuList()
{
    delete pinboardEntry;
}

void coVRMenuList::menuEvent(coMenuItem *menuItem)
{
    reset();
    while (current())
    {
        if (current()->menuItem == menuItem)
        {
            if (current()->menuItem->getState())
            {
                current()->show();
            }
            else
            {
                current()->hide();
            }
            break;
        }
        next();
    }
}

bool coVRMenuList::add(RenderObject *dobj, osg::Node *n)
{

    int i = 0;
    char buf[1000];
    sprintf(buf, "MENU_TEXTURE");
    bool foundAttrib = false;
    while (const char *attrib = dobj->getAttribute(buf))
    {
        foundAttrib = true;
        std::string attr = attrib;
        coVRMenuImage *menuImage = find(attr);
        if (menuImage)
            menuImage->updateImage(n);
        else
        {

            if (pinboardEntry == NULL)
            {
                pinboardEntry = new coSubMenuItem("Plot");
                cover->getMenu()->add(pinboardEntry);
                plotMenu = new coRowMenu("Plots");
                pinboardEntry->setMenu(plotMenu);
            }
            menuImage = new coVRMenuImage(attrib, n);
            append(menuImage);
        }
        sprintf(buf, "MENU_IMAGE%d", i++);
    }
    if (foundAttrib)
        return false; // do not add this node to the scenegraph
    return true;
}

coVRMenuImage *coVRMenuList::find(std::string &name)
{
    reset();
    while (current())
    {
        if (current()->getName() == name)
            return (current());
        next();
    }

    return (NULL);
}

coVRMenuImage *coVRMenuList::find(osg::Node *n)
{
    reset();
    while (current())
    {
        if (current()->node == n)
            return (current());

        next();
    }

    return (NULL);
}

void coVRMenuList::removeAll(std::string nodeName)
{
    reset();
    while (current())
    {
        if (current()->getNodeName() == nodeName)
            remove();
        else
            next();
    }
}

void coVRMenuList::update()
{
    reset();
    while (current())
    {
        current()->update();
        next();
    }
}

coVRMenuList *coVRMenuList::instance()
{

    static coVRMenuList _instance;
    return &_instance;
}
