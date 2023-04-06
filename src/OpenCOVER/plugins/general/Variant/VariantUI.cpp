/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "VariantUI.h"
#include "VariantPlugin.h"
#include "Variant.h"
#include <net/tokenbuffer.h>
#include <PluginUtil/PluginMessageTypes.h>

using namespace opencover;

VariantUI::VariantUI(std::string varName, ui::Menu *Variant_menu, coTUITab *VariantPluginTab)
{
    //new ui::Menu *variantMenu

    //Creating the Checkbox for show/hide the variant in the Cover VR-menue
    Cb_item = new ui::Button(Variant_menu, ui::Owner::makeName(varName));
    Cb_item->setState(true);
    Cb_item->setCallback([this, varName](bool state){
        TokenBuffer tb;
        tb << varName;
        if (state)
            cover->sendMessage(VariantPlugin::plugin, coVRPluginSupport::TO_ALL, PluginMessageTypes::VariantShow, tb.getData().length(), tb.getData().data());
        else
            cover->sendMessage(VariantPlugin::plugin, coVRPluginSupport::TO_ALL, PluginMessageTypes::VariantHide, tb.getData().length(), tb.getData().data());
    });
    //Cb_item->setMenuListener(Variant::variantClass);
    //Variant_menu->add(Cb_item);

    //Creating the Checkbox for show/hide the variant in the TabletUI ("Variant"-Tab)
    VariantPluginTUIItem = new coTUIToggleButton(varName.c_str(), VariantPluginTab->getID()); //new Button for tabletUi
    VariantPluginTUIItem->setEventListener(Variant::variantClass);
    VariantPluginTUIItem->setState(true);
    //Creating the Radiobutton for show/hide the variant in the TabletUI ("Variant"-Tab)
    VariantRadioButton = new coTUIToggleButton("X", VariantPluginTab->getID()); //new Button for tabletUi
    VariantRadioButton->setEventListener(Variant::variantClass);
    VariantRadioButton->setState(true);

    xTRansItem = new coTUIEditFloatField("X_Trans", VariantPluginTab->getID(), 0);
    xTRansItem->setEventListener(Variant::variantClass);
    xTRansItem->setValue(0);
    yTRansItem = new coTUIEditFloatField("Y_Trans", VariantPluginTab->getID(), 0);
    yTRansItem->setEventListener(Variant::variantClass);
    yTRansItem->setValue(0);
    zTRansItem = new coTUIEditFloatField("Z_Trans", VariantPluginTab->getID(), 0);
    zTRansItem->setEventListener(Variant::variantClass);
    zTRansItem->setValue(0);
}

VariantUI::~VariantUI()
{
    delete Cb_item;
    delete VariantPluginTUIItem;
    delete xTRansItem;
    delete yTRansItem;
    delete zTRansItem;
}

ui::Button *VariantUI::getVRUI_Item()
{
    return Cb_item;
}

coTUIToggleButton *VariantUI::getTUI_Item()
{
    return VariantPluginTUIItem;
}

coTUIToggleButton *VariantUI::getRadioTUI_Item()
{
    return VariantRadioButton;
}

void VariantUI::setPosTUIItems(int pos)
{
    VariantRadioButton->setPos(0, pos);
    VariantPluginTUIItem->setPos(1, pos);
    xTRansItem->setPos(2, pos);
    yTRansItem->setPos(3, pos);
    zTRansItem->setPos(4, pos);
}

osg::Vec3d VariantUI::getTransVec()
{
    osg::Vec3d vec;
    vec.set(xTRansItem->getValue(), yTRansItem->getValue(), zTRansItem->getValue());
    return vec;
}

void VariantUI::setTransVec(osg::Vec3d vec)
{
    xTRansItem->setValue(vec.x());
    yTRansItem->setValue(vec.y());
    zTRansItem->setValue(vec.z());
}

coTUIEditFloatField *VariantUI::getXTransItem()
{
    return xTRansItem;
}

coTUIEditFloatField *VariantUI::getYTransItem()
{
    return yTRansItem;
}

coTUIEditFloatField *VariantUI::getZTransItem()
{
    return zTRansItem;
}
