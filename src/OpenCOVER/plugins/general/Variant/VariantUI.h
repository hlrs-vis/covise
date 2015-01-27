/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/* 
 * File:   VariantUI.h
 * Author: hpcagott
 *
 * Created on 18. September 2009, 14:50
 */

#ifndef _VARIANTUI_H
#define _VARIANTUI_H

#include <OpenVRUI/coMenuItem.h>
#include <OpenVRUI/coCheckboxMenuItem.h>
#include <OpenVRUI/coPotiMenuItem.h>
#include <OpenVRUI/coSubMenuItem.h>
#include <OpenVRUI/coRowMenu.h>
#include <OpenVRUI/coButtonMenuItem.h>
#include <OpenVRUI/coCheckboxGroup.h>
#include <OpenVRUI/coLabelMenuItem.h>

#include <cover/coTabletUI.h>
#include <cover/coVRTui.h>

using namespace vrui;
using namespace opencover;

class VariantUI
{
public:
    VariantUI(std::string varName, coRowMenu *Variant_menu, coTUITab *VariantPluginTab);
    ~VariantUI();

    coCheckboxMenuItem *getVRUI_Item();
    coTUIToggleButton *getRadioTUI_Item();
    coTUIToggleButton *getTUI_Item();
    coTUIEditFloatField *getXTransItem();
    coTUIEditFloatField *getYTransItem();
    coTUIEditFloatField *getZTransItem();
    void setPosTUIItems(int pos);
    osg::Vec3d getTransVec();
    void setTransVec(osg::Vec3d vec);

private:
    coCheckboxMenuItem *Cb_item;
    coTUIToggleButton *VariantRadioButton;
    coTUIToggleButton *VariantPluginTUIItem;
    coTUIEditFloatField *xTRansItem;
    coTUIEditFloatField *yTRansItem;
    coTUIEditFloatField *zTRansItem;
};

#endif /* _VARIANTUI_H */
