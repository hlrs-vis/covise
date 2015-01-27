/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "SurfaceInteraction.h"
#include <cover/coInteractor.h>
#include <OpenVRUI/coSliderMenuItem.h>
#include <OpenVRUI/coRowMenu.h>
#include "SurfacePlugin.h"

#include <OpenVRUI/coSubMenuItem.h>
#include <OpenVRUI/coButtonMenuItem.h>
#include "coIntersecCheckboxMenuItem.h"
#include "ColorBar.h"
#include <cover/coVRPluginSupport.h>
#include <OpenVRUI/coMenu.h>
#include <cover/RenderObject.h>

#include <config/CoviseConfig.h>

SurfaceInteraction::SurfaceInteraction(coInteractor *inter,
                                       SurfacePlugin *containerPlugin,
                                       int scale, string /*module*/)
    : ModuleFeedbackManager(NULL, inter)
    , _containerPlugin(containerPlugin)
    , _lengthScale(NULL)
    , _SCALE(scale)
    , _Execute(NULL)
{

    // copy and execute module
    const char *modName = _inter->getModuleName();
    int len = strlen(modName);
    if (len >= 4 && string("Comp") == modName + len - 4)
    {
        _copyAndExecute = new coButtonMenuItem("New");
        _copyAndExecute->setMenuListener(this);
        _deleteModule = new coButtonMenuItem("Delete");
        _deleteModule->setMenuListener(this);

        //_menu->add(_copyAndExecute); // to _commonItems
    }
    else
    {
        _copyAndExecute = NULL;
        _deleteModule = NULL;
    }

    if (!coCoviseConfig::isOn("COVERConfig.ExecuteOnChange", true))
    {
        _Execute = new coButtonMenuItem("Execute");
        _Execute->setMenuListener(this);
    }

    // hide-show geometry
    _hideGeometry = new coIntersecCheckboxMenuItem("Hide", 0);
    _hideGeometry->setMenuListener(this);
    // _menu->add(_hideGeometry); // to _commonItems
    _commonItems.push_back(_hideGeometry);
    _commonItems.push_back(_copyAndExecute);
    _commonItems.push_back(_deleteModule);
    _commonItems.push_back(_Execute);

    _menu->setMenuListener(containerPlugin);
}

SurfaceInteraction::~SurfaceInteraction()
{
    delete _lengthScale;
    delete _copyAndExecute;
    delete _Execute;
    delete _hideGeometry;
}

bool
SurfaceInteraction::MappingVectorField(coInteractor *inter) const
{
    RenderObject *obj = inter->getObject();
    vector<string> types;
    types.push_back("LINES");
    if (obj == NULL || obj->IsTypeField(types, false))
    {
        return true;
    }
    return false;
}

void
SurfaceInteraction::update(RenderObject *do_geom, coInteractor *inter)
{
    ModuleFeedbackManager::update(do_geom, inter);

    if (_hideGeometry->getState())
    {
        _hideGeometry->setState(0);
    }

    if (MappingVectorField(inter))
    {
        if (_lengthScale == NULL)
        {
            float minLength, maxLength, length;
            inter->getFloatSliderParam(_SCALE, minLength, maxLength, length);

            _lengthScale = new coSliderMenuItem(_inter->getParaName(_SCALE),
                                                minLength, maxLength, length);
            _lengthScale->setMenuListener(this);
            _menu->add(_lengthScale);
        }
    }
    else if (_lengthScale)
    {
        _menu->remove(_lengthScale);
        delete _lengthScale;
        _lengthScale = NULL;
    }
}

void
SurfaceInteraction::menuEvent(coMenuItem *menuItem)
{
    if (menuItem == _copyAndExecute)
    {
        _inter->copyModuleExec();
    }
    else if (menuItem == _deleteModule)
    {
        _inter->deleteModule();
    }
    else if (_Execute && menuItem == _Execute)
    {
        // bring movable icon to fixed one and remove fixed icon if necessary
        preExecCB(_inter);
        _inter->executeModule();
        _containerPlugin->RemoveFixedIcon();
        AdditionalRemoveOnExecute();
    }
    else if (menuItem == _hideGeometry)
    {
        // toggle setTravMask in the pertinent node(s)
        // .... search through container
        _containerPlugin->ToggleVisibility(_inter->getObjName());
    }
    else if (_lengthScale && menuItem == _lengthScale)
    {
        if (cover->getPointerButton()->wasReleased())
        {
            float min, max, val;
            _inter->getFloatSliderParam(_SCALE, min, max, val);
            _inter->setSliderParam(_inter->getParaName(_SCALE),
                                   min, max, _lengthScale->getValue());
            preExecCB(_inter);
            _inter->executeModule();
        }
    }
}

void
SurfaceInteraction::AdditionalRemoveOnExecute()
{
}
