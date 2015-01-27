/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _SURFACE_INTERACTION_H_
#define _SURFACE_INTERACTION_H_

#include <util/common.h>
#include "ModuleFeedbackManager.h"

class SurfacePlugin;
class coMenuItem;
class coSliderMenuItem;
class coIntersecCheckboxMenuItem;
class coButtonMenuItem;
class coSubMenuItem;
class RenderObject;

#include <OpenVRUI/coMenuItem.h>

namespace opencover
{
class PLUGIN_UTILEXPORT SurfaceInteraction : public ModuleFeedbackManager
{
public:
    SurfaceInteraction(coInteractor *, SurfacePlugin *containerPlugin, int scale,
                       string module);
    virtual ~SurfaceInteraction();
    virtual void menuEvent(coMenuItem *menuItem);

protected:
    bool MappingVectorField(coInteractor *) const;
    virtual void update(RenderObject *container, coInteractor *);

    SurfacePlugin *_containerPlugin;
    coSliderMenuItem *_lengthScale;
    const int _SCALE;
    std::vector<coMenuItem *> _commonItems;

    coButtonMenuItem *_Execute;
    virtual void AdditionalRemoveOnExecute();

private:
    coButtonMenuItem *_copyAndExecute;
    coButtonMenuItem *_deleteModule;
    coIntersecCheckboxMenuItem *_hideGeometry;
};
}
#endif
