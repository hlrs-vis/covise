/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _CO_VR_3D_GIZMO
#define _CO_VR_3D_GIZMO

#include <memory>
#include <util/coExport.h>
#include <cover/coVRIntersectionInteractor.h>

namespace opencover
{
class coVR3DGizmoType;

class PLUGIN_UTILEXPORT coVR3DGizmo
{
public:
    enum class GIZMO_TYPE{ROTATE, TRANSLATE, SCALE};

    coVR3DGizmo(GIZMO_TYPE gizmptype ,osg::Matrix m, float s, vrui::coInteraction::InteractionType type, const char *iconName, const char *interactorName, vrui::coInteraction::InteractionPriority priority);
    ~coVR3DGizmo();

    GIZMO_TYPE getType(){return _type;}

    void preFrame(); 
    void changeGizmoType();

private:
    GIZMO_TYPE _type;
    std::unique_ptr<coVR3DGizmoType> _gizmo;

};




}
#endif