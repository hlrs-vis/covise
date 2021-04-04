/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _CO_VR_3D_GIZMO
#define _CO_VR_3D_GIZMO

#include <memory>
#include <util/coExport.h>
#include <PluginUtil/coVR3DGizmoType.h>

namespace opencover
{
class coVR3DGizmoType;

class PLUGIN_UTILEXPORT coVR3DGizmo
{
public:
    enum class GIZMO_TYPE{ROTATE, TRANSLATE, SCALE};
    coVR3DGizmo(GIZMO_TYPE gizmptype, bool translate, bool rotate, bool scale, osg::Matrix m, float s, vrui::coInteraction::InteractionType type, const char *iconName, const char *interactorName, vrui::coInteraction::InteractionPriority priority);
    ~coVR3DGizmo();

    GIZMO_TYPE getType(){return _type;}
    void changeGizmoType();
    void setGizmoTypes(bool translate, bool rotate, bool scale);

private:
    GIZMO_TYPE _type; // current type
    std::unique_ptr<coVR3DGizmoType> _gizmo;
    bool _translate;
    bool _rotate;
    bool _scale;

public:     //functions which are forewarded to gizmotype
    void preFrame(){_gizmo->preFrame();} 
    void startInteraction(){_gizmo->startInteraction();}
    void stopInteraction(){_gizmo->stopInteraction();}
    void doInteraction(){_gizmo->doInteraction();}
    bool wasStarted(){return _gizmo->wasStarted();}
    bool wasStopped(){return _gizmo->wasStopped();}
    bool isRunning(){return _gizmo->isRunning();}
    void resetState(){_gizmo->resetState();}
    vrui::coInteraction::InteractionState getState(){return _gizmo->getState();}
    bool wasHit(){return _gizmo->wasHit();}
    int isIntersected(){return _gizmo->isIntersected();}
    void show(){_gizmo->show();}
    void hide(){_gizmo->hide();}
    void enableIntersection(){_gizmo->enableIntersection();}
    void disableIntersection(){_gizmo->disableIntersection();}
    bool isInitializedThroughSharedState(){return _gizmo->isInitializedThroughSharedState();}
    void updateTransform(osg::Matrix m){_gizmo->updateTransform(m);}
    osg::Vec3 getHitPos(){return _gizmo->getHitPos();}
    //const osg::Matrix &getMatrix(){return _gizmo->getMatrix();} const
    osg::Matrix getMatrix(){return _gizmo -> getMatrix();}
    osg::Matrix getMoveMatrix_o()const{return _gizmo -> getMoveMatrix_o();}
    osg::Matrix getMoveMatrix_w()const{return _gizmo -> getMoveMatrix_w();}

};
}
#endif