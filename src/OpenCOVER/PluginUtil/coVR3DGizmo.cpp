#include "coVR3DGizmo.h"
#include "coVR3DTransGizmo.h"
#include "coVR3DRotGizmo.h"
#include "coVR3DScaleGizmo.h"

#include <exception>
#include <cover/coVRPluginSupport.h>
using namespace opencover;


coVR3DGizmo::coVR3DGizmo(GIZMO_TYPE gizmotype,bool translate, bool rotate, bool scale, osg::Matrix m, float s, vrui::coInteraction::InteractionType type, const char *iconName, const char *interactorName, vrui::coInteraction::InteractionPriority priority)
:_translate(translate),_rotate(rotate),_scale(scale)
{
    _type = gizmotype;

    if(gizmotype == GIZMO_TYPE::ROTATE)
        _gizmo.reset(new coVR3DRotGizmo(m, s, type, iconName, interactorName, priority, this));
    else if(gizmotype == GIZMO_TYPE::TRANSLATE)
        _gizmo.reset(new coVR3DTransGizmo(m, s, type, iconName, interactorName, priority, this));
    else if(gizmotype == GIZMO_TYPE::SCALE)
         _gizmo.reset(new coVR3DScaleGizmo(m, s, type, iconName, interactorName, priority, this));

    _gizmo->enableIntersection();
    _gizmo->show();
}

coVR3DGizmo::~coVR3DGizmo()
{
}

void coVR3DGizmo::changeGizmoType()
{
    float _interSize = cover->getSceneSize() / 50 ;
    osg::Matrix matrix = _gizmo->getMatrix();

    try{
    if(dynamic_cast<coVR3DRotGizmo*>(_gizmo.get()) != nullptr)
    {
        if(_translate)
        {
            _gizmo.reset(new coVR3DTransGizmo(matrix, _interSize, vrui::coInteraction::ButtonA, "hand", "coVR3DTransGizmo", vrui::coInteraction::Medium, this));
            _type = GIZMO_TYPE::TRANSLATE;
        }
        else if(_scale)
        {
            _gizmo.reset(new coVR3DScaleGizmo(matrix, _interSize, vrui::coInteraction::ButtonA, "hand", "coVR3DScaleGizmo", vrui::coInteraction::Medium, this));
            _type = GIZMO_TYPE::SCALE;
        }
    }
    else if(dynamic_cast<coVR3DTransGizmo*>(_gizmo.get()) != nullptr)
    {
        if(_scale)
        {
            _gizmo.reset(new coVR3DScaleGizmo(matrix, _interSize, vrui::coInteraction::ButtonA, "hand", "coVR3DScaleGizmo", vrui::coInteraction::Medium, this));
            _type = GIZMO_TYPE::SCALE;
        }
        else if(_rotate)
        {
            _gizmo.reset(new coVR3DRotGizmo(matrix, _interSize, vrui::coInteraction::ButtonA, "hand", "coVR3DRotateGizmo", vrui::coInteraction::Medium, this));
            _type = GIZMO_TYPE::ROTATE;
        }
    }   
    else if(dynamic_cast<coVR3DScaleGizmo*>(_gizmo.get()) != nullptr)
    {
        if(_rotate)
        {
            _gizmo.reset(new coVR3DRotGizmo(matrix, _interSize, vrui::coInteraction::ButtonA, "hand", "coVR3DRotGizmo", vrui::coInteraction::Medium, this));
            _type = GIZMO_TYPE::ROTATE;
        }
        else if(_translate)
        {
            _gizmo.reset(new coVR3DTransGizmo(matrix, _interSize, vrui::coInteraction::ButtonA, "hand", "coVR3DTransGizmo", vrui::coInteraction::Medium, this));
            _type = GIZMO_TYPE::TRANSLATE;
        }
    }       
    }catch (std::exception& e) {std::cout << "Exception: " << e.what();}
    
    _gizmo->enableIntersection();
    _gizmo->show();
    std::cout<<_translate<<_rotate<<_scale<<std::endl;
}

void coVR3DGizmo::setGizmoTypes(bool translate, bool rotate, bool scale)
{
    _translate = translate;
    _rotate = rotate;
    _scale = scale;
}

