
#ifndef _CO_VR_3D_TRANS_GIZMO
#define _CO_VR_3D_TRANS_GIZMO

#include "coVR3DGizmoType.h"

/* ToDo;
    - if ray of pointer and axis are almost parallel and the closest point between two lines is far away then the gizmo disappears
*/
namespace opencover
{

class PLUGIN_UTILEXPORT coVR3DTransGizmo : public coVR3DGizmoType
{
private:
    const float _arrowLength{5.0f};

    bool _translateXonly{false}, _translateYonly{false}, _translateZonly{false};
    bool _translateXYonly{false}, _translateXZonly{false}, _translateYZonly{false};
    
    osg::ref_ptr<osg::MatrixTransform> axisTransform; 
    osg::ref_ptr<osg::MatrixTransform> xAxisTransform;
    osg::ref_ptr<osg::MatrixTransform> yAxisTransform;
    osg::ref_ptr<osg::MatrixTransform> zAxisTransform;
    osg::ref_ptr<osg::MatrixTransform> xConeTransform;
    osg::ref_ptr<osg::MatrixTransform> yConeTransform;
    osg::ref_ptr<osg::MatrixTransform> zConeTransform;
    osg::ref_ptr<osg::MatrixTransform> xyPlaneTransform;
    osg::ref_ptr<osg::MatrixTransform> xzPlaneTransform;
    osg::ref_ptr<osg::MatrixTransform> yzPlaneTransform;
  
    osg::ref_ptr<osg::Geode> sphereGeode;
    osg::ref_ptr<osg::Geode> translateXaxisGeode;
    osg::ref_ptr<osg::Geode> translateYaxisGeode;
    osg::ref_ptr<osg::Geode> translateZaxisGeode;
    osg::ref_ptr<osg::Geode> translateXconeGeode;
    osg::ref_ptr<osg::Geode> translateYconeGeode;
    osg::ref_ptr<osg::Geode> translateZconeGeode;
    osg::ref_ptr<osg::Geode> translateXYplaneGeode;
    osg::ref_ptr<osg::Geode> translateXZplaneGeode;
    osg::ref_ptr<osg::Geode> translateYZplaneGeode;

    // the point on Line 1(pointer direction) that is nearest to Line 2 (axis direction) is used to calculate the new position
    osg::Vec3 calculatePointOfShortestDistance(const osg::Vec3& lp0, const osg::Vec3& lp1, osg::Vec3 axis_o) const;

protected:
    virtual void createGeometry() override;

    // called every time when the geometry is intersected
    int hit(vrui::vruiHit *hit) override;

public:
    coVR3DTransGizmo(osg::Matrix m, float s, coInteraction::InteractionType type, const char *iconName, const char *interactorName, coInteraction::InteractionPriority priority, coVR3DGizmo* gizmoPointer = nullptr);
    virtual ~coVR3DTransGizmo();

    virtual void startInteraction() override;
    virtual void doInteraction() override;
    virtual void stopInteraction() override;
    virtual void resetState() override; 


};

}
#endif
