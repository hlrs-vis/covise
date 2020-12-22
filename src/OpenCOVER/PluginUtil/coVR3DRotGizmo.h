#ifndef _CO_VR_3D_ROT_GIZMO
#define _CO_VR_3D_ROT_GIZMO

#include <PluginUtil/coVR3DGizmoType.h>
#include <OpenVRUI/osg/mathUtils.h>

/* ToDo;
    - implement snapping for 1 axis only
    - other interactors have: forbid translation in y-direction if traverseInteractors is on --> why do we need this ????
    - doInteraction: if (coVRNavigationManager::instance()->getMode() == coVRNavigationManager::TraverseInteractors) --> for what ?
    - for what is shared state necessary ?
*/

namespace opencover
{

class PLUGIN_UTILEXPORT coVR3DRotGizmo : public coVR3DGizmoType
{
private:
    bool _rotateXonly{false}, _rotateYonly{false}, _rotateZonly{false};
    bool _wristRotation{false};
    const float _radius{3};
    osg::Vec3 _startPointOnCircle;     // point on circle which was selected when interaction started
    
    enum class RotationAxis { Z = 0, X, Y };

    osg::ref_ptr<osg::MatrixTransform> _axisTransform; // all the Geometry
    osg::ref_ptr<osg::Geode> _sphereGeode;
    osg::ref_ptr<osg::Group> _xRotCylGroup;
    osg::ref_ptr<osg::Group> _zRotCylGroup;
    osg::ref_ptr<osg::Group> _yRotCylGroup;

    // draw circles with osg::DrawArrays (not intersectable)
    // osg::Geode* circles( RotationAxis axis, int approx, osg::Vec4 color );
    // draw circles with cylinders                                    
    osg::Group* circlesFromCylinders( RotationAxis axis, int approx, osg::Vec4 color, float cylinderLength ); 
    // calculate the verts for a circle
    osg::Vec3Array* circleVerts(RotationAxis axis, int approx);
    

    // calculate the rotation for a 2D input device
    osg::Matrix calcRotation2D(const osg::Vec3& lp0_o, const osg::Vec3& lp1_o, osg::Vec3 rotationAxis); 
    // calculate the rotation for a 3D input device (use wrist rotation)
    osg::Matrix calcRotation3D(osg::Vec3 rotationAxis);
    void snapTo45(float& angle) const;


    // get the rotation axis
    bool rotateAroundSpecificAxis(osg::Group *group)const;
    // if 3D input device is used check if wrist rotation should be used
    bool useWristRotation()const;

    
    // returns the closest distance between a point and circle which are in the same plane
    // pointOnCircle ist the corresponding point on the circle    
    float closestDistanceLineCircle(const osg::Vec3& lp0, const osg::Vec3& lp1,osg::Vec3 rotationAxis, osg::Vec3& pointOnCircle) const;
    
    // calculate the angle between two 3d vectors in the range 0-360 degree
    double vecAngle360(const osg::Vec3 vec1, const osg::Vec3 &vec2, const osg::Vec3& refVec);

protected:
    void createGeometry() override;
    

public:
    coVR3DRotGizmo(osg::Matrix m, float s, coInteraction::InteractionType type, const char *iconName, const char *interactorName, coInteraction::InteractionPriority priority,coVR3DGizmo* gizmoPointer = nullptr);
    
    virtual ~coVR3DRotGizmo();

    virtual void startInteraction() override;
    virtual void stopInteraction() override;
    virtual void doInteraction() override;


    // const osg::Matrix &getMatrix() const
    // {
       // return _interMat_o;
        // return coVRIntersectionInteractor::getMatrix();
    // }
    // const osg::Matrix getMatrix() const
    // {
        // return coVRIntersectionInteractor::getMatrix();
// 
    // }
// 

};

}
#endif
