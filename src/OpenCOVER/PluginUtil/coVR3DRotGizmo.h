#ifndef _CO_VR_3D_ROT_GIZMO
#define _CO_VR_3D_ROT_GIZMO

#include <PluginUtil/coVR3DGizmoType.h>
#include <OpenVRUI/osg/mathUtils.h>


namespace opencover
{

class PLUGIN_UTILEXPORT coVR3DRotGizmo : public coVR3DGizmoType
{
private:
    bool _rotateXonly{false}, _rotateYonly{false}, _rotateZonly{false};
    const float _radius{3};
    osg::Matrix _interMat_o, _oldHandMat; // löschen
    osg::Matrix _invOldHandMat_o;         //löschen
    osg::Vec3 _closestStartPoint_o;
    osg::Matrix _startHandMat;
    osg::Matrix _currentHandMat;
    
    enum class RotationAxis { Z = 0, X, Y };

    osg::ref_ptr<osg::MatrixTransform> _axisTransform; // all the Geometry
    osg::ref_ptr<osg::Geode> _sphereGeode;
    osg::ref_ptr<osg::Group> _xRotCylGroup;
    osg::ref_ptr<osg::Group> _zRotCylGroup;
    osg::ref_ptr<osg::Group> _yRotCylGroup;

    osg::Geode* circles( RotationAxis axis, int approx, osg::Vec4 color );                                    // draw circles with osg::DrawArrays (not intersectable)
    osg::Group* circlesFromCylinders( RotationAxis axis, int approx, osg::Vec4 color, float cylinderLength ); // draw circles with cylinders
    osg::Vec3Array* circleVerts(RotationAxis axis, int approx);
    
    // osg::Vec3 calcPlaneLineIntersection(const osg::Vec3& lp0, const osg::Vec3& lp1, osg::Vec3 axis) const;
                                               // calc verts for circles

    osg::Matrix calcRotation2D(const osg::Vec3& lp0_o, const osg::Vec3& lp1_o, osg::Vec3 rotationAxis); 
    osg::Matrix calcRotation3D(osg::Vec3 rotationAxis);

    bool rotateAroundSpecificAxis(osg::Group *group)const;

    float closestDistanceLineCircle(const osg::Vec3& lp0, const osg::Vec3& lp1,osg::Vec3 rotationAxis, osg::Vec3& closestPoint) const;
    
    //function that takes two input vectors v1 and v2, and a vector n that is not in the plane of v1 & v2. Here n is used to determine the "direction" of the angle between v1 and v2 in a right-hand-rule sense. I.e., cross(n,v1) would point in the "positive" direction of the angle starting from v1
    double vecAngle360(const osg::Vec3 vec1, const osg::Vec3 &vec2, const osg::Vec3& refVec);
protected:
    void createGeometry() override;
    

public:
    coVR3DRotGizmo(osg::Matrix m, float s, coInteraction::InteractionType type, const char *iconName, const char *interactorName, coInteraction::InteractionPriority priority);
    
    // delete scene graph
    virtual ~coVR3DRotGizmo();

    // start the interaction (grab pointer, set selected hl, store dcsmat)
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
