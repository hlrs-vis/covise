#ifndef _CO_VR_3D_GIZMO_TYPE
#define _CO_VR_3D_GIZMO_TYPE

#include <cover/coVRIntersectionInteractor.h>
#include <vrbclient/SharedStateSerializer.h>
#include <cover/MatrixSerializer.h>
#include <net/tokenbuffer.h>

#include <PluginUtil/coPlane.h>
#include <PluginUtil/coLine.h>
namespace opencover
{

class coVR3DGizmoType : public coVRIntersectionInteractor
{
protected:

    float _distance{0};
    osg::Vec3 _diff; 
    osg::Vec4 _red{0.5, 0.2, 0.2, 1.0}, _blue{0.2, 0.2, 0.5, 1.0}, _green{0.2, 0.5, 0.2, 1.0}, _grey{0.5, 0.5, 0.5, 1};
    
    osg::Matrix _interMat_o;            // current Matrix
    osg::Matrix _oldInterMat_o;         // last Matrix
    osg::Matrix _startInterMat_o;       // Matrix when interaction was started

    std::unique_ptr<opencover::coPlane> _helperPlane; 
    std::unique_ptr<opencover::coLine> _helperLine;   

    virtual void createGeometry() = 0;
    void updateSharedState() override; // braucht man das ?
    typedef vrb::SharedState<osg::Matrix> SharedMatrix; // check if I need this

    // calculate the intersection point between the pointer direction and a plane is used to calculate the new position
    osg::Vec3 calcPlaneLineIntersection(const osg::Vec3& lp0, const osg::Vec3& lp1, osg::Vec3 axis) const;

    void calculatePointerDirection_w(osg::Vec3& lp0, osg::Vec3& lp1, osg::Vec3& pointerDirVec ) const;
    void calculatePointerDirection_o(osg::Vec3& lp0_o, osg::Vec3& lp1_o, osg::Vec3& pointerDirVec ) const;
    
public:
    coVR3DGizmoType(osg::Matrix m, float s, coInteraction::InteractionType type, const char *iconName, const char *interactorName, coInteraction::InteractionPriority priority);

    virtual ~coVR3DGizmoType();

    void startInteraction() override;
    void stopInteraction() override; 
    virtual void updateTransform(osg::Matrix m);

    const osg::Matrix &getMatrix() const
    {
        return _interMat_o;
    }

    void setShared(bool state) override;


};

}
#endif