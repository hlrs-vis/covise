#ifndef _CO_VR_3D_GIZMO_TYPE
#define _CO_VR_3D_GIZMO_TYPE

#include <cover/coVRIntersectionInteractor.h>
#include <net/tokenbuffer_serializer.h>
#include <cover/MatrixSerializer.h>
#include <net/tokenbuffer.h>

#include <PluginUtil/coPlane.h>
#include <PluginUtil/coLine.h>

#include <OpenVRUI/coAction.h>
#include <OpenVRUI/coCombinedButtonInteraction.h>
#include <OpenVRUI/sginterface/vruiHit.h>
#include <OpenVRUI/osg/OSGVruiHit.h>

namespace opencover
{

class coVR3DGizmo;
class PLUGIN_UTILEXPORT coVR3DGizmoType : public coVRIntersectionInteractor
{
private:
    bool _changeGizmoType{false};
    coVR3DGizmo* _observer{nullptr};                             // if coVR3DGizmoType is constructed from a coVR3DGizmo the Gizmotype can be changed
    std::unique_ptr<coCombinedButtonInteraction> _interactionB;  // interaction to switch GizmoType

    void changeGizmoType(); 

protected:

    float _distance{0};
    osg::Vec3 _diff; //diff between intersection point and sphere center
    osg::Vec4 _red{0.5, 0.2, 0.2, 1.0}, _blue{0.2, 0.2, 0.5, 1.0}, _green{0.2, 0.5, 0.2, 1.0}, _grey{0.5, 0.5, 0.5, 1};
    
    osg::Matrix _interMat_o;            // current Matrix
    osg::Matrix _startInterMat_w;       // Matrix when interaction was started
    osg::Matrix _startHandMat_o;
    osg::Matrix _startHandMat_w;

    std::unique_ptr<opencover::coPlane> _helperPlane; 
    std::unique_ptr<opencover::coLine> _helperLine;   

    void createGeometry() override = 0;
    void updateSharedState() override; // braucht man das ?
    typedef vrb::SharedState<osg::Matrix> SharedMatrix; // check if I need this

    // calculate the intersection point between the pointer direction and a plane is used to calculate the new position
    osg::Vec3 calcPlaneLineIntersection(const osg::Vec3& lp0, const osg::Vec3& lp1, osg::Vec3 axis) const;

    void calculatePointerDirection_w(osg::Vec3& lp0, osg::Vec3& lp1, osg::Vec3& pointerDirVec ) const;
    void calculatePointerDirection_o(osg::Vec3& lp0_o, osg::Vec3& lp1_o, osg::Vec3& pointerDirVec ) const;
    
public:
    coVR3DGizmoType(osg::Matrix m, float s, coInteraction::InteractionType type, const char *iconName, const char *interactorName, coInteraction::InteractionPriority priority ,coVR3DGizmo* gizmoPointer = nullptr);

    virtual ~coVR3DGizmoType();
    virtual void updateTransform(osg::Matrix m);

    void preFrame() override;
    void startInteraction() override;
    void stopInteraction() override; 
    int hit(vrui::vruiHit *hit) override;
    void miss() override;
    void resetState() override; // un-highlight

    void update() override;
    void setShared(bool state) override;

    virtual osg::Matrix getMoveMatrix_o() const; // returns diff Mat between start and end of movement in object coordinates
    osg::Matrix getMoveMatrix_w() const; // returns diff Mat between start and end of movement in world coordinates


};

}
#endif
