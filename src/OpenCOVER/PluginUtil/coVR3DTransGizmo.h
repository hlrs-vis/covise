
#ifndef _CO_VR_3D_TRANS_GIZMO
#define _CO_VR_3D_TRANS_GIZMO
#include <cover/coVRIntersectionInteractor.h>
#include <vrbclient/SharedStateSerializer.h>
#include <cover/MatrixSerializer.h>
#include <net/tokenbuffer.h>

namespace opencover
{

class PLUGIN_UTILEXPORT coVR3DTransGizmo : public coVRIntersectionInteractor
{
private:
    bool _translateXonly{false}, _translateYonly{false}, _translateZonly{false};
    bool _translateXYonly{false}, _translateXZonly{false}, _translateYZonly{false};
    osg::Matrix _interMat_o, _oldHandMat;
    osg::Matrix _invOldHandMat_o;
    osg::Matrix _oldInteractorXformMat_o;

    osg::ref_ptr<osg::MatrixTransform> axisTransform; ///< all the Geometry
    osg::ref_ptr<osg::MatrixTransform> xAxisTransform;
    osg::ref_ptr<osg::MatrixTransform> yAxisTransform;
    osg::ref_ptr<osg::MatrixTransform> zAxisTransform;
    osg::ref_ptr<osg::MatrixTransform> xyPlaneTransform;
    osg::ref_ptr<osg::MatrixTransform> xzPlaneTransform;
    osg::ref_ptr<osg::MatrixTransform> yzPlaneTransform;
  
    osg::ref_ptr<osg::Geode> sphereGeode;
    osg::ref_ptr<osg::Geode> translateXaxisGeode;
    osg::ref_ptr<osg::Geode> translateYaxisGeode;
    osg::ref_ptr<osg::Geode> translateZaxisGeode;
    osg::ref_ptr<osg::Geode> translateXYplaneGeode;
    osg::ref_ptr<osg::Geode> translateXZplaneGeode;
    osg::ref_ptr<osg::Geode> translateYZplaneGeode;

    float _distance{0.0f};
    osg::Vec3 _diff;

protected:
    virtual void createGeometry() override;
    void updateSharedState() override;
    typedef vrb::SharedState<osg::Matrix> SharedMatrix;


public:
    coVR3DTransGizmo(osg::Matrix m, float s, coInteraction::InteractionType type, const char *iconName, const char *interactorName, coInteraction::InteractionPriority priority);

    // delete scene graph
    virtual ~coVR3DTransGizmo();

    // start the interaction (grab pointer, set selected hl, store dcsmat)
    virtual void startInteraction() override;
    virtual void doInteraction() override;

    virtual void updateTransform(osg::Matrix m);

    const osg::Matrix &getMatrix() const
    {
        return _interMat_o;
    }
    void setShared(bool state) override;

};


}
#endif