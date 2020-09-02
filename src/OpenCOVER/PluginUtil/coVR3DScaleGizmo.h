
#ifndef _CO_VR_3D_SCALE_GIZMO
#define _CO_VR_3D_SCALE_GIZMO
#include <cover/coVRIntersectionInteractor.h>
#include <vrbclient/SharedStateSerializer.h>
#include <cover/MatrixSerializer.h>
#include <net/tokenbuffer.h>
#include <PluginUtil/coLine.h>


namespace opencover
{
class coVR3DGizmo;
class PLUGIN_UTILEXPORT coVR3DScaleGizmo : public coVRIntersectionInteractor
{
private:
    bool _scaleXonly{false}, _scaleYonly{false}, _scaleZonly{false}, _scaleAll{false};
    osg::Matrix _interMat_o, _oldHandMat;
    osg::Matrix _invOldHandMat_o;
    osg::Matrix _oldInteractorXformMat_o;
    osg::Vec3 _startInterPos, _startPointerDirection_o;
    osg::Matrix  _startxAxisMatrix,_startyAxisMatrix, _startzAxisMatrix;
    float _distance{0.0f};
    osg::Vec3 _diff;

    osg::ref_ptr<osg::MatrixTransform> axisTransform;       // all the Geometry
    osg::ref_ptr<osg::MatrixTransform> xAxisTransform;
    osg::ref_ptr<osg::MatrixTransform> yAxisTransform;
    osg::ref_ptr<osg::MatrixTransform> zAxisTransform;
    osg::ref_ptr<osg::MatrixTransform> xSphereTransform;
    osg::ref_ptr<osg::MatrixTransform> ySphereTransform;
    osg::ref_ptr<osg::MatrixTransform> zSphereTransform;

    osg::ref_ptr<osg::MatrixTransform> tempxAxisTransform;  
    osg::ref_ptr<osg::MatrixTransform> tempyAxisTransform;  
    osg::ref_ptr<osg::MatrixTransform> tempzAxisTransform;  

    osg::ref_ptr<osg::Geode> sphereGeode;
    osg::ref_ptr<osg::Geode> scaleXaxisGeode;
    osg::ref_ptr<osg::Geode> scaleYaxisGeode;
    osg::ref_ptr<osg::Geode> scaleZaxisGeode;
    osg::ref_ptr<osg::Geode> scaleXSphereGeode;
    osg::ref_ptr<osg::Geode> scaleYSphereGeode;
    osg::ref_ptr<osg::Geode> scaleZSphereGeode;

    std::unique_ptr<opencover::coLine> _line;   


    osg::Vec3 calculatePointOfShortestDistance(const osg::Vec3& lp0, const osg::Vec3& lp1, osg::Vec3 axis = osg::Vec3(0,0,0)) const;

    
protected:
    virtual void createGeometry() override;
    void updateSharedState() override;
    typedef vrb::SharedState<osg::Matrix> SharedMatrix;


public:
    coVR3DScaleGizmo(osg::Matrix m, float s, coInteraction::InteractionType type, const char *iconName, const char *interactorName, coInteraction::InteractionPriority priority,coVR3DGizmo* gizmoPointer = nullptr);

    // delete scene graph
    virtual ~coVR3DScaleGizmo();

    // start the interaction (grab pointer, set selected hl, store dcsmat)
    virtual void startInteraction() override;
    virtual void doInteraction() override;
    virtual void stopInteraction() override;

    virtual void updateTransform(osg::Matrix m);

    const osg::Matrix &getMatrix() const
    {
        return _interMat_o;
    }
    void setShared(bool state) override;

};


}
#endif