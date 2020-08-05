#ifndef _CO_VR_3D_ROT_GIZMO
#define _CO_VR_3D_ROT_GIZMO
#include <cover/coVRIntersectionInteractor.h>
#include <vrbclient/SharedStateSerializer.h>
#include <cover/MatrixSerializer.h>
#include <net/tokenbuffer.h>
#include <osgSim/SphereSegment>


namespace opencover
{

class PLUGIN_UTILEXPORT coVR3DRotGizmo : public coVRIntersectionInteractor
{
private:
    bool _rotateXonly{false}, _rotateYonly{false}, _rotateZonly{false};
    const float _radius{3};
    float _distance{0.0f};
    osg::Vec3 _diff;
    osg::Matrix _interMat_o, _oldHandMat;
    osg::Matrix _invOldHandMat_o;
    osg::Matrix _oldInteractorXformMat_o;
    
    osg::ref_ptr<osg::MatrixTransform> _axisTransform; ///< all the Geometry
    osg::ref_ptr<osg::MatrixTransform> _xAxisTransform;
    osg::ref_ptr<osg::MatrixTransform> _yAxisTransform;
    osg::ref_ptr<osg::MatrixTransform> _zAxisTransform;

    osg::ref_ptr<osg::Geode> _sphereGeode;
    osg::ref_ptr<osg::Geode> _rotateXaxisGeode;
    osg::ref_ptr<osg::Geode> _rotateYaxisGeode;
    osg::ref_ptr<osg::Geode> _rotateZaxisGeode;

    osg::ref_ptr<osg::MatrixTransform> _zRotCylTransform;
    osg::ref_ptr<osg::MatrixTransform> _yRotCylTransform;
    osg::ref_ptr<osg::MatrixTransform> _xRotCylTransform;

    osg::ref_ptr<osg::Geode> _zRotCylGeode;
    osg::ref_ptr<osg::Geode> _yRotCylGeode;
    osg::ref_ptr<osg::Geode> _xRotCylGeode;

    osg::Geode* circles( int plane, int approx, osg::Vec4 color );
    osg::Vec3Array* circleVerts(int plane, int approx);
    osg::Matrix calcRotation(osg::Vec3 rotationAxis, osg::Vec3 cylinderDirectionVector);

protected:
    virtual void createGeometry() override;
    void updateSharedState() override;
    typedef vrb::SharedState<osg::Matrix> SharedMatrix;

public:
    coVR3DRotGizmo(osg::Matrix m, float s, coInteraction::InteractionType type, const char *iconName, const char *interactorName, coInteraction::InteractionPriority priority);
    
    // delete scene graph
    virtual ~coVR3DRotGizmo();

    // start the interaction (grab pointer, set selected hl, store dcsmat)
    virtual void startInteraction() override;
    virtual void doInteraction() override;
    virtual void preFrame() override;

    virtual void updateTransform(osg::Matrix m);

    //const osg::Matrix &getMatrix() const
    //{
    //    return _interMat_o;
    //}
    void setShared(bool state) override;

};

}

#endif
/*

if (_rotateZonly)
        interactorXformMat_o = calcRotation(osg::Z_AXIS, osg::Vec3(0, -1, 0));
    else if(_rotateYonly)
        interactorXformMat_o = calcRotation(osg::Y_AXIS, osg::Vec3(-1, 0, 0));
    else if(_rotateXonly)
        interactorXformMat_o = calcRotation(osg::X_AXIS, osg::Vec3(0, 0, 1));



osg::Matrix coVR3DRotGizmo::calcRotation(osg::Vec3 rotationAxis, osg::Vec3 cylinderDirectionVector)
{
    osg::Matrix interactorXformMat_o; 
    osg::Matrix w_to_o = cover->getInvBaseMat();
    osg::Matrix currHandMat = getPointerMat();
    osg::Vec3 origin{0, 0, 0};
    osg::Vec3 yaxis(0, 1, 0);
    osg::Matrix i_to_o = scaleTransform->getMatrix()*moveTransform->getMatrix();
    osg::Matrix o_to_i = osg::Matrix::inverse(i_to_o);
    osg::Vec3 hand_i = origin * currHandMat * w_to_o * o_to_i;
    osg::Vec3 pos = hand_i;
    osg::Vec3 dir = yaxis * currHandMat * w_to_o * o_to_i;
    dir -= pos;
    dir.normalize();
    // std::cerr << "pos: " << pos << ", dir: " << dir << std::endl;
    double R = _diff.length() / getScale();
    double a = dir[0]*dir[0] + dir[1]*dir[1] + dir[2]*dir[2];
    double b = 2.*(dir[0]*pos[0] + dir[1]*pos[1] + dir[2]*pos[2]);
    double c = pos[0]*pos[0] + pos[1]*pos[1] + pos[2]*pos[2] - R*R;
    double D = b*b-4*a*c;
    // std::cerr << "scale=" << getScale() << ", a=" << a << ", b=" << b << ", c=" << c << ", disc=" << D << std::endl;
    double t = -1.;
    if (D >= 0)
    {
        double t1 = 0.5*(-b-sqrt(D))/a;
        double t2 = 0.5*(-b+sqrt(D))/a;
        if (t1 < 0)
        {
            t = t2;
        }
        else if (is2D())
        {
            t = t1;
        }
        else
        {
            double old = _distance / getScale();
            if (std::abs(old-t1) < std::abs(old-t2))
                t = t1;
            else
                t = t2;
        }
        // std::cerr << "solution: t1=" << t1 << ", t2=" << t2 << ", t=" << t << std::endl;
        // // osg::Vec3 v1 = pos+dir*t1;
        // // osg::Vec3 v2 = pos+dir*t2;
        // std::cerr << "    v1: " << v1 << ", v2: " << v2 << std::endl;
    }
    if (t < 0)
    {
        t = -dir * pos;
    }
    if (t >= 0)
    {
        _distance = t * getScale();
        osg::Vec3 isect = pos+dir*t;
        // std::cerr << "valid intersection: t=" << t << ", p=" << isect << ", dist=" << isect.length() << std::endl;
        osg::Matrix rot;
        rot.makeRotate( cylinderDirectionVector, isect);
        interactorXformMat_o = rot * getMatrix();
        
        // restrict rotation to specific axis (therefor we use euler: h=zAxis, p=xAxis, r=yAxis)
        coCoord euler = interactorXformMat_o;
        coCoord Oldeuler = _oldInteractorXformMat_o;
        if(rotationAxis == osg::Z_AXIS)
        {
            euler.hpr[1] = Oldeuler.hpr[1]; 
            euler.hpr[2] = Oldeuler.hpr[2]; 
        }
        else if(rotationAxis == osg::Y_AXIS)
        {
            euler.hpr[0] = Oldeuler.hpr[0]; 
            euler.hpr[1] = Oldeuler.hpr[1];  
        }
        else if(rotationAxis == osg::X_AXIS)
        {
            euler.hpr[0] = Oldeuler.hpr[0];
            euler.hpr[2] = Oldeuler.hpr[2];
        }
        
        euler.makeMat(interactorXformMat_o);
    }
    else
    {
        //  std::cerr <<"distance = 0"<<std::endl;
        _distance = 0;
    }   

    return interactorXformMat_o;
}
*/