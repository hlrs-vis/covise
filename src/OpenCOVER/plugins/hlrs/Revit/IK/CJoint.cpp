#include "CJoint.h"

void CJoint::RotateJoint( const float & rot_angle )
{
    theta += rot_angle;
}

void CJoint::TranlsateJoint( float & displacement )
{
    d+=displacement;    
}

CJoint::CJoint( float d_in , float theta_in , JointT jt_in , std::string jname)
                                                            :d(d_in),
                                                            theta(theta_in),
                                                            current_joint_type(jt_in),
                                                            joint_name(jname)
{}

void CJoint::SetGlobalPosition( const Vector3f & vec )
{
    global_joint_origin = vec;
}

OUT float CJoint::GiveMeVariableParametr()
{
    switch(current_joint_type)
    {
    case REVOLUTE:
        return theta;
    case PRISMATIC:
        return d;
    }

    return 0.0f;
}