#ifndef __CJOINT__
#define __CJOINT__

#include "SharedTypes.h"
#include <string>

class CJoint
{
    std::string joint_name;

    /************************************************************************/
    /* Origin of reference frame                                            */
    /************************************************************************/

    Vector3f global_joint_origin;

    Vector3f x_axis_unit_point;
    Vector3f y_axis_unit_point;
    Vector3f z_axis_unit_point;

    /************************************************************************/
    /* DH parametrs                                                         */
    /************************************************************************/

    float d;
    float theta;

    /************************************************************************/
    /* Joint type                                                           */
    /************************************************************************/

    JointT current_joint_type;

public:

    CJoint(float d_in , float theta_in , JointT jt_in , std::string jname = "unnamed");

    void RotateJoint(IN const float & rot_angle);
    void TranlsateJoint(IN float & displacement);
    OUT JointT GetJointType() {return current_joint_type;}
    OUT float GiveMeVariableParametr();

    OUT float & GetDisplasmentParametr_d(){return d;}
    OUT float & GetRotationParametr_theta(){return theta;}

    OUT std::string & GetJointName(){return joint_name;}
    void SetGlobalPosition(IN const Vector3f & vec);
};

#endif