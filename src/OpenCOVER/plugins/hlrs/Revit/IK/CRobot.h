#ifndef __CROBOT__
#define __CROBOT__

#include "RobotTypes.h"
#include "CMatrixFactory.h"

class CRobot
{
/************************************************************************/
/* Original robot configuration                                         */
/************************************************************************/
    dh_table robot_dh;
    unsigned int number_of_var_parameters;
/************************************************************************/
/* Origin aka x0y0z0                                                                     */
/************************************************************************/
    Vector3f zero_origin;
/************************************************************************/
/* Joint,Links,Matrix containers                                        */
/************************************************************************/
    JointHandler    jhandle;
    LinkHandler     linkhadle;
    HomMatrixHolder hmtx;
/************************************************************************/
/* Full transformation matrix (from i frame to 0 frame)                 */
/************************************************************************/
    Matrix4f        from_i_1_to_i;
/************************************************************************/
/* Matrix factory (not pattern)                                         */
/************************************************************************/
    CMatrixFactory * matrix_algo;
public:
    CRobot(Vector3f & vec);
    bool LoadConfig(IN const dh_table & tbl);
    void SetOrigin(IN Vector3f & newOrigin);

    bool RotateJoint(IN unsigned int ind , IN float angle);
    bool TranslateJoint(IN unsigned int ind , IN float displasment);
    bool SetJointVariable(IN unsigned int , IN float);

    bool PrintHomogenTransformationMatrix();
    bool CaclulateFullTransormationMatrix();

    bool PrintFullTransformationMatrix();
    bool CalculateNumberOfVariableParametrs();
    bool CalculateJoint(unsigned int ind);

    /************************************************************************/
    /* Help functions                                                       */
    /************************************************************************/
    Matrix4f & GiveMeFullHM()
    {
        return from_i_1_to_i;
    }

    CRobot * GiveMeRobot()
    {
        return this;
    }

    JointHandler & GiveMeJoints()
    {
        return jhandle;
    }

    HomMatrixHolder & GiveMeMatrixHolder()
    {
        return  hmtx;
    }

    void PrintConfiguration();

};

#endif