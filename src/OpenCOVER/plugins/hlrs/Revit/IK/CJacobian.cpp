#include "CJacobian.h"
#include <iostream>
#include "config.h"

CJacobian::CJacobian() : mtxf(CMatrixFactory::GetInstance())
{}

void CJacobian::SetJacobianConfiguration( unsigned int row , unsigned int col )
{
    //We don't know configuration
    _jacobian.resize(row , col);
}

MatrixXf & CJacobian::GetJacobian()
{
    //Get calculated Jacobian
    return _jacobian;
}

// Transformation matrix representation
//
//|r11 r12 r13 d14|
//|r21 r22 r23 d24|
//|r31 r32 r33 d34|
//|  0   0   0   1|
//

/************************************************************************/
/* Main routine for J calculation                                       */
/************************************************************************/
void CJacobian::CalculateJacobian( HomMatrixHolder & hom_matrix_handler , JointHandler & jhandler , IN Matrix4f & full)
{
    unsigned int col_num = (unsigned int)jhandler.size();

    if (!col_num)
    {
        //Zero size not allowed
        return;
    }

    unsigned int row_num = NUMBEROFSETPARAMETERS;

    //Set J confiruration
    SetJacobianConfiguration(row_num , col_num);

    for (unsigned int i = 1 ; i < col_num + 1 ; ++i)
    {
        CalculateColumnOfJacobian_New(hom_matrix_handler,DHINDEX(i),jhandler[DHINDEX(i)].GetJointType(),full);
    }
}

CJacobian * CJacobian::_instance = NULL;

CJacobian * CJacobian::GetInstance()
{
    if (!_instance)
        _instance = new CJacobian();

    return _instance;
}

template<typename _Matrix_Type_>
bool mypseudoInverse(const _Matrix_Type_& a, _Matrix_Type_& result, double
    epsilon = std::numeric_limits<typename _Matrix_Type_::Scalar>::epsilon())
{
    if (a.rows() < a.cols())
        return false;
    Eigen::JacobiSVD< _Matrix_Type_ > svd = a.jacobiSvd(ComputeThinU | ComputeThinV);

    typename _Matrix_Type_::Scalar tolerance = (typename _Matrix_Type_::Scalar)(epsilon * std::max(a.cols(),
        a.rows()) * svd.singularValues().array().abs().maxCoeff());

    result = svd.matrixV() * _Matrix_Type_((svd.singularValues().array().abs() >
        tolerance).select(svd.singularValues().
            array().inverse(), 0)).asDiagonal() * svd.matrixU().adjoint();

    return true;
}


OUT MatrixXf CJacobian::PsevdoInverse()
{
    MatrixXf inv;
   /* JacobiSVD<MatrixXf> svd;
    svd.compute(_jacobian , Eigen::ComputeThinU | Eigen::ComputeThinV);
    svd.pinv(inv);*/
    mypseudoInverse(_jacobian, inv);
    return inv;
}

void CJacobian::CalculateColumnOfJacobian_New( IN HomMatrixHolder & hom_matrix_handler , IN unsigned int ind , IN JointT jt , Matrix4f & fullm)
{
    Vector3f z0(0.0f , 0.0f , 1.0f);
    Vector3f zi;
    Matrix4f transf_matrix;
    Matrix3f rot_m;

    Vector3f p_end_effector;
    Vector3f pi;

    //Position of end effector
    p_end_effector<< fullm(0,3), fullm(1,3), fullm(2,3);

#ifdef JACOBIANDEBUGOUTPUT
    std::cout<<"Pe "<<std::endl<<p_end_effector<<std::endl;
#endif

    transf_matrix = Matrix4f::Identity();

    //
    
    for(unsigned int i = 0 ; i < ind ; ++i)
        transf_matrix *= hom_matrix_handler[i];

#ifdef JACOBIANDEBUGOUTPUT
    std::cout<<"Transform matrix 4x4"<<std::endl<<transf_matrix<<std::endl;
#endif

    rot_m = transf_matrix.block(0,0,3,3);

#ifdef JACOBIANDEBUGOUTPUT
    std::cout<<"Rotation matrix 3x3"<<std::endl<<rot_m<<std::endl;
#endif

    //
    //  Zi-1
    //

    zi = rot_m * z0;
    pi << transf_matrix(0,3) , transf_matrix(1,3) , transf_matrix(2,3);

#ifdef JACOBIANDEBUGOUTPUT
    std::cout<<"Zi "<<std::endl<<zi<<std::endl;
    std::cout<<"Pi "<<std::endl<<pi<<std::endl;
#endif

    //
    //  (Pe - Pi-1)
    //
    Vector3f delta_vec = p_end_effector - pi;

#ifdef JACOBIANDEBUGOUTPUT
    std::cout<<"Delta vectors "<<std::endl<<delta_vec<<std::endl;
#endif
    //
    //  Zi x (Pe - Pi-1)
    //
    Vector3f d_rev = zi.cross(delta_vec);

#ifdef JACOBIANDEBUGOUTPUT
    std::cout<<"Eigen mult d_rev"<<std::endl<<d_rev<<std::endl;
#endif

    //We should get type of joint and go further    
    switch(jt)
    {
    case PRISMATIC:
        //For prismatic joint everything is simple :
        //        | z | <--- calculated vector z
        // Cind = |   |
        //        | 0 | <--- zero vector3f
        _jacobian(0,ind) = zi(0);
        _jacobian(1,ind) = zi(1);
        _jacobian(2,ind) = zi(2);
        _jacobian(3,ind) = 0.0f;
        _jacobian(4,ind) = 0.0f;
        _jacobian(5,ind) = 0.0f;
        //Mission for this column complete
        return;
    case REVOLUTE:
        //For revolute joint everything is harder :
        //        | z * d | <--- calculated vector z * vector d
        // Cind = |       |
        //        |   z   | <--- calculated vector z
        _jacobian(0,ind) = d_rev(0);
        _jacobian(1,ind) = d_rev(1);
        _jacobian(2,ind) = d_rev(2);
        _jacobian(3,ind) = zi(0);
        _jacobian(4,ind) = zi(1);
        _jacobian(5,ind) = zi(2);
        break;
    case CONSTANTJOINT:
    case NOTSET:
        break;
    }

#ifdef JACOBIANDEBUGOUTPUT
    std::cout<<"Jacobian column "<<ind<<std::endl<<_jacobian.col(ind)<<std::endl;
#endif

}
