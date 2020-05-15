#ifndef __CJACOBIAN__
#define __CJACOBIAN__

#include "JacobianTypes.h"
#include "CMatrixFactory.h"

class CJacobian
{
/************************************************************************/
/* Jacobian holder                                                      */
/************************************************************************/
    MatrixXf  _jacobian;
/************************************************************************/
/* Matrix and vector help functions                                     */
/************************************************************************/
    CMatrixFactory * mtxf;

    CJacobian();

    static CJacobian * _instance;

public:
    
/************************************************************************/
/* Set and get functions                                                */
/************************************************************************/
    void SetJacobianConfiguration( 
                                        IN unsigned int row , 
                                        IN unsigned int col
                                  );
    OUT MatrixXf & GetJacobian();
/************************************************************************/
/* One column of J calculation function                                 */
/************************************************************************/
    void CalculateColumnOfJacobian_New( 
                                            IN HomMatrixHolder & hom_matrix_handler ,
                                            IN unsigned int ind ,
                                            IN JointT jt,
                                            IN Matrix4f & full
                                       );

    void CalculateJacobian(
                                IN HomMatrixHolder & hom_matrix_handler ,
                                IN JointHandler    & jhandler,
                                IN Matrix4f & full
                           );

    OUT MatrixXf  PsevdoInverse();
    
    static CJacobian * GetInstance();
};


#endif