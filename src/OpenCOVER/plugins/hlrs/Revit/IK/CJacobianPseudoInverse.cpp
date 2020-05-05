#include "CJacobianPseudoInverse.h"
#include "CJacobian.h"
#include <iostream>

CJacobianPseudoInverse::CJacobianPseudoInverse( 
                                                    IN VectorXf & desired_position ,
                                                    IN CRobot   & robot
                                               ):
                                                    mtxinstance(CMatrixFactory::GetInstance()),
                                                    _desired_position(desired_position),
                                                    _robot(robot)
                                                    
{
    current_position.resize(6);
}

OUT VectorXf CJacobianPseudoInverse::CalculateData()
{
    //Result vector
    VectorXf delta_theta(_robot.GiveMeMatrixHolder().size());
    //Vector for delta moving in space
    VectorXf delta_translation(6);

    VectorXf _temp = _desired_position;

    CJacobian * jac = CJacobian::GetInstance();
    //TODO
    //should be a parameter of function
    //Desired accuracy
    float epsilon = 0.1f;

    for (;;)
    {
        jac->CalculateJacobian(_robot.GiveMeMatrixHolder(),_robot.GiveMeJoints(),_robot.GiveMeFullHM());
        //calculation delta
        current_position << _robot.GiveMeFullHM()(0,3) ,    //X
                            _robot.GiveMeFullHM()(1,3) ,    //Y
                            _robot.GiveMeFullHM()(2,3) ,    //Z
                            0.0f,                           //Orientation
                            0.0f,                           //Orientation
                            0.0f;                           //Orientation

#ifdef JACOBIANDEBUGOUTPUT
        std::cout<<"Current position"<<std::endl<<current_position<<std::endl;
        std::cout<<"Desired position"<<std::endl<<_desired_position<<std::endl;
#endif

        delta_translation = _desired_position - current_position;

#ifdef JACOBIANDEBUGOUTPUT
        std::cout<<"Delta position"<<std::endl<<delta_translation<<std::endl;
#endif
        //compare delta with desired accuracy
        float n = delta_translation.norm();
        if (n < epsilon)
        {
            //Done
            break;
        }
        //Algo
        delta_theta = jac->PsevdoInverse() * delta_translation;
        
#ifdef JACOBIANDEBUGOUTPUT
        std::cout<<"Delta theta"<<std::endl<<delta_theta<<std::endl;
#endif
        UpdateJoints(delta_theta);
    }

    return _temp;
}

void CJacobianPseudoInverse::UpdateJoints( VectorXf & delta_theta )
{
    JointHandler & _j = _robot.GiveMeJoints();

    for (int i = 0 ; i < delta_theta.size() ; i++)
    {
        //TODO ADD CONSTRAINTS
        //new var = old var + delta var
        float res = delta_theta[i] + _j[i].GiveMeVariableParametr();
        _robot.SetJointVariable(i,res);
    }    
}