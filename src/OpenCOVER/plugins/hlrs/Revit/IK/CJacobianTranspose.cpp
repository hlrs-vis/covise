#include "CJacobianTranspose.h"
#include "CJacobian.h"
#include "CRobot.h"
#include <iostream>

CJacobianTranspose::CJacobianTranspose(
                                            IN VectorXf        & desired_position , 
                                            IN CRobot          & robot                                                                                        
                                       ): 
                                            mtxinstance(CMatrixFactory::GetInstance()),
                                            _desired_position(desired_position),
                                            _robot(robot)
                                         
{
    current_position.resize(6);
}

void CJacobianTranspose::SetAdditionalParametr( IN float & add_in )
{
    lamda_coefficent = add_in;    
}

OUT VectorXf CJacobianTranspose::CalculateData()
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
                            _robot.GiveMeFullHM()(2,3) ,    //Y
                            0.0f,
                            0.0f,
                            0.0f;
                            
        delta_translation = _desired_position - current_position;
#ifdef JACOBIANDEBUGOUTPUT
        std::cout<<"Current position"<<std::endl<<current_position<<std::endl;
        std::cout<<"Desired position"<<std::endl<<_desired_position<<std::endl;
        std::cout<<"Delta translation "<<std::endl<<delta_translation<<std::endl;
#endif
        //compare delta with desired accuracy
        float n = delta_translation.norm();
        if (n < epsilon)
        {
            //Done
            break;
        }

        //Lets calculate lambda
        //TODO optimize it

        MatrixXf _jac_tr =  jac->GetJacobian().transpose();
        
        VectorXf upper = jac->GetJacobian() * _jac_tr * delta_translation;
        VectorXf double_upper = upper;
        float    one   = delta_translation.dot(upper);
        float    down  = upper.dot(double_upper);

        lamda_coefficent = one/down;


        delta_theta = lamda_coefficent*jac->GetJacobian().transpose() * delta_translation;

#ifdef JACOBIANDEBUGOUTPUT
        std::cout<<"Delta theta"<<std::endl<<delta_theta<<std::endl;
#endif

        UpdateJoints(delta_theta);
    }

    return _temp;
}

void CJacobianTranspose::UpdateJoints(VectorXf & delta_theta)
{
    JointHandler & _j = _robot.GiveMeJoints();

    for (int i = 0 ; i < delta_theta.size() ; i++)
    {
        //new var = old var + delta var
        float old = _j[i].GiveMeVariableParametr();
        float delta = delta_theta[i];
        float res = delta + old;
        _robot.SetJointVariable(i,res);
    }    
}


