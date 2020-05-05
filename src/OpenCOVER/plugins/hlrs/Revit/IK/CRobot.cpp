#include "CRobot.h"
#include "CMatrixFactory.h"
#include <iostream>

bool CRobot::LoadConfig( const dh_table & tbl )
{
    unsigned int sz = tbl.size();
    if(!sz) return false;

    robot_dh = tbl;

    for (unsigned int i = 0 ; i < sz ; ++i)
    {
        //Saving joint data to robot
        jhandle.push_back(CJoint(robot_dh[i].d,robot_dh[i].theta,robot_dh[i].z_joint_type,robot_dh[i].joint_name));
        //Saving link data  to robot
        linkhadle.push_back(CLink(robot_dh[i].a,robot_dh[i].alpha));
        //Calculating h_matrix and saving to robot
        hmtx.push_back(
            matrix_algo->CalculateHTranslationMatrix(
                                                        linkhadle[i].GetZAxisRotationParametr_aplha(),
                                                        linkhadle[i].GetCommonNormalParametr_a(),
                                                        jhandle[i].GetDisplasmentParametr_d(),
                                                        jhandle[i].GetRotationParametr_theta()
                                                     )                            
                      );
    }

    CaclulateFullTransormationMatrix();
    
    return true;            
}

CRobot::CRobot( Vector3f & vec ) : zero_origin(vec) , matrix_algo(CMatrixFactory::GetInstance()) , number_of_var_parameters(0)
{}

void CRobot::SetOrigin( Vector3f & newOrigin )
{
    zero_origin = newOrigin;
}

bool CRobot::RotateJoint( unsigned int ind , float angle )
{
    if (ind > jhandle.size())
    {
        //Bad index
        return false;
    }
    CJoint & current_joint = jhandle[ind];

    if (current_joint.GetJointType() != REVOLUTE)
    {
        //We can rotate only revolute joint
        return false;
    }
    //TODO
    //add signal for this peace of code
    current_joint.RotateJoint(angle);
    //Recalculate HM for specified joint
    CalculateJoint(ind);

    return true;
}

bool CRobot::TranslateJoint( unsigned int ind , float displasment )
{
    if (ind > jhandle.size())
    {
        //Bad index
        return false;
    }
    CJoint & current_joint = jhandle[ind];
    //We can translate only prismatic joint
    if (current_joint.GetJointType() != PRISMATIC)
    {
        //We can translate only prismatic joint
        return false;
    }
    //TODO
    //add signal for this peace of code
    current_joint.TranlsateJoint(displasment);

    //Recalculate HM for specified joint
    CalculateJoint(ind);

    return true;
}

bool CRobot::PrintHomogenTransformationMatrix()
{
    if(jhandle.size() != hmtx.size())
    {
        //TODO
        //May be i should add some extra info here
        std::cout<<"Number of Joints doesn't match number of ham. matrix. Suppose you have got extra for end-effector"<<std::endl;
        return false;
    }

    for (unsigned int i = 0 ; i < jhandle.size() ; ++i)
    {
        std::cout<<"Joint name : "<<jhandle[i].GetJointName()<<std::endl<<"Joint ID : "<<i+1<<std::endl;
        std::cout<<"Matrix : "<<std::endl<<hmtx[i]<<std::endl<<"------------------------"<<std::endl;
    }

    return true;
}

bool CRobot::CaclulateFullTransormationMatrix()
{
    if (hmtx.empty())
    {
        return false;
    }
    //For beginning we have identity matrix
    from_i_1_to_i = Matrix4f::Identity();
   
    unsigned int j = hmtx.size() - 1;
    //TODO
    //i should add equation representation here
    //for more clearance
    for (unsigned int i = 0 ; i < hmtx.size() ; i++)
    {
        Matrix4f _temp = from_i_1_to_i;
        from_i_1_to_i = _temp * hmtx[i];
    }

    return true;
}

bool CRobot::PrintFullTransformationMatrix()
{
    std::cout<<"Full HM matrix :"<<std::endl<<from_i_1_to_i<<std::endl;
    return true;
}

bool CRobot::CalculateNumberOfVariableParametrs()
{
    if(jhandle.empty())
        return false;

    for (unsigned int i = 0 ; i < jhandle.size() ; ++i)
    {
        if (jhandle[i].GetJointType() != CONSTANTJOINT)
        {
            number_of_var_parameters++;
        }
    }

    return true;
}

bool CRobot::SetJointVariable( IN unsigned int ind, IN float new_var)
{
    bool retval = true;

    switch(jhandle[ind].GetJointType())
    {
    case REVOLUTE:
        jhandle[ind].GetRotationParametr_theta() = new_var;
        break;
    case PRISMATIC:
        jhandle[ind].GetDisplasmentParametr_d() = new_var;
        break;
    default:
        retval = false;
    }

    if (retval)
    {
        CalculateJoint(ind);
    }

    return retval;
}

bool CRobot::CalculateJoint( unsigned int ind )
{
    //TODO
    //add checking
    //Recalculate HM for specified joint
    hmtx[ind] = matrix_algo->CalculateHTranslationMatrix(
                                                                linkhadle[ind].GetZAxisRotationParametr_aplha(),
                                                                linkhadle[ind].GetCommonNormalParametr_a(),
                                                                jhandle[ind].GetDisplasmentParametr_d(),
                                                                jhandle[ind].GetRotationParametr_theta()
                                                         );

    CaclulateFullTransormationMatrix();

    return true;
}

void CRobot::PrintConfiguration()
{
    unsigned int sz = jhandle.size();

    for (unsigned int i = 0 ; i < sz ; ++i)
    {
        std::cout<<"Joint name : "<<jhandle[i].GetJointName()<<std::endl;
        float jvar = jhandle[i].GiveMeVariableParametr();
        
        switch(jhandle[i].GetJointType())
        {
            case PRISMATIC:
                std::cout<<"Prismatic, VAR = ";
                break;
            case REVOLUTE:
                std::cout<<"Revolute, VAR = ";
                break;
        }

        std::cout<<jvar<<std::endl;
    }

    float x_pos = from_i_1_to_i(0,3);
    float y_pos = from_i_1_to_i(1,3);
    float z_pos = from_i_1_to_i(2,3);

    std::cout<<"Position :"<<std::endl;
    std::cout<<"X : "<<x_pos<<std::endl;
    std::cout<<"Y : "<<y_pos<<std::endl;
    std::cout<<"Z : "<<z_pos<<std::endl;
}