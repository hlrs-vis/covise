#include "Sensor.h"
#include "Helper.h"
#include "UI.h"
#include "DataManager.h"
#include "Profiling.h"

#include <osg/Geometry>
#include <numeric>

using namespace opencover;

void printCoCoord(coCoord m)
{
    std::cout<<"translation: "<<"x:"<<m.xyz[0]<< " y:"<<m.xyz[1]<<" z:"<<m.xyz[2]<<std::endl;
    std::cout<<"rotation: "<<"z:"<<m.hpr[0]<< " x:"<<m.hpr[1]<<" y:"<<m.hpr[2]<<std::endl;
}

double calcValueInRange(double oldMin, double oldMax, double newMin, double newMax, double oldValue)
{
    double oldRange = oldMax - oldMin;
    double newRange = newMax - newMin;
    double newValue = (((oldValue - oldMin) * newRange) / oldRange )+ newMin;
    return newValue;
}

std::vector<osg::Vec3> Vec2DimToVec(std::vector<std::vector<osg::Vec3>> input)
{    
  /*  std::vector<osg::Vec3> result;
    
    size_t reserve_size = 0;
    
    for(const auto& i : input)
        reserve_size += i.size();

    result.reserve(reserve_size);
        
    for (int i=0; i<bits.size(); ++i)
    {
        const vector<int> & v = bits[i];  // just to make code more readable (note ..  a reference)
        
        bits1d.insert( bits1d.end() , v.begin() , v.end() );
    }
    for(const auto& i : input.size())
    {
        const std::vector<osg::Vec3> tmp = i;
        result.insert(result.end(),tmp.begin(),tmp.end());
    }
    */
    std::vector<osg::Vec3> result;
    return result;
}

/*void checkVisibility(const osg::Matrix& sensorMatrix, VisbilityMatrix& visMat)
{

};
*/
Orientation::Orientation(osg::Matrix matrix):m_Matrix(matrix)
{
}
Orientation::Orientation(osg::Matrix matrix,VisibilityMatrix<float>&& visMat):m_Matrix(matrix),m_Euler(matrix),m_VisibilityMatrix(visMat)
{
}

Orientation::Orientation(coCoord euler,VisibilityMatrix<float>&& visMat):m_Euler(euler),m_VisibilityMatrix(visMat)
{
    euler.makeMat(m_Matrix);
}

void Orientation::setMatrix(osg::Matrix matrix)
{
   m_Matrix = matrix;
   m_Euler = matrix; 
}

void Orientation::setMatrix(coCoord euler)
{
   m_Euler = euler; 
   euler.makeMat(m_Matrix);
}

void Orientation::setVisibilityMatrix(VisibilityMatrix<float>&& visMat)
{
    m_VisibilityMatrix.clear();
    m_VisibilityMatrix = std::move(visMat);
}

// bool Orientation::operator >> (const Orientation& other) const
// {
//     size_t onlyVisibleForThisSensor{0}, onlyVisibleForOtherSensor{0};

//     auto ItThisSensor = m_VisibilityMatrix.begin(); 
//     auto ItOtherSensor = other.m_VisibilityMatrix.begin();

//     if(m_VisibilityMatrix.size() == other.m_VisibilityMatrix.size())
//     {
//         while(ItThisSensor != m_VisibilityMatrix.end() || ItOtherSensor != other.m_VisibilityMatrix.end() )
//         {
//             if(*ItThisSensor > 0 && *ItOtherSensor == 0)
//                 ++onlyVisibleForThisSensor;
//             if(*ItThisSensor == 0 && *ItOtherSensor > 0)
//                 ++onlyVisibleForOtherSensor;

//             if((onlyVisibleForThisSensor && onlyVisibleForOtherSensor) !=0) 
//                 return false; //each sensor can see points, which the other can't see --> keep both 
            
//             // Loop further
//             if(ItThisSensor != m_VisibilityMatrix.end())
//             {
//                 ++ItOtherSensor;
//                 ++ItThisSensor;
//             }
//         }
//     }

//     if(onlyVisibleForThisSensor != 0 && onlyVisibleForOtherSensor == 0)
//         return true;// this sensor can see points, which other can't see
//     else if(onlyVisibleForThisSensor == 0 && onlyVisibleForOtherSensor != 0)
//         return false; // other sensor can see points, which this can't see
//     else if(onlyVisibleForThisSensor == 0 && onlyVisibleForOtherSensor == 0)
//     {
//         //TODO: prefere no Rotation around own axis !
//         //TODO: This is specific to sensor Type ????? 

//         //both sensors see exactly the same points -> check other Sensor characteristics
//         double coefficientsThis = std::accumulate(m_VisibilityMatrix.begin(),m_VisibilityMatrix.end(),0.0);
//         double coefficientOther = std::accumulate(other.m_VisibilityMatrix.begin(),other.m_VisibilityMatrix.end(),0.0);
//         if(coefficientOther > coefficientsThis)
//             return false;
//         else
//             return true;
        
//     }


// }


SensorPosition::SensorPosition(osg::Matrix matrix, bool visible = true):m_CurrentOrientation(matrix)
{
    m_SensorGroup = new osg::Group();
    m_SensorGroup->setName("Sensor");

    m_SensorMatrix = new osg::MatrixTransform(matrix);
    m_SensorMatrix->setName("Sensor matrix");

    m_SensorGroup->addChild(m_SensorMatrix.get());

    float _interSize = cover->getSceneSize() / 50 ;
    m_Interactor = myHelpers::make_unique<coVR3DTransRotInteractor>(matrix, _interSize, vrui::coInteraction::ButtonA, "hand", "CamInteractor", vrui::coInteraction::Medium);
    m_Interactor->enableIntersection();

}
void SensorPosition::calcVisibility()
{
    coCoord euler = m_SensorMatrix->getMatrix();
    checkForObstacles();
    m_CurrentOrientation.setVisibilityMatrix(calcVisibilityMatrix(euler));
}

osg::ref_ptr<osg::Group> SensorPosition::getSensor()const
{
    return m_SensorGroup;
}

void SensorPosition::setMatrix(osg::Matrix matrix)
{
    m_CurrentOrientation.setMatrix(matrix);
    m_SensorMatrix->setMatrix(matrix);
    m_Interactor->updateTransform(matrix);
}

void SensorPosition::setCurrentOrientation(Orientation orientation)
{
    m_CurrentOrientation = orientation;
    //Missing update Interactor !!! not possible here? 
    m_SensorMatrix->setMatrix(orientation.getMatrix());
}

bool SensorPosition::preFrame()
{
    if(m_Interactor->wasStarted())
    {   
        if(UI::m_DeleteStatus)
            return false;
    }
    else if(m_Interactor->isRunning())
    {
        m_SensorMatrix->setMatrix(m_Interactor->getMatrix());
    }
    else if(m_Interactor->wasStopped())
    {
        osg::Matrix matrix= m_Interactor->getMatrix();
        
        SP_PROFILE_BEGIN_SESSION("CreateOrientations","SensorPlacement-CreateOrientations.json");
        
        setMatrix(matrix);
        coCoord euler = matrix;
        checkForObstacles();
        m_CurrentOrientation.setVisibilityMatrix(calcVisibilityMatrix(euler));
        
        //DataManager::highlitePoints(m_Sensor->getVisibilityMatrix());
    }
    return true; 
}

osg::Matrix SensorPosition::getMatrix()const 
{
    return m_SensorMatrix->getMatrix();
}

void SensorPosition::checkForObstacles()
{
    SP_PROFILE_FUNCTION();

    auto allObservationPoints = DataManager::GetInstance().GetWorldPosOfObervationPoints();
    
    m_VisMatSensorPos.clear();
    m_VisMatSensorPos.reserve(allObservationPoints.size());
    
    osg::ref_ptr<osgUtil::IntersectorGroup> intersectorGroup = new osgUtil::IntersectorGroup();

    osg::Vec3 sensorPos = m_SensorMatrix->getMatrix().getTrans();
    for(const auto& point : allObservationPoints)
    {
        osg::ref_ptr<osgUtil::LineSegmentIntersector> intersector = new osgUtil::LineSegmentIntersector(sensorPos, point);
        intersectorGroup->addIntersector( intersector.get() );
    }
    osgUtil::IntersectionVisitor visitor(intersectorGroup.get());
    visitor.setTraversalMask(~DataManager::GetRootNode()->getNodeMask()); //make nodes of sensor placement plugin not pickable by the visitor
    cover->getObjectsRoot()->accept(visitor);

    osgUtil::IntersectorGroup::Intersectors& intersectors = intersectorGroup->getIntersectors();
    for(osgUtil::IntersectorGroup::Intersectors::iterator intersector_itr = intersectors.begin();
        intersector_itr != intersectors.end(); 
        ++intersector_itr)
    {
        osgUtil::LineSegmentIntersector* lsi = dynamic_cast<osgUtil::LineSegmentIntersector*>(intersector_itr->get());
        if(lsi)
        {
            osgUtil::LineSegmentIntersector::Intersections& hits = lsi->getIntersections();
            //std::cout<<hits.size()<<std::endl;
            if(hits.size() > 0 ) 
                m_VisMatSensorPos.push_back(0); // not visible
            else
                m_VisMatSensorPos.push_back(1); // visible
        }
    }
  
};

void SensorPosition::VisualizationVisible(bool status)const
{
    if(status)
        m_SensorGroup->setNodeMask(m_NodeMask);
    else
        m_SensorGroup->setNodeMask(0);
    
}

SensorWithMultipleOrientations::SensorWithMultipleOrientations(osg::Matrix matrix):SensorPosition(matrix)
{
    m_OrientationsGroup = new osg::Group();
    m_OrientationsGroup->setName("Orientations");
    m_SensorGroup->addChild(m_OrientationsGroup.get());
}

bool SensorWithMultipleOrientations::preFrame()
{
    bool status = SensorPosition::preFrame();
    if(m_Interactor->wasStarted())
        deleteSensorOrientations();

    if(m_Interactor->wasStopped())
    {   
        if(UI::m_showOrientations)
            createSensorOrientations();
        
        SP_PROFILE_END_SESSION();
    }

    return status;
}

void SensorWithMultipleOrientations::calcVisibility()
{
    SensorPosition::calcVisibility();
    createSensorOrientations();
}

// const Orientation* SensorWithMultipleOrientations::getRandomOrientation()const
// {
//     //TODO: add random
//     return &m_Orientations.at(0);
// }

void SensorWithMultipleOrientations::createSensorOrientations()
{
    SP_PROFILE_FUNCTION();

    deleteSensorOrientations();
    //m_Orientations.reserve(); // sinnvoll weil kann auch nur eine übrig bleiben? 
    osg::Vec3 pos = m_CurrentOrientation.getMatrix().getTrans();
    osg::Matrix matrix = osg::Matrix::translate(pos);
    coCoord euler = matrix;

    size_t rotz{0},rotx{0},roty{0};
    while(rotz < 360/m_SensorProps.m_StepSizeZ ) // z rotation
    {
        while(rotx < 180/m_SensorProps.m_StepSizeX) // x rotation
        {
            while(roty < 360/m_SensorProps.m_StepSizeY) // y rotation
            {
                decideWhichOrientationsAreRequired(Orientation{euler,calcVisibilityMatrix(euler)});
                euler.hpr[2] += m_SensorProps.m_StepSizeY;
                roty++;
            }
            euler.hpr[1] += m_SensorProps.m_StepSizeX;
            roty = 0;
            rotx++;
        }
        euler.hpr[0] += m_SensorProps.m_StepSizeZ;
        rotx = 0;
        rotz++;
    }
}

void SensorWithMultipleOrientations::deleteSensorOrientations()
{
    m_Orientations.clear();
    m_OrientationsGroup->removeChildren(0,m_OrientationsGroup->getNumChildren());
}


void SensorWithMultipleOrientations::setMatrix(osg::Matrix matrix)
{
    SensorPosition::setMatrix(matrix);
};

void SensorWithMultipleOrientations::decideWhichOrientationsAreRequired(const Orientation&& orientation)
{
    SP_PROFILE_FUNCTION();

    if(isVisibilityMatrixEmpty(orientation))
        return; // this orientation isn't required

    if(!m_Orientations.empty())
    {
        auto it = m_Orientations.begin();
        while(it != m_Orientations.end())
        {
            if(compareOrientations(*it, orientation))
                return; // this orientation isn't required
            else if(compareOrientations(orientation,*it))
            {
                if(m_Orientations.size() > 1)
                    replaceOrientationWithLastElement(std::distance(m_Orientations.begin(),it));
                else
                    m_Orientations.pop_back();
            
                continue; // do not incremenct iterator !
            }
            else
                ++it;  
        }
    }

    m_Orientations.push_back(orientation);
}

bool SensorWithMultipleOrientations::isVisibilityMatrixEmpty(const Orientation& orientation)
{
    SP_PROFILE_FUNCTION();

    bool empty = std::all_of(orientation.getVisibilityMatrix().begin(), orientation.getVisibilityMatrix().end(), [](float i) { return i == 0.0; });
    if(empty)
        return true;
    else
        return false;
}

void SensorWithMultipleOrientations::replaceOrientationWithLastElement(int index)
{
    SP_PROFILE_FUNCTION();

    if(!m_Orientations.empty())
    {
        m_Orientations.at(index) = m_Orientations.back();
        m_Orientations.pop_back();
    }   
}

bool SensorWithMultipleOrientations::compareOrientations(const Orientation& lhs, const Orientation& rhs)
{
    SP_PROFILE_FUNCTION();

    size_t onlyVisibleForLhsSensor{0}, onlyVisibleForRhsSensor{0};

    auto ItLhsSensor = lhs.getVisibilityMatrix().begin(); 
    auto ItRhsSensor = rhs.getVisibilityMatrix().begin();

    if(lhs.getVisibilityMatrix().size() == rhs.getVisibilityMatrix().size())
    {
        while(ItLhsSensor != lhs.getVisibilityMatrix().end() || ItRhsSensor != rhs.getVisibilityMatrix().end() )
        {
            if(*ItLhsSensor > 0 && *ItRhsSensor == 0)
                ++onlyVisibleForLhsSensor;
            if(*ItLhsSensor == 0 && *ItRhsSensor > 0)
                ++onlyVisibleForRhsSensor;

            if((onlyVisibleForLhsSensor && onlyVisibleForRhsSensor) !=0) 
                return false; //each sensor can see points, which the other can't see --> keep both 
            
            // Loop further
            if(ItLhsSensor != lhs.getVisibilityMatrix().end())
            {
                ++ItRhsSensor;
                ++ItLhsSensor;
            }
        }
    }

    if(onlyVisibleForLhsSensor != 0 && onlyVisibleForRhsSensor == 0) // Lhs sensor can see points, which Rhs can't see
        return true;
    else if(onlyVisibleForLhsSensor == 0 && onlyVisibleForRhsSensor != 0) // Rhs sensor can see points, which Lhs can't see
        return false; 
    else if(onlyVisibleForLhsSensor == 0 && onlyVisibleForRhsSensor == 0) // both sensors see exactly the same points --> check other Sensor characteristics
    {
        //TODO: prefere no Rotation around own axis !
        //TODO: Lhs is specific to sensor Type ????? 

        //both sensors see exactly the same points -> check Rhs Sensor characteristics
        double coefficientsLhs = std::accumulate(lhs.getVisibilityMatrix().begin(),lhs.getVisibilityMatrix().end(),0.0);
        double coefficientRhs = std::accumulate(rhs.getVisibilityMatrix().begin(),rhs.getVisibilityMatrix().end(),0.0);
        if(coefficientRhs > coefficientsLhs)
            return false;
        else
            return true;  
    }
    return false;

}













