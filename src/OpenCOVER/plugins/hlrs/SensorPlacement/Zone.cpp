#include <iterator>

#include<osg/Geode>
#include<osg/Geometry>
#include<osg/LineWidth>
#include<osg/Material>
#include<osg/LightModel>

#include <cover/coVRPluginSupport.h>
#include <cover/VRSceneGraph.h>

#include "Helper.h"
#include "Zone.h"
#include "UI.h"
#include "Sensor.h"
#include "Factory.h"
#include "Profiling.h"

using namespace opencover;

Zone::Zone(osg::Matrix matrix,osg::Vec4 color,float length, float width , float height)
    : m_Color(color), m_Length(length), m_Width(width), m_Height(height)
{
    m_Distance = findLargestVectorComponent(findLongestSide()) / 5;
    std::cout <<"Distance"<<m_Distance <<std::endl;

    m_LocalDCS = new osg::MatrixTransform(matrix);
    m_Geode = draw();
    m_LocalDCS->addChild(m_Geode);

    m_Group = new osg::Group;
    m_Group->addChild(m_LocalDCS.get());

    float _interSize = cover->getSceneSize() / 50;

    osg::Matrix startPosInterator= osg::Matrix::translate(m_Verts->at(2)*matrix);
    m_Interactor= myHelpers::make_unique<coVR3DTransRotInteractor>(startPosInterator, _interSize, vrui::coInteraction::ButtonA, "hand", "Interactor", vrui::coInteraction::Medium);
    m_Interactor->show();
    m_Interactor->enableIntersection();

    osg::Vec3 startPosSizeInteractor= m_Verts->at(4)*matrix;
    m_SizeInteractor = myHelpers::make_unique<coVR3DTransInteractor>(startPosSizeInteractor, _interSize, vrui::coInteraction::ButtonA, "hand", "SizeInteractor", vrui::coInteraction::Medium);
    m_SizeInteractor->show();
    m_SizeInteractor->enableIntersection();

    startPosSizeInteractor= (m_Verts->at(2)-osg::Vec3(m_Distance,0,0))*matrix;
    m_DistanceInteractor = myHelpers::make_unique<coVR3DTransInteractor>(startPosSizeInteractor, _interSize, vrui::coInteraction::ButtonA, "hand", "DistanceInteractor", vrui::coInteraction::Medium);
    m_DistanceInteractor->show();
    m_DistanceInteractor->enableIntersection();
};

osg::Geode* Zone::draw()
{
    osg::Geode* geode = new osg::Geode();
    geode->setName("Wireframee");
    m_Geom = new osg::Geometry();
    osg::StateSet *stateset = VRSceneGraph::instance()->loadDefaultGeostate(osg::Material::AMBIENT_AND_DIFFUSE);
    geode->setStateSet(stateset);
   // setStateSet(stateset);
    //necessary for dynamic redraw (command:dirty)
    m_Geom->setDataVariance(osg::Object::DataVariance::DYNAMIC) ;
    m_Geom->setUseDisplayList(false);
    m_Geom->setNormalBinding(osg::Geometry::BIND_OVERALL);
    m_Geom->setUseVertexBufferObjects(true);
    geode->addDrawable(m_Geom);
    geode->getStateSet()->setMode( GL_BLEND, osg::StateAttribute::ON );
    geode->getStateSet()->setRenderingHint(osg::StateSet::TRANSPARENT_BIN);
    // Declare an array of vertices to create a simple pyramid.
    m_Verts = new osg::Vec3Array;
    m_Verts->push_back( osg::Vec3( -m_Length, m_Width, 0 ) ); // lower back left
    m_Verts->push_back( osg::Vec3( -m_Length,0, 0 ) );// lower front left
    m_Verts->push_back( osg::Vec3(  0,0, 0 ) );// lower front right
    m_Verts->push_back( osg::Vec3(  0, m_Width, 0 ) ); // lower back right
    m_Verts->push_back( osg::Vec3( -m_Length, m_Width,  m_Height ) ); // upper back left
    m_Verts->push_back( osg::Vec3( -m_Length,0,  m_Height ) );// upper front left
    m_Verts->push_back( osg::Vec3(  0,0,  m_Height ) );// upper front right
    m_Verts->push_back( osg::Vec3(  0, m_Width,  m_Height) ); // upper back right

    // Associate this set of vertices with the Geometry.
    m_Geom->setVertexArray(m_Verts);

    //set normals
    osg::ref_ptr<osg::Vec3Array> normals = new osg::Vec3Array;
    normals->push_back((osg::Vec3(0.0,0.0,-1.0)));
    m_Geom->setNormalArray(normals);
    // Next, create primitive sets and add them to the Geometry.
    // Each primitive set represents a Line of the Wireframe

    //Lower Rectangle
    osg::DrawElementsUInt* line = new osg::DrawElementsUInt(osg::PrimitiveSet::LINES, 0,2);
    line->push_back(0);
    line->push_back(1);
    m_Geom->addPrimitiveSet(line);

    line = new osg::DrawElementsUInt(osg::PrimitiveSet::LINES, 0,2);
    line->push_back(1);
    line->push_back(2);
    m_Geom->addPrimitiveSet(line);

    line = new osg::DrawElementsUInt(osg::PrimitiveSet::LINES, 0,2);
    line->push_back(2);
    line->push_back(3);
    m_Geom->addPrimitiveSet(line);

    line = new osg::DrawElementsUInt(osg::PrimitiveSet::LINES, 0,2);
    line->push_back(3);
    line->push_back(0);
    m_Geom->addPrimitiveSet(line);

    //UpperRectangle
    line = new osg::DrawElementsUInt(osg::PrimitiveSet::LINES, 0,2);
    line->push_back(4);
    line->push_back(5);
    m_Geom->addPrimitiveSet(line);

    line = new osg::DrawElementsUInt(osg::PrimitiveSet::LINES, 0,2);
    line->push_back(5);
    line->push_back(6);
    m_Geom->addPrimitiveSet(line);

    line = new osg::DrawElementsUInt(osg::PrimitiveSet::LINES, 0,2);
    line->push_back(6);
    line->push_back(7);
    m_Geom->addPrimitiveSet(line);

    line = new osg::DrawElementsUInt(osg::PrimitiveSet::LINES, 0,2);
    line->push_back(7);
    line->push_back(4);
    m_Geom->addPrimitiveSet(line);

    //Vertical Lines
    line = new osg::DrawElementsUInt(osg::PrimitiveSet::LINES, 0,2);
    line->push_back(0);
    line->push_back(4);
    m_Geom->addPrimitiveSet(line);

    line = new osg::DrawElementsUInt(osg::PrimitiveSet::LINES, 0,2);
    line->push_back(1);
    line->push_back(5);
    m_Geom->addPrimitiveSet(line);

    line = new osg::DrawElementsUInt(osg::PrimitiveSet::LINES, 0,2);
    line->push_back(2);
    line->push_back(6);
    m_Geom->addPrimitiveSet(line);

    line = new osg::DrawElementsUInt(osg::PrimitiveSet::LINES, 0,2);
    line->push_back(3);
    line->push_back(7);
    m_Geom->addPrimitiveSet(line);

    osg::LineWidth *lw = new osg::LineWidth(3.0);
   // stateset->setAttribute(lw);

    m_Colors = new osg::Vec4Array;
    m_Colors->push_back(m_Color);
    m_Geom->setColorArray(m_Colors);
    m_Geom->setColorBinding(osg::Geometry::BIND_OVERALL);
    geode->getStateSet()->setAttribute(lw);
    return geode;
};

bool Zone::preFrame()
{
    m_Interactor->preFrame();
    m_SizeInteractor->preFrame();
    m_DistanceInteractor->preFrame();

    static osg::Vec3 startPos_SizeInteractor_w,startPos_SizeInteractor_o;
    static osg::Vec3 startPos_DistanceInteractor_w,startPos_DistanceInteractor_o;

    static osg::Matrix startMatrix_Interactor_to_w,startMatrix_Interactor_to_w_inverse;
    if(m_Interactor->wasStarted())
    {
        if(UI::m_DeleteStatus)
            return false;

        osg::Matrix interactor_to_w = m_Interactor->getMatrix();
        startPos_SizeInteractor_w = m_SizeInteractor->getPos();
        startPos_DistanceInteractor_w = m_DistanceInteractor->getPos();

        osg::Vec3 interactor_pos_w = interactor_to_w.getTrans();
        startPos_SizeInteractor_o = osg::Matrix::transform3x3(startPos_SizeInteractor_w-interactor_pos_w, interactor_to_w.inverse(interactor_to_w));
        startPos_DistanceInteractor_o = osg::Matrix::transform3x3(startPos_DistanceInteractor_w-interactor_pos_w, interactor_to_w.inverse(interactor_to_w));

    }
    else if(m_Interactor->isRunning())
    {
        osg::Matrix interactor_to_w = m_Interactor->getMatrix();
        m_LocalDCS->setMatrix(interactor_to_w);
        osg::Vec3 interactor_pos_w = interactor_to_w.getTrans();
        
        //update Interactors
        osg::Vec3 sizeInteractor_pos_w = osg::Matrix::transform3x3(startPos_SizeInteractor_o, interactor_to_w);
        sizeInteractor_pos_w +=interactor_pos_w;
        m_SizeInteractor->updateTransform(sizeInteractor_pos_w);

        osg::Vec3 distanceInteractor_pos_w = osg::Matrix::transform3x3(startPos_DistanceInteractor_o, interactor_to_w);
        distanceInteractor_pos_w +=interactor_pos_w;
        m_DistanceInteractor->updateTransform(distanceInteractor_pos_w);
    }
    else if(m_SizeInteractor->wasStarted())
    {
        startMatrix_Interactor_to_w = m_Interactor->getMatrix();
        startMatrix_Interactor_to_w_inverse = startMatrix_Interactor_to_w.inverse(startMatrix_Interactor_to_w); 

        osg::Vec3 interactor_pos_w = startMatrix_Interactor_to_w.getTrans();
        startPos_DistanceInteractor_w = m_DistanceInteractor->getPos();
        startPos_DistanceInteractor_o = osg::Matrix::transform3x3(startPos_DistanceInteractor_w-interactor_pos_w, startMatrix_Interactor_to_w_inverse);
        startPos_DistanceInteractor_o = osg::Vec3(std::abs(startPos_DistanceInteractor_o.x()),std::abs(startPos_DistanceInteractor_o.y()),std::abs(startPos_DistanceInteractor_o.z()));

        deleteGridPoints();
        m_DistanceInteractor->hide();
    }
    else if(m_SizeInteractor->isRunning())
    {        
        // update vertices
        osg::Vec3 verts_o= osg::Matrix::transform3x3(m_SizeInteractor->getPos()-startMatrix_Interactor_to_w.getTrans(), startMatrix_Interactor_to_w_inverse);
        updateGeometry(verts_o);   
    }
    else if(m_SizeInteractor->wasStopped())
    {
        calcPositionOfDistanceInteractor(startPos_DistanceInteractor_o);
        m_DistanceInteractor->show();
        drawGrid();
    }
    else if(m_DistanceInteractor->wasStarted())
    {
        osg::Matrix interactor_to_w = m_Interactor->getMatrix();
        startPos_DistanceInteractor_w = m_DistanceInteractor->getPos();
        startPos_DistanceInteractor_o= osg::Matrix::transform3x3(startPos_DistanceInteractor_w, interactor_to_w.inverse(interactor_to_w));
    }
    else if(m_DistanceInteractor->isRunning())      
        restrictDistanceInteractor(startPos_DistanceInteractor_o);

    else if(m_DistanceInteractor->wasStopped())
    {
        deleteGridPoints();
        drawGrid();
    }
    return true;
};

void Zone::calcPositionOfDistanceInteractor(osg::Vec3& startPos_DistanceInteractor_o)
{
    osg::Vec3 longestSide = findLongestSide();
    float posFactor = 3;

    if(longestSide.y() != 0)
    {
        float startPosY = findLargestVectorComponent(startPos_DistanceInteractor_o);
        if(std::abs(startPosY) > m_Width)
            startPos_DistanceInteractor_o.y() = m_Width / posFactor;
        else
            startPos_DistanceInteractor_o.y() = startPosY;

        startPos_DistanceInteractor_o.z() = 0;
        startPos_DistanceInteractor_o.x() = 0;
    }
    else if(longestSide.x() != 0)
    {
        float startPosX = findLargestVectorComponent(startPos_DistanceInteractor_o);
        if(std::abs(startPosX) > m_Length)   
            startPos_DistanceInteractor_o.x() =   m_Length / posFactor;
        
        else
            startPos_DistanceInteractor_o.x() = startPosX;

        startPos_DistanceInteractor_o.y() = 0;
        startPos_DistanceInteractor_o.z() = 0;
    }
    else if(longestSide.z() != 0)
    {
        float startPosZ = findLargestVectorComponent(startPos_DistanceInteractor_o);
        if(std::abs(startPosZ) > m_Height)
            startPos_DistanceInteractor_o.z() = m_Height / posFactor;
        else
            startPos_DistanceInteractor_o.z() = startPosZ;

        startPos_DistanceInteractor_o.x() = 0;
        startPos_DistanceInteractor_o.y() = 0;
    }

    osg::Vec3 sign = calcSign();
    osg::Vec3 newSign_o = osg::Vec3(startPos_DistanceInteractor_o.x()*sign.x(),startPos_DistanceInteractor_o.y()*sign.y(),startPos_DistanceInteractor_o.z()*sign.z());
    osg::Vec3 distanceInteractor_pos_w = osg::Matrix::transform3x3(newSign_o, m_Interactor->getMatrix());
    osg::Vec3 interactor_pos_w =  m_Interactor->getMatrix().getTrans();
    distanceInteractor_pos_w +=interactor_pos_w; 
    m_DistanceInteractor->updateTransform(distanceInteractor_pos_w);
}

void Zone::restrictDistanceInteractor(osg::Vec3& startPos_DistanceInteractor_o)
{
        
    osg::Matrix interactor_to_w = m_Interactor->getMatrix();
    osg::Vec3 distanceInteractor_pos_o = osg::Matrix::transform3x3(m_DistanceInteractor->getPos(),interactor_to_w.inverse(interactor_to_w));
    osg::Vec3 sizeInteractor_pos_o = osg::Matrix::transform3x3(m_SizeInteractor->getPos(),interactor_to_w.inverse(interactor_to_w));
    osg::Vec3 interactor_pos_o = osg::Matrix::transform3x3(interactor_to_w.getTrans(),interactor_to_w.inverse(interactor_to_w));
    
    osg::Vec3 freeAxis = findLongestSide();
    osg::Vec3 sign = calcSign();
    if(freeAxis.x() != 0)
    {
        // restrict degree of freedom
        distanceInteractor_pos_o.y() = startPos_DistanceInteractor_o.y(); 
        distanceInteractor_pos_o.z() = startPos_DistanceInteractor_o.z(); 
        
        // set min and max position
        if(distanceInteractor_pos_o.x() < sizeInteractor_pos_o.x() && sign.x() < 0 )
            distanceInteractor_pos_o.x() = sizeInteractor_pos_o.x();
        else if(distanceInteractor_pos_o.x() > sizeInteractor_pos_o.x() && sign.x() > 0 )
            distanceInteractor_pos_o.x() = sizeInteractor_pos_o.x();
        else if (distanceInteractor_pos_o.x() > interactor_pos_o.x() && sign.x() < 0)  
            distanceInteractor_pos_o.x() = interactor_pos_o.x();  
        else if (distanceInteractor_pos_o.x() < interactor_pos_o.x() && sign.x() > 0)  
            distanceInteractor_pos_o.x() = interactor_pos_o.x();  
    }
    else if(freeAxis.y() != 0)
    {
        // restrict degree of freedom
        distanceInteractor_pos_o.x() = startPos_DistanceInteractor_o.x(); 
        distanceInteractor_pos_o.z() = startPos_DistanceInteractor_o.z(); 

        // set min and max position
        if(distanceInteractor_pos_o.y() > sizeInteractor_pos_o.y() && sign.y() > 0 )
            distanceInteractor_pos_o.y() = sizeInteractor_pos_o.y();
        else if(distanceInteractor_pos_o.y() < sizeInteractor_pos_o.y() && sign.y() < 0 )
            distanceInteractor_pos_o.y() = sizeInteractor_pos_o.y();
        else if (distanceInteractor_pos_o.y() < interactor_pos_o.y() && sign.y() > 0)  
            distanceInteractor_pos_o.y() = interactor_pos_o.y();  
        else if (distanceInteractor_pos_o.y() > interactor_pos_o.y() && sign.y() < 0)  
            distanceInteractor_pos_o.y() = interactor_pos_o.y();      
    }
    else if(freeAxis.z() != 0)
    {
        // restrict degree of freedom
        distanceInteractor_pos_o.y() = startPos_DistanceInteractor_o.y(); 
        distanceInteractor_pos_o.x() = startPos_DistanceInteractor_o.x(); 

        // set min and max position
        if(distanceInteractor_pos_o.z() > sizeInteractor_pos_o.z() && sign.z() > 0 )
            distanceInteractor_pos_o.z() = sizeInteractor_pos_o.z();
        else if(distanceInteractor_pos_o.z() < sizeInteractor_pos_o.z() && sign.z() < 0 )
            distanceInteractor_pos_o.z() = sizeInteractor_pos_o.z();
        else if (distanceInteractor_pos_o.z() < interactor_pos_o.z() && sign.z() > 0)  
            distanceInteractor_pos_o.z() = interactor_pos_o.z();  
        else if (distanceInteractor_pos_o.z() > interactor_pos_o.z() && sign.z() < 0)  
            distanceInteractor_pos_o.z() = interactor_pos_o.z();      
    }
    
    osg::Vec3 distanceInteractor_pos_w = osg::Matrix::transform3x3(distanceInteractor_pos_o,interactor_to_w);
    m_DistanceInteractor->updateTransform(distanceInteractor_pos_w);
}

osg::Vec3 Zone::findLongestSide() const
{
    osg::Vec3 result;

    if (m_Width >= m_Length && m_Width >= m_Height)
        result = {0,m_Width,0};                                         // y (width) is longest side

    if (m_Length >= m_Width && m_Length >= m_Height)
        result = {m_Length,0,0};                                        // x (length) is longest side

    if (m_Height >= m_Width && m_Height >= m_Length)
        result = {0,0,m_Height};                                        // z (height) is longest side

    return result;
}

float Zone::findLargestVectorComponent(osg::Vec3 vec) const 
{ 
    float result;
    
    if (vec.x() >= vec.y() && vec.x() >= vec.z())
        result = vec.x();

    if (vec.y() >= vec.x() && vec.y() >= vec.z())
        result = vec.y();

    if (vec.z() >= vec.x() && vec.z() >= vec.y())
        result = vec.z();    

    return result;
}

float Zone::calculateGridPointDistance() const
{
    float result;
    osg::Matrix interactor_to_w = m_Interactor->getMatrix();
    osg::Vec3 distanceInteractor_pos_o = osg::Matrix::transform3x3(m_DistanceInteractor->getPos(),interactor_to_w.inverse(interactor_to_w));
    osg::Vec3 interactor_pos_o = osg::Matrix::transform3x3(interactor_to_w.getTrans(),interactor_to_w.inverse(interactor_to_w));

    osg::Vec3 longestSide = findLongestSide();

    if(longestSide.x() != 0)
        result = std::abs(interactor_pos_o.x()-distanceInteractor_pos_o.x());
    else if(longestSide.y() != 0)
        result = std::abs(interactor_pos_o.y()-distanceInteractor_pos_o.y());
    else if(longestSide.z() != 0)
        result = std::abs(interactor_pos_o.z()-distanceInteractor_pos_o.z());
    
    return result;
}
void Zone::drawGrid()
{
    m_Distance = calculateGridPointDistance();
    float minDistance = findLargestVectorComponent(findLongestSide())  / 20 ; // minimum distance between gridpoints

    if(m_Distance < minDistance)
        std::cout << "Zone: No gridpoints created, distance is too small" << std::endl;
    else
        createGrid();
} 

void Zone::updateGeometry(osg::Vec3& vec)
{
    //update y and z coordinate
     m_Verts->at(3) =osg::Vec3(m_Verts->at(3).x(),vec.y(),m_Verts->at(3).z());
     m_Verts->at(4) =osg::Vec3(m_Verts->at(4).x(),vec.y(),vec.z());
     m_Verts->at(7) =osg::Vec3(m_Verts->at(7).x(),vec.y(),vec.z());
     m_Verts->at(0) =osg::Vec3(m_Verts->at(0).x(),vec.y(),m_Verts->at(0).z());
    
    //update x and z coordinate
     m_Verts->at(0) =osg::Vec3(vec.x(),m_Verts->at(0).y(),m_Verts->at(0).z());
     m_Verts->at(4) =osg::Vec3(vec.x(),m_Verts->at(4).y(),m_Verts->at(4).z());
     m_Verts->at(5) =osg::Vec3(vec.x(),m_Verts->at(5).y(),vec.z());
     m_Verts->at(1) =osg::Vec3(vec.x(),m_Verts->at(1).y(),m_Verts->at(1).z());

    //update z coordinate
     m_Verts->at(6) =osg::Vec3(m_Verts->at(6).x(),m_Verts->at(6).y(),vec.z());

     m_Verts->dirty();
     m_Geom->dirtyBound();
     m_Width = std::abs(m_Verts->at(7).y()-m_Verts->at(6).y());
     m_Length = std::abs(m_Verts->at(5).x()-m_Verts->at(6).x());
     m_Height = std::abs(m_Verts->at(6).z()-m_Verts->at(2).z());
};

void Zone::setPosition(osg::Matrix matrix)
{
    m_Interactor->updateTransform(matrix);
    m_LocalDCS->setMatrix(matrix);
    //updatePos of Points!!!!
};

void Zone::deleteGridPoints()
{
    for(const auto& point :m_GridPoints)
        m_LocalDCS->removeChild(point.getPoint());
    m_GridPoints.clear();
};

void Zone::createInner3DGrid(const osg::Vec3& startPoint, const osg::Vec3& sign, const osg::Vec3& limit)
{
    float incrementLength{0.0f}, incrementWidth{0.0f}, incrementHeight{0.0f};
    float widthLimit{limit.y()}, lengthLimit{limit.x()}, heightLimit{limit.z()};
    while(incrementWidth < widthLimit)
    {
        while(incrementLength < lengthLimit)
        {
            while(incrementHeight < heightLimit)
            {   
                osg::Vec3f point = startPoint+osg::Vec3(sign.x()*incrementLength,sign.y()*incrementWidth,sign.z()*incrementHeight);
                addPointToVec(point);
                incrementHeight += m_Distance;
            }
            incrementHeight = 0.0;
            incrementLength += m_Distance;
            
        }
        incrementLength = 0.0;
        incrementWidth += m_Distance;
    }
}

void Zone::addPointToVec(osg::Vec3 point) // use rvalue here ? 
{
    float radius = findLargestVectorComponent(findLongestSide()/15);
    if(radius>0.5)
        radius = 0.5;
    m_GridPoints.push_back(GridPoint(point,m_Color,radius));
    m_LocalDCS->addChild(m_GridPoints.back().getPoint());
}

void Zone::createOuter3DGrid(const osg::Vec3& sign)
{
    float incrementLength{0.0f}, incrementWidth{0.0f}, incrementHeight{0.0f};

    // right side
    osg::Vec3 startPoint = m_Verts.get()->at(2);
    while(incrementWidth < m_Width)
    {
        while(incrementHeight < m_Height)
        {   
            osg::Vec3f point = startPoint+osg::Vec3(sign.x()*incrementLength,sign.y()*incrementWidth,sign.z()*incrementHeight);
            addPointToVec(point);
            incrementHeight += m_Distance;
        }
        incrementHeight = 0.0;
        incrementWidth += m_Distance;
    }

    incrementWidth = incrementLength = incrementHeight = 0;

    // left side
    startPoint = m_Verts.get()->at(1);
    while(incrementWidth < m_Width)
    {
        while(incrementHeight < m_Height)
        {   
            osg::Vec3f point = startPoint+osg::Vec3(sign.x()*incrementLength,sign.y()*incrementWidth,sign.z()*incrementHeight);
            addPointToVec(point);

            incrementHeight += m_Distance;
        }
        incrementHeight = 0.0;
        incrementWidth += m_Distance;
    }

    incrementWidth = incrementLength = incrementHeight = 0;

    // bottom
    startPoint = m_Verts.get()->at(2) + osg::Vec3(m_Distance * sign.x(),0,0);
    while(incrementLength < m_Length - m_Distance)
    {
        while(incrementWidth < m_Width )
        {   
            osg::Vec3f point = startPoint+osg::Vec3(sign.x()*incrementLength,sign.y()*incrementWidth,sign.z()*incrementHeight);
            addPointToVec(point);

            incrementWidth += m_Distance;
        }
        incrementWidth = 0.0;
        incrementLength += m_Distance;
    }

    incrementWidth = incrementLength = incrementHeight = 0;

    // top
    startPoint = m_Verts.get()->at(6) ;
    while(incrementLength < m_Length)
    {
        while(incrementWidth < m_Width )
        {   
            osg::Vec3f point = startPoint+osg::Vec3(sign.x()*incrementLength,sign.y()*incrementWidth,sign.z()*incrementHeight);
            addPointToVec(point);

            incrementWidth += m_Distance;
        }
        incrementWidth = 0.0;
        incrementLength += m_Distance;
    }

    incrementWidth = incrementLength = incrementHeight = 0;

    // front
    startPoint = m_Verts.get()->at(2)+osg::Vec3(m_Distance*sign.x(), 0, m_Distance*sign.z());
    while(incrementLength < m_Length -m_Distance)
    {
        while(incrementHeight < m_Height -m_Distance )
        {   
            osg::Vec3f point = startPoint+osg::Vec3(sign.x()*incrementLength,sign.y()*incrementWidth,sign.z()*incrementHeight);
            addPointToVec(point);

            incrementHeight += m_Distance;
        }
        incrementHeight = 0.0;
        incrementLength += m_Distance; 
    }
    incrementWidth = incrementLength = incrementHeight = 0;

    // back
    startPoint = m_Verts.get()->at(3);
    while(incrementLength < m_Length )
    {
        while(incrementHeight < m_Height  )
        {   
            osg::Vec3f point = startPoint+osg::Vec3(sign.x()*incrementLength,sign.y()*incrementWidth,sign.z()*incrementHeight);
            addPointToVec(point);

            incrementHeight += m_Distance;
        }
        incrementHeight = 0.0;
        incrementLength += m_Distance;
    }
    incrementWidth = incrementLength = incrementHeight = 0;

    //missing line from 7 to 4
    startPoint = m_Verts.get()->at(7);
    while(incrementLength < m_Length -m_Distance )
    {   
        osg::Vec3f point = startPoint+osg::Vec3(sign.x()*incrementLength,sign.y()*incrementWidth,sign.z()*incrementHeight);
        addPointToVec(point);
        incrementLength += m_Distance;
    }
    incrementWidth = incrementLength = incrementHeight = 0;

    //missing line from 5 to 4 
    startPoint = m_Verts.get()->at(5);
    while(incrementWidth < m_Width -m_Distance )
    {   
        osg::Vec3f point = startPoint+osg::Vec3(sign.x()*incrementLength,sign.y()*incrementWidth,sign.z()*incrementHeight);
        addPointToVec(point);
        incrementWidth += m_Distance;
    }
    incrementWidth = incrementLength = incrementHeight = 0;

    //missing line from 0 to 4
    startPoint = m_Verts.get()->at(0);
    while(incrementHeight < m_Height -m_Distance )
    {   
        osg::Vec3f point = startPoint+osg::Vec3(sign.x()*incrementLength,sign.y()*incrementWidth,sign.z()*incrementHeight);
        addPointToVec(point);
        incrementHeight += m_Distance;
    }
    incrementWidth = incrementLength = incrementHeight = 0;
};

osg::Vec3 Zone::calcSign() const
{
    float diffY= m_Verts.get()->at(3).y()-m_Verts.get()->at(2).y();
    float diffX = m_Verts.get()->at(2).x()-m_Verts.get()->at(1).x();
    float diffZ = m_Verts.get()->at(6).z()-m_Verts.get()->at(2).z();

    osg::Vec3 sign;

    if(diffY > 0 && diffX > 0 && diffZ > 0 )    
         sign = {-1,1,1};  
    else if(diffY < 0 && diffX > 0 && diffZ > 0 )
       sign = {-1,-1,1};     
    else if(diffY > 0 && diffX > 0 && diffZ < 0 )
       sign = {-1,1,-1};
    else if(diffY < 0 && diffX > 0 && diffZ < 0 ) 
        sign = {-1,-1,-1};      
    else if(diffY > 0 && diffX < 0 && diffZ > 0 )   
        sign = {1,1,1};      
    else if(diffY < 0 && diffX < 0 && diffZ > 0 ) 
        sign = {1,-1,1};   
    else if(diffY > 0 && diffX < 0 && diffZ < 0 )
        sign = {1,1,-1};     
    else if(diffY < 0 && diffX < 0 && diffZ < 0 )
        sign = {1,-1,-1};

    return sign;
}

osg::Vec3 Zone::defineStartPointForInnerGrid()const
{
    osg::Vec3 corner = m_Verts.get()->at(2);
    osg::Vec3 diff = {m_Distance,m_Distance,m_Distance};
    
    if(m_Length - m_Distance < 0)
        diff[0] = m_Length / 2;  
    
    if(m_Width-m_Distance < 0)    
        diff[1] = m_Width / 2; 

    if(m_Height - m_Distance < 0)
        diff[2] = m_Height / 2;
    
    osg::Vec3 sign = calcSign();

    return  corner + osg::Vec3(diff.x()*sign.x(),diff.y()*sign.y(), diff.z()*sign.z());
}

osg::Vec3 Zone::defineLimitsOfInnerGridPoints()const
{
    osg::Vec3 limits{m_Length-m_Distance,m_Width-m_Distance,m_Height-m_Distance};

    // if distance between Points is larger than one specific site -> position of gridpoints is in the center
    if(m_Length - m_Distance < 0)
        limits[0] = m_Length / 2; 

    if(m_Width-m_Distance < 0)    
        limits[1] = m_Width / 2;  

    if(m_Height - m_Distance < 0)
        limits[2] = m_Height / 2;

    return limits;
}

void Zone::highlitePoints(std::vector<float>& visiblePoints)
{
    if(visiblePoints.size() == m_GridPoints.size())
    {
        for(auto it = visiblePoints.begin(); it != visiblePoints.end(); ++it)
        {
            if(*it != 0)
                m_GridPoints.at(std::distance(visiblePoints.begin(), it)).setColor(osg::Vec4(0,1,0,1));
        }
    }
    else
        std::cout<<"error this shouldn't happen !"<<std::endl; //TODO Abort !!!!
}

void Zone::setOriginalColor()
{
    for(auto& point : m_GridPoints)
        point.setOriginalColor();
}

SafetyZone::SafetyZone(osg::Matrix matrix,float length, float width , float height):Zone(matrix,osg::Vec4{1,0.5,0,1}, length,width,height)
{
    createGrid();
}

void SafetyZone::createGrid()
{
  //createInner3DGrid(defineStartPointForInnerGrid(),calcSign(),defineLimitsOfInnerGridPoints());
  createOuter3DGrid(calcSign());
}

SensorZone::SensorZone(SensorType type, osg::Matrix matrix,float length, float width , float height):Zone(matrix,osg::Vec4{1,0,1,1},length,width,height), m_SensorType(type)
{
    createGrid();
    m_SensorGroup = new osg::Group();
    m_Group->addChild(m_SensorGroup.get());

    createSpecificNbrOfSensors();
}

bool SensorZone::preFrame()
{
    bool status = Zone::preFrame();

    if(m_Interactor->wasStarted() || m_SizeInteractor->wasStarted())
        removeAllSensors();
        
    if(m_Interactor->wasStopped() || m_SizeInteractor->wasStopped() || m_DistanceInteractor->wasStopped()) 
        createSpecificNbrOfSensors();

    //check & restrict all sensors positions
    for(const auto& sensor : m_Sensors)
    {
        sensor->preFrame();
        if(sensor->getMatrix().getTrans().z()>m_SizeInteractor->getMatrix().getTrans().z())
        {  
            //std::cout<<"error"<<std::endl;
            //osg::Matrix m = sensor->getMatrix();
            //m.setTrans(m.getTrans().x(),m.getTrans().y(),m_SizeInteractor->getMatrix().getTrans().z()); 
            //sensor->setMatrix(m);
        }
    }
    return status;
}

void SensorZone::createGrid()
{
  createInner3DGrid(defineStartPointForInnerGrid(),calcSign(),defineLimitsOfInnerGridPoints());
  //createOuter3DGrid(calcSign());
}

osg::Vec3 SensorZone::getFreeSensorPosition() const
{
    int nbrOfSensors = m_Sensors.size();
    return m_GridPoints.at(m_GridPoints.size() - nbrOfSensors -1).getPosition() ;
}

void SensorZone::addSensor(osg::Matrix matrix, bool visible)
{
    m_Sensors.push_back(createSensor(m_SensorType, matrix,visible));
    m_SensorGroup->addChild(m_Sensors.back()->getSensor());
}

void SensorZone::removeAllSensors()
{
    m_Sensors.clear();
    m_SensorGroup->removeChildren(0,m_SensorGroup->getNumChildren());
}

void SensorZone::createAllSensors()
{
    SP_PROFILE_BEGIN_SESSION("CreateSensorZone","SensorPlacement-CreateSensorZone.json");

    SP_PROFILE_FUNCTION();
    
    removeAllSensors();
  
    auto worldpositions = getWorldPositionOfPoints();
    for(const auto& worldpos : worldpositions)
        addSensor(osg::Matrix::translate(worldpos),true);
    
    for(const auto& sensor : m_Sensors)
        sensor->calcVisibility();

    SP_PROFILE_END_SESSION();
}

void SensorZone::createSpecificNbrOfSensors()
{
    removeAllSensors();

    if(m_NbrOfSensors <= m_GridPoints.size())
    {
        for(size_t i{0};i<m_NbrOfSensors;i++)
        {
            osg::Matrix matrix = osg::Matrix::translate(getFreeSensorPosition()*m_LocalDCS->getMatrix());
            osg::Quat q(osg::DegreesToRadians(50.0),osg::X_AXIS);
            matrix.setRotate(q);
            addSensor(matrix,true);
        }
    }
    else
        std::cout <<"Sensor Zone: No sensors created, not enough grid points available" << std::endl;
    
}

GridPoint::GridPoint(osg::Vec3 pos,osg::Vec4& color, float radius)
{
    osg::Matrix local;
    local.setTrans(pos);
    m_LocalDCS = new osg::MatrixTransform();
    m_LocalDCS->setMatrix(local);
    m_LocalDCS->setName("Translation");
    m_Sphere = new osg::Sphere(osg::Vec3(0,0,0), radius);
    m_SphereDrawable = new osg::ShapeDrawable(m_Sphere);
    m_SphereDrawable->setColor(color);
    m_Geode = new osg::Geode();
    //osg::StateSet *mystateSet = m_Geode->getOrCreateStateSet();
    //setStateSet(mystateSet);
    m_Geode->setName("Point");
    m_Geode->addDrawable(m_SphereDrawable);
    m_LocalDCS->addChild(m_Geode.get());
}

std::vector<osg::Vec3> Zone::getWorldPositionOfPoints()
{
    std::vector<osg::Vec3> result;
    result.reserve(m_GridPoints.size());
    for(const auto& point : m_GridPoints)
    {
        result.push_back(point.getPosition()*m_LocalDCS->getMatrix());
    }
    return result;
}

void GridPoint::setColor(const osg::Vec4& color)
{
    m_SphereDrawable->setColor(color);
}

void GridPoint::setOriginalColor()
{
    m_SphereDrawable->setColor(m_Color);
}






