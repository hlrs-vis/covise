#include<osg/Geode>
#include<osg/Geometry>
#include<osg/LineWidth>
#include<osg/Material>
#include<osg/LightModel>

#include <cover/coVRPluginSupport.h>

#include "Helper.h"
#include "Zone.h"
#include "UI.h"
using namespace opencover;


Zone::Zone(osg::Matrix matrix)
{
    m_LocalDCS = new osg::MatrixTransform(matrix);
    m_Geode = draw();
    m_LocalDCS->addChild(m_Geode);

    float _interSize = cover->getSceneSize() / 50;

    osg::Matrix startPosInterator= osg::Matrix::translate(m_Verts->at(2)*matrix);
    m_Interactor= myHelpers::make_unique<coVR3DTransRotInteractor>(startPosInterator, _interSize, vrui::coInteraction::ButtonA, "hand", "Interactor", vrui::coInteraction::Medium);
    m_Interactor->show();
    m_Interactor->enableIntersection();

    osg::Vec3 startPosSizeInteractor= m_Verts->at(4)*matrix;
    m_SizeInteractor = myHelpers::make_unique<coVR3DTransInteractor>(startPosSizeInteractor, _interSize, vrui::coInteraction::ButtonA, "hand", "SizeInteractor", vrui::coInteraction::Medium);
    m_SizeInteractor->show();
    m_SizeInteractor->enableIntersection();

    //startPosSizeInteractor= (m_Verts->at(2)-osg::Vec3(m_Length/2,0,0))*matrix;
    //m_DistanceInteractor = myHelpers::make_unique<coVR3DTransInteractor>(startPosSizeInteractor, _interSize, vrui::coInteraction::ButtonA, "hand", "DistanceInteractor", vrui::coInteraction::Medium);
    //m_DistanceInteractor->show();
    //m_DistanceInteractor->enableIntersection();

  //  cover->getObjectsRoot()->addChild(m_LocalDCS.get());
    createGridPoints();

};

osg::Geode* Zone::draw()
{
    osg::Geode* geode = new osg::Geode();
    geode->setName("Wireframee");
    m_Geom = new osg::Geometry();
    osg::StateSet *stateset = geode->getOrCreateStateSet();
   // setStateSet(stateset);
    //necessary for dynamic redraw (command:dirty)
    m_Geom->setDataVariance(osg::Object::DataVariance::DYNAMIC) ;
    m_Geom->setUseDisplayList(false);
    m_Geom->setColorBinding(osg::Geometry::BIND_OVERALL);
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
    stateset->setAttribute(lw);
    return geode;
};

bool Zone::preFrame()
{
    m_Interactor->preFrame();
    m_SizeInteractor->preFrame();
    //m_DistanceInteractor->preFrame();

    static osg::Vec3 startPos_SizeInteractor_w,startPos_SizeInteractor_o;
    static osg::Vec3 startPos_DistanceInteractor_w,startPos_DistanceInteractor_o;

    static osg::Matrix startMatrix_Interactor_to_w,startMatrix_Interactor_to_w_inverse;
    if(m_Interactor->wasStarted())
    {
        if(UI::m_DeleteStatus)
            return false;
            
        osg::Matrix interactor_to_w = m_Interactor->getMatrix();
        startPos_SizeInteractor_w = m_SizeInteractor->getPos();
        //startPos_DistanceInteractor_w = m_DistanceInteractor->getPos();

        osg::Vec3 interactor_pos_w = interactor_to_w.getTrans();
        startPos_SizeInteractor_o= osg::Matrix::transform3x3(startPos_SizeInteractor_w-interactor_pos_w, interactor_to_w.inverse(interactor_to_w));
        //startPos_DistanceInteractor_o= osg::Matrix::transform3x3(startPos_DistanceInteractor_w-interactor_pos_w, interactor_to_w.inverse(interactor_to_w));

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
        //m_DistanceInteractor->updateTransform(distanceInteractor_pos_w);
    }
    else if(m_Interactor->wasStopped())
    {
        // update world pos of points ? 
    }

    else if(m_SizeInteractor->wasStarted())
    {
        startMatrix_Interactor_to_w = m_Interactor->getMatrix();
        startMatrix_Interactor_to_w_inverse = startMatrix_Interactor_to_w.inverse(startMatrix_Interactor_to_w); 

        deleteGridPoints();
    }
    else if(m_SizeInteractor->isRunning())
    {        
        // update vertices
        osg::Vec3 verts_o= osg::Matrix::transform3x3(m_SizeInteractor->getPos()-startMatrix_Interactor_to_w.getTrans(), startMatrix_Interactor_to_w_inverse);
        updateGeometry(verts_o);        
    }
    else if(m_SizeInteractor->wasStopped())
    {
        createGridPoints();
    }

    // else if(m_DistanceInteractor->wasStarted())
    // {
    //     osg::Matrix interactor_to_w = m_Interactor->getMatrix();
    //     startPos_DistanceInteractor_w = m_DistanceInteractor->getPos();
    //     startPos_DistanceInteractor_o= osg::Matrix::transform3x3(startPos_DistanceInteractor_w, interactor_to_w.inverse(interactor_to_w));

    // }
    // else if(m_DistanceInteractor->isRunning())
    // {        
    //    //restrict directions of m_DistanceInteractor
    //     osg::Matrix interactor_to_w = m_Interactor->getMatrix();
    //     osg::Vec3 distanceInteractor_pos_o = osg::Matrix::transform3x3(m_DistanceInteractor->getPos(),interactor_to_w.inverse(interactor_to_w));
    //     distanceInteractor_pos_o.y() = startPos_DistanceInteractor_o.y(); //set y value to old y value
    //     distanceInteractor_pos_o.z() = startPos_DistanceInteractor_o.z(); //set z value to old z value
    //     osg::Vec3 distanceInteractor_pos_w = osg::Matrix::transform3x3(distanceInteractor_pos_o,interactor_to_w);
    //     m_DistanceInteractor->updateTransform(distanceInteractor_pos_w);
    // }
    // else if(m_DistanceInteractor->wasStopped())
    // {
       
    // }
    return true;
};
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

void Zone::create3DGrid(const osg::Vec3& startPoint, const osg::Vec3& sign)
{
    static int count{0};
    float incrementLength{0.0f}, incrementWidth{0.0f}, incrementHeight{0.0f};
    
    while(incrementWidth < m_Width-m_Distance)
    {
        while(incrementLength < m_Length-m_Distance)
        {
            while(incrementHeight < m_Height-m_Distance)
            {
                osg::Vec3f point = startPoint+osg::Vec3(sign.x()*incrementLength,sign.y()*incrementWidth,sign.z()*incrementHeight);
                m_GridPoints.push_back(GridPoint(point,m_Color));
                m_LocalDCS->addChild(m_GridPoints.back().getPoint());
                incrementHeight += m_Distance;
            }
            incrementHeight = 0.0;
            incrementLength += m_Distance;
        }
        incrementLength = 0.0;
        incrementWidth += m_Distance;
    }
};

void Zone::createGridPoints()
{
    float diffY= m_Verts.get()->at(3).y()-m_Verts.get()->at(2).y();
    float diffX = m_Verts.get()->at(2).x()-m_Verts.get()->at(1).x();
    float diffZ = m_Verts.get()->at(6).z()-m_Verts.get()->at(2).z();

    osg::Vec3 startPoint = m_Verts.get()->at(2);//+osg::Vec3(-m_Distance, m_Distance, 0);

    //create 3D Grid
    if(diffY > 0 && diffX > 0 && diffZ > 0 )           
        create3DGrid(startPoint,osg::Vec3(-1,1,1));
    else if(diffY < 0 && diffX > 0 && diffZ > 0 )
        create3DGrid(startPoint,osg::Vec3(-1,-1,1));
    else if(diffY > 0 && diffX > 0 && diffZ < 0 )
        create3DGrid(startPoint,osg::Vec3(-1,1,-1));
    else if(diffY < 0 && diffX > 0 && diffZ < 0 )
        create3DGrid(startPoint,osg::Vec3(-1,-1,-1)); 
    else if(diffY > 0 && diffX < 0 && diffZ > 0 )
        create3DGrid(startPoint,osg::Vec3(1,1,1));
    else if(diffY < 0 && diffX < 0 && diffZ > 0 )
        create3DGrid(startPoint,osg::Vec3(1,-1,1)); 
    else if(diffY > 0 && diffX < 0 && diffZ < 0 )
        create3DGrid(startPoint,osg::Vec3(1,1,-1));
    else if(diffY < 0 && diffX < 0 && diffZ < 0 )
        create3DGrid(startPoint,osg::Vec3(1,-1,-1));     
};

GridPoint::GridPoint(osg::Vec3 pos,osg::Vec4& color)
{
    osg::Matrix local;
    local.setTrans(pos);
    m_LocalDCS = new osg::MatrixTransform();
    m_LocalDCS->setMatrix(local);
    m_LocalDCS->setName("Translation");
    m_Sphere = new osg::Sphere(osg::Vec3(0,0,0), 0.45);
    m_SphereDrawable = new osg::ShapeDrawable(m_Sphere);
    m_SphereDrawable->setColor(color);
    m_Geode = new osg::Geode();
    //osg::StateSet *mystateSet = m_Geode->getOrCreateStateSet();
    //setStateSet(mystateSet);
    m_Geode->setName("Point");
    m_Geode->addDrawable(m_SphereDrawable);
    m_LocalDCS->addChild(m_Geode.get());
};
