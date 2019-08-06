 /* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */
#include <iostream>

#include <Cam.h>

using namespace opencover;

double Cam::imgHeightPixel = 1080;
double Cam::imgWidthPixel = 1920;
double Cam::fov = 60;
double Cam::depthView = 30;
double Cam::focalLengthPixel = Cam::imgWidthPixel*0.5/(std::tan(Cam::fov*0.5*M_PI/180));
double Cam::imgWidth = 2*depthView*std::tan(Cam::fov/2*osg::PI/180);
double Cam::imgHeight = Cam::imgWidth/(Cam::imgWidthPixel/Cam::imgHeightPixel);

Cam::Cam(const osg::Vec3 pos, const osg::Vec2 rot, const osg::Vec3Array &observationPoints):pos(pos),rot(rot)
{
    calcVisMat(observationPoints);
    std::cout<<"new Cam with visMat"<<std::endl;
}

Cam::Cam(const osg::Vec3 pos,const osg::Vec2 rot):pos(pos),rot(rot)
{

    std::cout<<"new Cam without visMat"<<std::endl;

}

Cam::~Cam()
{

}

void Cam::calcVisMat(const osg::Vec3Array &observationPoints)
{
     visMat.clear();
 /*   osg::Matrix T={1,0,0,0,
                   0,1,0,0,
                   0,0,1,0,
                   -pos.x(),-pos.y(),-pos.z(),1};

    osg::Matrix zRot={cos(osg::DegreesToRadians(rot.x())),-sin(osg::DegreesToRadians(rot.x())),0,0,
                      sin(osg::DegreesToRadians(rot.x())), cos(osg::DegreesToRadians(rot.x())),0,0,
                       0, 0, 1, 0,
                       0, 0, 0, 1
                      };

    osg::Matrix yRot = {cos(osg::DegreesToRadians(rot.y())), 0, sin(osg::DegreesToRadians(rot.y())),0,
                        0, 1, 0, 0,
                        -sin(osg::DegreesToRadians(rot.y())), 0, cos(osg::DegreesToRadians(rot.y())), 0,
                        0, 0, 0, 1
                       };
*/
//    osg::Matrix T;
//    T.makeTranslate(-pos);
    osg::Matrix T = osg::Matrix::translate(-pos);
    osg::Matrix zRot = osg::Matrix::rotate(osg::DegreesToRadians(-rot.x()), osg::Z_AXIS);
    osg::Matrix yRot = osg::Matrix::rotate(osg::DegreesToRadians(-rot.y()), osg::Y_AXIS);
    for(const auto& p : observationPoints)
    {

        auto newPoint = p*T*zRot*yRot;
        //auto newPoint1 =p*T1*zRot1*yRot1;
        std::cout<<"NewPoint:"<<(float)newPoint.x()<<", "<<(float)newPoint.y()<<", "<<(float)newPoint.z()<<std::endl;

        if((newPoint.x()<=Cam::depthView ) && (newPoint.x()>=0) &&
           (std::abs(newPoint.y()) <= Cam::imgWidth/2 * newPoint.x()/Cam::depthView) &&
           (std::abs(newPoint.z())<=Cam::imgHeight/2 * newPoint.x()/Cam::depthView))
        {
            visMat.push_back(1);
        }
        else
            visMat.push_back(0);

    }

    std::cout<<"visMat: ";
    for(auto x: visMat )
        std::cout <<x<<" ";
    std::cout <<"\n"<<std::endl;

}

CamDrawable::CamDrawable(const osg::Vec3 pos,const osg::Vec2 rot):Cam(pos,rot) //call Cam Constructor
{
    fprintf(stderr, "new CamDrawable from Point\n");
    camGeode = plotCam();
    transMat=new osg::MatrixTransform();
    osg::Matrix m;
    m.setTrans(pos.x(),pos.y(),pos.z());
    rotMat = new osg::MatrixTransform();
    osg::Matrix r;
    osg::Quat yRot, zRot;

    zRot.makeRotate(osg::DegreesToRadians((float)rot.x()), osg::Z_AXIS);
    yRot.makeRotate(osg::DegreesToRadians((float)rot.y()), osg::Y_AXIS);

    // concatenate the 2 into a resulting quat
    osg::Quat fullRot = zRot * yRot;
    r.setRotate(fullRot);
    rotMat->setMatrix(r);
    transMat->setMatrix(m);
    //OpenGL first rotate than translate
    rotMat->addChild(camGeode);
    transMat->addChild(rotMat);
    cover->getObjectsRoot()->addChild(transMat);

   /* revolution =new osg::PositionAttitudeTransform();
    revolution->setUpdateCallback( new RotationCallback());
    revolution->addChild(camGeode);
    cover->getObjectsRoot()->addChild(revolution);
    */
}

/*
CamDrawable::CamDrawable(Cam cam):Cam(cam.pos,cam.rot)
{
    std::cout<<"new CamDrawable from Cam"<<std::endl;
    camGeode = plotCam();
    transMat=new osg::MatrixTransform();
    osg::Matrix m;
    m.setTrans(pos.x(),pos.y(),pos.z());
    rotMat = new osg::MatrixTransform();
    osg::Matrix r;
    osg::Quat xRot, zRot;

    xRot.makeRotate(osg::DegreesToRadians((float)rot.x()), osg::X_AXIS);
    zRot.makeRotate(osg::DegreesToRadians((float)rot.y()), osg::Z_AXIS);
    // concatenate the 2 into a resulting quat
    osg::Quat fullRot = xRot * zRot;
    r.setRotate(fullRot);
    rotMat->setMatrix(r);
    transMat->setMatrix(m);
    //OpenGL first rotate than translate
    rotMat->addChild(camGeode);
    transMat->addChild(rotMat);
    cover->getObjectsRoot()->addChild(transMat);

    revolution =new osg::PositionAttitudeTransform();
    revolution->setUpdateCallback( new RotationCallback());
    revolution->addChild(camGeode);
    cover->getObjectsRoot()->addChild(revolution);

}
*/

CamDrawable::~CamDrawable()
{

}

osg::Geode* CamDrawable::plotCam()
{
    // The Drawable geometry is held under Geode objects.
    osg::Geode* geode = new osg::Geode();
    geode->setName("Cam");
    osg::Geometry* geom = new osg::Geometry();
    osg::StateSet *stateset = geode->getOrCreateStateSet();
    //necessary for dynamic redraw (command:dirty)
    geom->setDataVariance(osg::Object::DataVariance::DYNAMIC) ;
    geom->setUseDisplayList(false);
    geom->setUseVertexBufferObjects(true);

   // stateset->setRenderingHint(osg::StateSet::TRANSPARENT_BIN);
   // stateset->setAttributeAndModes(new osg::BlendFunc(GL_SRC_ALPHA ,GL_ONE_MINUS_SRC_ALPHA), osg::StateAttribute::ON);
    // Associate the Geometry with the Geode.
    geode->addDrawable(geom);
    geode->getStateSet()->setRenderingHint(osg::StateSet::TRANSPARENT_BIN);
    // Declare an array of vertices to create a simple pyramid.
    verts = new osg::Vec3Array;
/*  verts->push_back( osg::Vec3(-Cam::imgWidth, -Cam::imgHeight, -Cam::depthView ) ); // 0 left  front base
    verts->push_back( osg::Vec3( Cam::imgWidth, -Cam::imgHeight, -Cam::depthView ) ); // 1 right front base
    verts->push_back( osg::Vec3( Cam::imgWidth,  Cam::imgHeight, -Cam::depthView ) ); // 2 right back  base
    verts->push_back( osg::Vec3(-Cam::imgWidth,  Cam::imgHeight, -Cam::depthView ) ); // 3 left  back  base
*/
    verts->push_back( osg::Vec3(Cam::depthView, -Cam::imgWidth/2, Cam::imgHeight/2 ) ); // 0 upper  front base
    verts->push_back( osg::Vec3(Cam::depthView, -Cam::imgWidth/2,-Cam::imgHeight/2 ) ); // 1 lower front base
    verts->push_back( osg::Vec3(Cam::depthView,  Cam::imgWidth/2,-Cam::imgHeight/2 ) ); // 3 lower  back  base
    verts->push_back( osg::Vec3(Cam::depthView,  Cam::imgWidth/2, Cam::imgHeight/2 ) ); // 2 upper back  base
    verts->push_back( osg::Vec3( 0,  0,  0) ); // 4 peak


    // Associate this set of vertices with the Geometry.
    geom->setVertexArray(verts);

    // Next, create primitive sets and add them to the Geometry.
    // Each primitive set represents one face of the pyramid.
    // 0 base
    osg::DrawElementsUInt* face =
       new osg::DrawElementsUInt(osg::PrimitiveSet::QUADS, 0);
    face->push_back(3);
    face->push_back(2);
    face->push_back(1);
    face->push_back(0);
    geom->addPrimitiveSet(face);
    // 1 left face
    face = new osg::DrawElementsUInt(osg::PrimitiveSet::TRIANGLES, 0);
    face->push_back(3);
    face->push_back(0);
    face->push_back(4);
    geom->addPrimitiveSet(face);
    // 2 right face
    face = new osg::DrawElementsUInt(osg::PrimitiveSet::TRIANGLES, 0);
    face->push_back(1);
    face->push_back(2);
    face->push_back(4);
    geom->addPrimitiveSet(face);
    // 3 front face
    face = new osg::DrawElementsUInt(osg::PrimitiveSet::TRIANGLES, 0);
    face->push_back(0);
    face->push_back(1);
    face->push_back(4);
    geom->addPrimitiveSet(face);
    // 4 back face
    face = new osg::DrawElementsUInt(osg::PrimitiveSet::TRIANGLES, 0);
    face->push_back(2);
    face->push_back(3);
    face->push_back(4);
    geom->addPrimitiveSet(face);

    //Create normals
    osg::Vec3Array* normals = new osg::Vec3Array();
    normals->push_back(osg::Vec3(-1.f ,-1.f, 0.f)); //left front
    normals->push_back(osg::Vec3(1.f ,-1.f, 0.f)); //right front
    normals->push_back(osg::Vec3(1.f ,1.f, 0.f));//right back
    normals->push_back(osg::Vec3(-1.f ,1.f, 0.f));//left back
    normals->push_back(osg::Vec3(0.f ,0.f, 1.f));//peak
    geom->setNormalArray(normals);
    geom->setNormalBinding(osg::Geometry::BIND_PER_PRIMITIVE_SET);

    //create Materal
    osg::Material *material = new osg::Material;
    material->setColorMode(osg::Material::AMBIENT_AND_DIFFUSE);
    material->setDiffuse(osg::Material::FRONT_AND_BACK, osg::Vec4(1.0f, 0.2f, 0.2f, 1.0f));
    material->setAmbient(osg::Material::FRONT_AND_BACK, osg::Vec4(0.1f, 0.1f, 0.1f, 1.0f));
    material->setSpecular(osg::Material::FRONT_AND_BACK, osg::Vec4(1.0, 1.0, 1.0, 1.0));
    material->setShininess(osg::Material::FRONT_AND_BACK, 25.0);
    material->setTransparency(osg::Material::FRONT,0);
    stateset->setAttributeAndModes(material);
    stateset->setNestRenderBins(false);

    // Create a separate color for each face.
    osg::Vec4Array* colors = new osg::Vec4Array;
    colors->push_back( osg::Vec4(1.0f, 1.0f, 0.0f, 0.1f) ); // yellow  - base
    colors->push_back( osg::Vec4(0.0f, 1.0f, 1.0f, 0.1f) ); // cyan    - left
    colors->push_back( osg::Vec4(0.0f, 1.0f, 1.0f, 0.1f) ); // cyan    - right
    colors->push_back( osg::Vec4(1.0f, 0.0f, 1.0f, 0.1f) ); // magenta - front
    colors->push_back( osg::Vec4(1.0f, 0.0f, 1.0f, 0.1f) ); // magenta - back
    // The next step is to associate the array of colors with the geometry.
    // Assign the color indices created above to the geometry and set the
    // binding mode to _PER_PRIMITIVE_SET.
    geom->setColorArray(colors);
    geom->setColorBinding(osg::Geometry::BIND_PER_PRIMITIVE_SET);
    // return the geode as the root of this geometry.
    return geode;
}

void CamDrawable::updateFOV(float value)
{
    Cam::fov = value;
    Cam::imgWidth = 2*depthView*std::tan(Cam::fov/2*osg::PI/180);
    Cam::imgHeight = Cam::imgWidth/(Cam::imgWidthPixel/Cam::imgHeightPixel);
    verts->resize(0);
    verts->push_back( osg::Vec3(Cam::depthView, -Cam::imgWidth/2, Cam::imgHeight/2 ) ); // 0 upper  front base
    verts->push_back( osg::Vec3(Cam::depthView, -Cam::imgWidth/2,-Cam::imgHeight/2 ) ); // 1 lower front base
    verts->push_back( osg::Vec3(Cam::depthView,  Cam::imgWidth/2,-Cam::imgHeight/2 ) ); // 3 lower  back  base
    verts->push_back( osg::Vec3(Cam::depthView,  Cam::imgWidth/2, Cam::imgHeight/2 ) ); // 2 upper back  base
    verts->push_back( osg::Vec3( 0,  0,  0) ); // 4 peak
    verts->dirty();



}

void CamDrawable::updateVisibility(float value)
{
    Cam::depthView = value;
    Cam::imgWidth = 2*depthView*std::tan(Cam::fov/2*osg::PI/180);
    Cam::imgHeight = Cam::imgWidth/(Cam::imgWidthPixel/Cam::imgHeightPixel);
    verts->resize(0);
    verts->push_back( osg::Vec3(Cam::depthView, -Cam::imgWidth/2, Cam::imgHeight/2 ) ); // 0 upper  front base
    verts->push_back( osg::Vec3(Cam::depthView, -Cam::imgWidth/2,-Cam::imgHeight/2 ) ); // 1 lower front base
    verts->push_back( osg::Vec3(Cam::depthView,  Cam::imgWidth/2,-Cam::imgHeight/2 ) ); // 3 lower  back  base
    verts->push_back( osg::Vec3(Cam::depthView,  Cam::imgWidth/2, Cam::imgHeight/2 ) ); // 2 upper back  base
    verts->push_back( osg::Vec3( 0,  0,  0) ); // 4 peak
    verts->dirty();



}

/*void Cam::isPointVisible()
{
osg::Matrix T(1,0,0,0,
               0,1,0,0,
               0,0,1,0,
               -pos.x(),-pos.y(),-pos.z(),1);
osg::Matrix Rz(std::cos(rot.x()),-std::sin(rot.x()),0,0,
               std::sin(rot.x()), std::cos(rot.x()),0,0,
               0, 0, 1, 0,
               0, 0, 0, 1);
osg::Matrix Ry(std::cos(rot.y()), 0, std::sin(rot.y()),0,
               0, 1, 0, 0,
               -std::sin(rot.y()), 0, std::cos(rot.y()), 0,
               0, 0, 0, 1);
}

*/
