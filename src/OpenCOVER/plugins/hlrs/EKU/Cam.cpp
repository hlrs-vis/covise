/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */
#include <iostream>

#include <Cam.h>

using namespace opencover;

Cam::Cam()
{
    fprintf(stderr, "new Cam\n");
    camGeode = createPyramid();
    //cover->getObjectsRoot()->addChild(camGeode);

    revolution =new osg::PositionAttitudeTransform();
    revolution->setUpdateCallback( new RotationCallback());
    revolution->addChild(camGeode);
    cover->getObjectsRoot()->addChild(revolution);
}

osg::Geode* Cam::createPyramid()
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
    // Associate the Geometry with the Geode.
    geode->addDrawable(geom);
    // Declare an array of vertices to create a simple pyramid.
    verts = new osg::Vec3Array;
    verts->push_back( osg::Vec3(-1.5f, -1.5f, -1.5f) ); // 0 left  front base
    verts->push_back( osg::Vec3( 1.5f, -1.5f, -1.5f) ); // 1 right front base
    verts->push_back( osg::Vec3( 1.5f,  1.5f, -1.5f) ); // 2 right back  base
    verts->push_back( osg::Vec3(-1.5f,  1.5f, -1.5f) ); // 3 left  back  base
    verts->push_back( osg::Vec3( 0.0f,  0.0f,  1.5f) ); // 4 peak


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
    stateset->setAttributeAndModes(material);
    stateset->setNestRenderBins(false);

    // Create a separate color for each face.
    osg::Vec4Array* colors = new osg::Vec4Array;
    colors->push_back( osg::Vec4(1.0f, 1.0f, 0.0f, 1.0f) ); // yellow  - base
    colors->push_back( osg::Vec4(0.0f, 1.0f, 1.0f, 1.0f) ); // cyan    - left
    colors->push_back( osg::Vec4(0.0f, 1.0f, 1.0f, 1.0f) ); // cyan    - right
    colors->push_back( osg::Vec4(1.0f, 0.0f, 1.0f, 1.0f) ); // magenta - front
    colors->push_back( osg::Vec4(1.0f, 0.0f, 1.0f, 1.5f) ); // magenta - back
    // The next step is to associate the array of colors with the geometry.
    // Assign the color indices created above to the geometry and set the
    // binding mode to _PER_PRIMITIVE_SET.
    geom->setColorArray(colors);
    geom->setColorBinding(osg::Geometry::BIND_PER_PRIMITIVE_SET);
    // return the geode as the root of this geometry.
    return geode;
}

void Cam::updateFOV(float value)
{
    verts->resize(0);
    verts->push_back( osg::Vec3(-value, -value, -1.5) ); // 0 left  front base
    verts->push_back( osg::Vec3( value, -value, -1.5) ); // 1 right front base
    verts->push_back( osg::Vec3( value,  value, -1.5) ); // 2 right back  base
    verts->push_back( osg::Vec3(-value,  value, -1.5) ); // 3 left  back  base
    verts->push_back( osg::Vec3( 0.0f,  0.0f,  1.5f) );  // 4 peak
    verts->dirty();

}

void Cam::updateVisibility(float value)
{
    verts->resize(0);
    verts->push_back( osg::Vec3(-1.5f, -1.5f, -value) ); // 0 left  front base
    verts->push_back( osg::Vec3( 1.5f, -1.5f, -value) ); // 1 right front base
    verts->push_back( osg::Vec3( 1.5f,  1.5f, -value) ); // 2 right back  base
    verts->push_back( osg::Vec3(-1.5f,  1.5f, -value) ); // 3 left  back  base
    verts->push_back( osg::Vec3( 0.0f,  0.0f,  1.5f) );  // 4 peak
    verts->dirty();
}


