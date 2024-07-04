/* This file is part of COVISE.
:
   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/****************************************************************************\ 
 **                                                            (C)2024 RRZK  **
 **                                                                          **
 ** Description: OrientationIndicator OpenCOVER Plugin (draws axis showing   **
 **              the orientation of the loaded model)                        **
 **                                                                          **
 ** Author: D.Wickeroth                                                      **
 **                                                                          **
 ** History:  								                                 **
 ** March 2024  v1	    				       		                         **
 **                                                                          **
 **                                                                          **
\****************************************************************************/

#include "OrientationIndicator.h"
#include <cover/coVRPluginSupport.h>
#include <config/CoviseConfig.h>
#include <osg/ShapeDrawable>
#include <osg/Material>
#include <osg/Vec4>


#include <iostream>

using namespace opencover;
OrientationIndicator::OrientationIndicator()
: coVRPlugin(COVER_PLUGIN_NAME)
{
    fprintf(stderr, "OrientationIndicator\n");

    //get offset from config. Determines where in world coordinates the indicator will be shown.
    float x = covise::coCoviseConfig::getFloat("x", "COVER.Plugin.OrientationIndicator.Position", -700.0);
    float y = covise::coCoviseConfig::getFloat("y", "COVER.Plugin.OrientationIndicator.Position", 0.0);
    float z = covise::coCoviseConfig::getFloat("z", "COVER.Plugin.OrientationIndicator.Position", -300.0);
    oi_offset = osg::Vec3f(x,y,z);

    //The Zoologists asked for this
    bool invert_X = covise::coCoviseConfig::isOn("COVER.Plugin.OrientationIndicator.InvertX", false);
    bool invert_Y = covise::coCoviseConfig::isOn("COVER.Plugin.OrientationIndicator.InvertY", false);
    bool invert_Z = covise::coCoviseConfig::isOn("COVER.Plugin.OrientationIndicator.InvertZ", false);    

    //get the size of the arrows from config
    float cylHeight = covise::coCoviseConfig::getFloat("height", "COVER.Plugin.OrientationIndicator.Cylinder", 200.0f);
    float cylRadius = covise::coCoviseConfig::getFloat("radius", "COVER.Plugin.OrientationIndicator.Cylinder", 18.03f);
    float coneHeight = covise::coCoviseConfig::getFloat("height", "COVER.Plugin.OrientationIndicator.Cone", 76.3f);
    float coneRadius = covise::coCoviseConfig::getFloat("radius", "COVER.Plugin.OrientationIndicator.Cone", 32.3f);

    /**   X    **/
    //create red material for the x-direction
    osg::Material *xMaterial = new osg::Material();
    xMaterial->setDiffuse(osg::Material::FRONT_AND_BACK, osg::Vec4(1,0,0,1));

    //create quat for x-direction
    float radians = 1.5707964f;
    if(invert_X) radians = -1.5707964f;
    osg::Quat xQuat(radians ,osg::Vec3f(0,1,0));    

    //create cylinder and cone to form green arrow in x-direction
    osg::Vec3f center = osg::Vec3f(cylHeight / 2.0f,0,0);
    if(invert_X) center = osg::Vec3f(-cylHeight / 2.0f,0,0);
    osg::Cylinder *xCylinder = new osg::Cylinder(center, cylRadius, cylHeight);
    xCylinder->setRotation(xQuat);
    xCylinder->setName("xCylinder");
    osg::ShapeDrawable *xCylinderDrawable = new osg::ShapeDrawable(xCylinder);
    xCylinderDrawable->setName("xCylinderDrawable");
    xCylinderDrawable->getOrCreateStateSet()->setAttribute(xMaterial, osg::StateAttribute::OVERRIDE);

    center = osg::Vec3f(cylHeight + (coneHeight / 4.0f),0,0);
    if(invert_X) center = osg::Vec3f(-cylHeight - (coneHeight / 4.0f),0,0);
    osg::Cone *xCone = new osg::Cone(center, coneRadius, coneHeight);
    xCone->setRotation(xQuat);
    xCone->setName("xCone");
    osg::ShapeDrawable *xConeDrawable = new osg::ShapeDrawable(xCone);
    xConeDrawable->setName("xConeDrawable");
    xConeDrawable->getOrCreateStateSet()->setAttribute(xMaterial, osg::StateAttribute::OVERRIDE);

    /**   Y    **/
    //create green material for the y-direction    
    osg::Material *yMaterial = new osg::Material();
    yMaterial->setDiffuse(osg::Material::FRONT_AND_BACK, osg::Vec4(0,1,0,1));

    //create quat for y-direction
    radians = -1.5707964f;
    if (invert_Y) radians = 1.5707964f;
    osg::Quat yQuat(radians, osg::Vec3f(1,0,0));

    //create cylinder and cone to form green arrow in y-direction
    center = osg::Vec3f(0, cylHeight / 2.0f, 0);
    if (invert_Y) center = osg::Vec3f(0, -cylHeight / 2.0f, 0);
    osg::Cylinder *yCylinder = new osg::Cylinder(center, cylRadius, cylHeight);
    yCylinder->setRotation(yQuat);
    yCylinder->setName("yCylinder");
    osg::ShapeDrawable *yCylinderDrawable = new osg::ShapeDrawable(yCylinder);
    yCylinderDrawable->setName("yCylinderDrawable");
    yCylinderDrawable->getOrCreateStateSet()->setAttribute(yMaterial, osg::StateAttribute::OVERRIDE);

    center = osg::Vec3(0,cylHeight + (coneHeight / 4.0f), 0);
    if (invert_Y) center = osg::Vec3(0, -cylHeight - (coneHeight / 4.0f), 0);
    osg::Cone *yCone = new osg::Cone(center, coneRadius, coneHeight);
    yCone->setRotation(yQuat);
    yCone->setName("yCone");
    osg::ShapeDrawable *yConeDrawable = new osg::ShapeDrawable(yCone);
    yConeDrawable->setName("yConeDrawable");
    yConeDrawable->getOrCreateStateSet()->setAttribute(yMaterial, osg::StateAttribute::OVERRIDE);

    /**   Z    **/
    //create blue material for the z-direction
    osg::Material *zMaterial = new osg::Material();
    zMaterial->setDiffuse(osg::Material::FRONT_AND_BACK, osg::Vec4(0,0,1,1));

    radians = 0;
    if(invert_Z) radians = 3.1415927;    
    osg::Quat zQuat(radians, osg::Vec3(1,0,0));

    //create cylinder and cone to form blue arrow in z-direction
    center = osg::Vec3f(0, 0, cylHeight / 2.0f);
    if (invert_Z) center = osg::Vec3f(0, 0, -cylHeight / 2.0f);
    osg::Cylinder *zCylinder = new osg::Cylinder(center, cylRadius, cylHeight);
    zCylinder->setRotation(zQuat);
    zCylinder->setName("zCylinder");
    osg::ShapeDrawable *zCylinderDrawable = new osg::ShapeDrawable(zCylinder);
    zCylinderDrawable->setName("zCylinderDrawable");
    zCylinderDrawable->getOrCreateStateSet()->setAttribute(zMaterial, osg::StateAttribute::OVERRIDE);

    center = osg::Vec3(0, 0, cylHeight + (coneHeight / 4.0f));
    if (invert_Z) center = osg::Vec3(0, 0, -cylHeight - (coneHeight / 4.0f));
    osg::Cone *zCone = new osg::Cone(center, coneRadius, coneHeight);
    zCone->setRotation(zQuat);
    zCone->setName("zCone");
    osg::ShapeDrawable *zConeDrawable = new osg::ShapeDrawable(zCone);
    zConeDrawable->setName("zConeDrawable");
    zConeDrawable->getOrCreateStateSet()->setAttribute(zMaterial, osg::StateAttribute::OVERRIDE);

    // Create the geode and add the drawables to it
    oi_Geode = new osg::Geode();
    oi_Geode->setName("OrientationIndicator");
    oi_Geode->addDrawable(xCylinderDrawable);
    oi_Geode->addDrawable(xConeDrawable);
    oi_Geode->addDrawable(yCylinderDrawable);
    oi_Geode->addDrawable(yConeDrawable);
    oi_Geode->addDrawable(zCylinderDrawable);
    oi_Geode->addDrawable(zConeDrawable);

    //create the transformation, attach the geode to the transformation and attach the transforation to the objects root.
    oi_Trans = new osg::MatrixTransform();
    oi_Trans->setName("oi_Trans");
    oi_Trans->addChild(oi_Geode.get());
    cover->getObjectsRoot()->addChild(oi_Trans.get());
}

bool OrientationIndicator::destroy()
{
    cover->getObjectsRoot()->removeChild(oi_Trans.get());
    return true;
}

void OrientationIndicator::preFrame()
{
    //get the inverted BaseMat as a starting point. Makes the indicator be always in the center.
    osg::Matrix oi_mat = cover->getInvBaseMat();
    
    //apply an offset, so that the indicator is always in the bottom left or whereever you want it.
    oi_mat.preMultTranslate(oi_offset);

    //apply the rotation part of the baseMat, so that the orientation of the indicator reflects the orientation of the objects
    osg::Matrix bM = cover->getBaseMat();
    osg::Vec3f bM_trans;
    osg::Quat bM_rot;
    osg::Vec3f bM_scale;
    osg::Quat bM_so;
    bM.decompose(bM_trans, bM_rot, bM_scale, bM_so);
    oi_mat.preMultRotate(bM_rot);

    //apply the matrix to the transform node
    oi_Trans->setMatrix(oi_mat);
}

// this is called if the plugin is removed at runtime
OrientationIndicator::~OrientationIndicator()
{
    fprintf(stderr, " Goodbye OrientationIndicator \n");
}

COVERPLUGIN(OrientationIndicator)
