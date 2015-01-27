/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/**************************************************************************
** TrafficSimulation
**   Frank Naegele 2010
**   <mail@f-naegele.de> <frank.naegele@porsche.de>
**   2/11/2010
**
** This class can be used to visualize Porsche radars with cones.
**
**************************************************************************/

#include "radarcones.hpp"

#include "osg/Geometry"
#include "osg/CullFace"
#include "../VehicleManager.h"

#include "math.h"

#define MAXRADARS 8

/** Constructor
*
*/
RadarCones::RadarCones(HumanVehicle *humanVehicle)
    : humanVehicle_(humanVehicle)
    , humanVehicleTransform_(NULL)
    , cones_(NULL)

{
    std::cerr << "Init radar cones!";
    init();
}

void
RadarCones::init()
{
    // Lots of safety precautions //
    //
    if (humanVehicleTransform_)
    {
        return; // already done
    }

    if (!humanVehicle_)
    {
        std::cerr << "FFZBroadcaster initialized with NULL HumanVehicle! Did you use the right constructor?" << std::endl;
        return;
    }

    VehicleGeometry *vehicleGeometry = humanVehicle_->getVehicleGeometry();
    if (!vehicleGeometry)
        return;

    HumanVehicleGeometry *humanVehicleGeometry = dynamic_cast<HumanVehicleGeometry *>(vehicleGeometry);
    if (!humanVehicleGeometry)
        return;

    osg::Node *humanVehicleNode = humanVehicleGeometry->getVehicleNode();
    if (!humanVehicleNode)
        return;

    humanVehicleTransform_ = humanVehicleNode->asTransform();
    if (!humanVehicleTransform_)
        return;

    std::cout << "init Radar Cones" << std::endl;

    // StateSet for the cones //
    //
    osg::CullFace *cull = new osg::CullFace();
    cull->setMode(osg::CullFace::BACK);

    coneStateSet_ = new osg::StateSet();
    coneStateSet_->setMode(GL_LIGHTING, osg::StateAttribute::OFF | osg::StateAttribute::OVERRIDE);
    coneStateSet_->setMode(GL_BLEND, osg::StateAttribute::ON | osg::StateAttribute::OVERRIDE);
    coneStateSet_->setAttributeAndModes(cull, osg::StateAttribute::ON | osg::StateAttribute::OVERRIDE);
    coneStateSet_->setRenderingHint(osg::StateSet::TRANSPARENT_BIN);
    coneStateSet_->setNestRenderBins(false);

    // Create Pyramid/Cone //
    //
    cones_ = new osg::Group();
    cones_->setName("RadarCones");

    coneGeometries_.resize(MAXRADARS, NULL); // resize vector

#if 1
    for (int i = 0; i < MAXRADARS; ++i)
    {

        // Geometry //
        //
        osg::Geometry *coneGeometry = new osg::Geometry();
        coneGeometries_[i] = coneGeometry;
        coneGeometry->setName("RadarCone");
        coneGeometry->setDataVariance(osg::Object::DYNAMIC);
        coneGeometry->setUseDisplayList(false);

        // Colors //
        //
        osg::Vec4Array *coneColors = new osg::Vec4Array(1);
        (*coneColors)[0] = osg::Vec4(1.0f, 0.0f, 0.0f, 0.5f); // default
        coneGeometry->setColorArray(coneColors);
        coneGeometry->setColorBinding(osg::Geometry::BIND_OVERALL);

        // Vertices in VRML coordinates //
        //
        osg::Vec3Array *coneVerts = new osg::Vec3Array(8);
        coneGeometry->setVertexArray(coneVerts);

        // Faces //
        //
        osg::DrawElementsUInt *coneQuads = new osg::DrawElementsUInt(osg::PrimitiveSet::QUADS, 0);
        coneQuads->push_back(0);
        coneQuads->push_back(1);
        coneQuads->push_back(2);
        coneQuads->push_back(3);
        coneQuads->push_back(0);
        coneQuads->push_back(4);
        coneQuads->push_back(5);
        coneQuads->push_back(1);
        coneQuads->push_back(0);
        coneQuads->push_back(3);
        coneQuads->push_back(7);
        coneQuads->push_back(4);
        coneQuads->push_back(4);
        coneQuads->push_back(7);
        coneQuads->push_back(6);
        coneQuads->push_back(5);
        coneQuads->push_back(1);
        coneQuads->push_back(5);
        coneQuads->push_back(6);
        coneQuads->push_back(2);
        coneQuads->push_back(2);
        coneQuads->push_back(6);
        coneQuads->push_back(7);
        coneQuads->push_back(3);

        coneGeometry->addPrimitiveSet(coneQuads);

        // State Set //
        //
        coneGeometry->setStateSet(coneStateSet_);

        // Geode //
        //
        osg::Geode *cone = new osg::Geode();
        cone->addDrawable(coneGeometry);
        cones_->addChild(cone);
    }

    // Add Cones //
    //
    humanVehicleTransform_->addChild(cones_);

#else
    osg::Geode *cone = new osg::Geode();

    osg::Geometry *coneGeometry = new osg::Geometry();
    coneGeometry->setDataVariance(osg::Object::DYNAMIC);
    coneGeometry->setUseDisplayList(false);

    cone->addDrawable(coneGeometry);
    cones_->addChild(cone);

    // Coordinates //
    //
    //	float dSpaceX = 1.0;
    //	float dSpaceY = 0.0;
    //	float dSpaceZ = 0.5;
    //
    //	float dSpaceRoll = 0.0;
    //	float dSpacePitch = 3.6;
    //	float dSpaceYaw = -9.0;

    float minRange = 0.5;
    float maxRange = 8.0;

    float azimuth = 4.5 * 2.0 * M_PI / 360.0;
    float elevation = 3.0 * 2.0 * M_PI / 360.0;

    osg::Vec3Array *coneVerts = new osg::Vec3Array;

    double aMin = tan(azimuth) * minRange;
    double aMax = tan(azimuth) * maxRange;
    double eMin = tan(elevation) * minRange;
    double eMax = tan(elevation) * maxRange;

    // Vertices in VRML coordinates //
    //
    coneVerts->push_back(osg::Vec3(aMin, eMin, -minRange));
    coneVerts->push_back(osg::Vec3(-aMin, eMin, -minRange));
    coneVerts->push_back(osg::Vec3(-aMin, -eMin, -minRange));
    coneVerts->push_back(osg::Vec3(aMin, -eMin, -minRange));

    coneVerts->push_back(osg::Vec3(aMax, eMax, -maxRange));
    coneVerts->push_back(osg::Vec3(-aMax, eMax, -maxRange));
    coneVerts->push_back(osg::Vec3(-aMax, -eMax, -maxRange));
    coneVerts->push_back(osg::Vec3(aMax, -eMax, -maxRange));

    coneGeometry->setVertexArray(coneVerts);

    // Faces //
    //
    osg::DrawElementsUInt *coneQuads = new osg::DrawElementsUInt(osg::PrimitiveSet::QUADS, 0);
    coneQuads->push_back(0);
    coneQuads->push_back(1);
    coneQuads->push_back(2);
    coneQuads->push_back(3);
    coneQuads->push_back(0);
    coneQuads->push_back(4);
    coneQuads->push_back(5);
    coneQuads->push_back(1);
    coneQuads->push_back(0);
    coneQuads->push_back(3);
    coneQuads->push_back(7);
    coneQuads->push_back(4);
    coneQuads->push_back(4);
    coneQuads->push_back(7);
    coneQuads->push_back(6);
    coneQuads->push_back(5);
    coneQuads->push_back(1);
    coneQuads->push_back(5);
    coneQuads->push_back(6);
    coneQuads->push_back(2);
    coneQuads->push_back(2);
    coneQuads->push_back(6);
    coneQuads->push_back(7);
    coneQuads->push_back(3);

    coneGeometry->addPrimitiveSet(coneQuads);

#if 0
	float rWidth	= 10.0f;
	float rHeight	= 2.0f;
	float maxRange	= 30.0f;

	osg::Vec3Array* coneVerts = new osg::Vec3Array;
	coneVerts->push_back(osg::Vec3(0, 0, 0));
//	coneVerts->push_back(osg::Vec3(maxRange, -rHeight, -rWidth));
//	coneVerts->push_back(osg::Vec3(maxRange, -rHeight,  rWidth));
//	coneVerts->push_back(osg::Vec3(maxRange,  rHeight,  rWidth));
//	coneVerts->push_back(osg::Vec3(maxRange,  rHeight, -rWidth));


	// VRML coordinate system, x right, y up, z back //
	//
	coneVerts->push_back(osg::Vec3(-rWidth, -rHeight, -maxRange));
	coneVerts->push_back(osg::Vec3( rWidth, -rHeight, -maxRange));
	coneVerts->push_back(osg::Vec3( rWidth,  rHeight, -maxRange));
	coneVerts->push_back(osg::Vec3(-rWidth,  rHeight, -maxRange));

	coneGeometry->setVertexArray(coneVerts);


//	osg::DrawElementsUInt* side = new osg::DrawElementsUInt(osg::PrimitiveSet::QUADS, 0);
//	side->push_back(3); side->push_back(2); side->push_back(1); side->push_back(0);
//	coneGeometry->addPrimitiveSet(side);

	osg::DrawElementsUInt* side0 = new osg::DrawElementsUInt(osg::PrimitiveSet::TRIANGLES, 0);
//	side0->push_back(0); side0->push_back(2); side0->push_back(3);
//	side0->push_back(0); side0->push_back(1); side0->push_back(4);
//	coneGeometry->addPrimitiveSet(side0);

//	osg::DrawElementsUInt* side1 = new osg::DrawElementsUInt(osg::PrimitiveSet::TRIANGLES, 0);
//	side1->push_back(0); side1->push_back(3); side1->push_back(4);
//	side1->push_back(1); side1->push_back(2); side1->push_back(4);
//	coneGeometry->addPrimitiveSet(side1);

//	osg::DrawElementsUInt* side2 = new osg::DrawElementsUInt(osg::PrimitiveSet::TRIANGLES, 0);
//	side2->push_back(0); side2->push_back(2); side2->push_back(1);
//	side2->push_back(2); side2->push_back(3); side2->push_back(4);
//	coneGeometry->addPrimitiveSet(side2);

//	osg::DrawElementsUInt* side3 = new osg::DrawElementsUInt(osg::PrimitiveSet::TRIANGLES, 0);
//	side3->push_back(0); side3->push_back(4); side3->push_back(1);
//	side3->push_back(3); side3->push_back(0); side3->push_back(4);
//	coneGeometry->addPrimitiveSet(side3);

	side0->push_back(0); side0->push_back(2); side0->push_back(3);
	side0->push_back(0); side0->push_back(3); side0->push_back(4);
	side0->push_back(0); side0->push_back(2); side0->push_back(1);
	side0->push_back(0); side0->push_back(4); side0->push_back(1);
	coneGeometry->addPrimitiveSet(side0);
#endif
    osg::Vec4Array *coneColors = new osg::Vec4Array;
    //	osg::Vec4Array* coneColors = new osg::Vec4Array(1);
    coneColors->push_back(osg::Vec4(1.0f, 0.0f, 0.0f, 0.5f));
    //	(*coneColors)[0] = osg::Vec4(1.0f, 0.0f, 0.0f, 0.5f);
    coneGeometry->setColorArray(coneColors);
    coneGeometry->setColorBinding(osg::Geometry::BIND_OVERALL);

    //	osg::Vec3Array* coneNormals = new osg::Vec3Array;
    //	coneNormals->push_back(osg::Vec3(1.0f, -1.0f, 0.0f));
    //	coneGeometry->setNormalArray(coneNormals);
    //	coneGeometry->setNormalBinding(osg::Geometry::BIND_OVERALL);

    coneGeometry->setStateSet(coneStateSet_);
#endif
}

void
RadarCones::updateCone(int i)
{
    RadarConeData coneData = conesData_->cones[i];

    // Colors //
    //
    float r = ((coneData.color & 0xff000000) >> 24) / 255.0;
    float g = ((coneData.color & 0x00ff0000) >> 16) / 255.0;
    float b = ((coneData.color & 0x0000ff00) >> 8) / 255.0;
    float a = ((coneData.color & 0x000000ff) >> 0) / 255.0;
    //std::cout << coneData.color << ", " << r << " " << g << " " << b << " " << a << std::endl;

    osg::Vec4Array *coneColors = dynamic_cast<osg::Vec4Array *>(coneGeometries_[i]->getColorArray());
    (*coneColors)[0] = osg::Vec4(r, g, b, a);

    // Transformation //
    //
    float dSpaceX = coneData.installPosX;
    float dSpaceY = coneData.installPosY;
    float dSpaceZ = coneData.installPosZ;

    float dSpaceRoll = coneData.installRoll * 2.0 * M_PI / 360.0;
    float dSpacePitch = coneData.installPitch * 2.0 * M_PI / 360.0;
    float dSpaceYaw = coneData.installYaw * 2.0 * M_PI / 360.0;

    osg::Matrix trafo;
    trafo.makeTranslate(-dSpaceY, dSpaceZ, -dSpaceX);
    osg::Quat quat(
        dSpaceYaw, osg::Vec3(0, 1, 0),
        dSpacePitch, osg::Vec3(-1, 0, 0),
        dSpaceRoll, osg::Vec3(0, 0, -1));
    trafo.setRotate(quat);

    // Geometry //
    //
    float minRange = coneData.minRange;
    float maxRange = coneData.maxRange;

    float azimuth = coneData.azimuth * 2.0 * M_PI / 360.0;
    float elevation = coneData.elevation * 2.0 * M_PI / 360.0;

    double aMin = tan(azimuth) * minRange;
    double aMax = tan(azimuth) * maxRange;
    double eMin = tan(elevation) * minRange;
    double eMax = tan(elevation) * maxRange;

    osg::Vec3Array *coneVerts = dynamic_cast<osg::Vec3Array *>(coneGeometries_[i]->getVertexArray());
    (*coneVerts)[0] = trafo.preMult(osg::Vec3(aMin, eMin, -minRange));
    (*coneVerts)[1] = trafo.preMult(osg::Vec3(-aMin, eMin, -minRange));
    (*coneVerts)[2] = trafo.preMult(osg::Vec3(-aMin, -eMin, -minRange));
    (*coneVerts)[3] = trafo.preMult(osg::Vec3(aMin, -eMin, -minRange));
    (*coneVerts)[4] = trafo.preMult(osg::Vec3(aMax, eMax, -maxRange));
    (*coneVerts)[5] = trafo.preMult(osg::Vec3(-aMax, eMax, -maxRange));
    (*coneVerts)[6] = trafo.preMult(osg::Vec3(-aMax, -eMax, -maxRange));
    (*coneVerts)[7] = trafo.preMult(osg::Vec3(aMax, -eMax, -maxRange));
}

void
RadarCones::update(RadarConesData *data)
{
    conesData_ = data;

    if (!humanVehicleTransform_)
    {
        init();
    }
    else
    {
        int n = conesData_->nSensors;
        if (n > MAXRADARS)
        {
            n = MAXRADARS;
        }

        for (int i = 0; i < n; ++i)
        {
            updateCone(i);
        }

        // TODO: hide others!!!
    }
}
