/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/****************************************************************************\
**                                                            (C)2021 HLRS  **
**                                                                          **
** Description: Rhino Plugin (read Rhino (3dm) files)                                 **
**                                                                          **
**                                                                          **
** Author: U.Woessner		                                                **
**                                                                          **
** History:  								                                **
** Sep-21  v1	    				       		                            **
**                                                                          **
**                                                                          **
\****************************************************************************/

#include <cover/coVRPluginSupport.h>
#include <cover/coVRFileManager.h>
#include <cover/coVRMSController.h>
#include <PluginUtil/coLOD.h>
#include <config/CoviseConfig.h>
#include <cover/coVRShader.h>
#include <codecvt>
#include <string>


#include <cover/RenderObject.h>
#include <cover/VRRegisterSceneGraph.h>
#include <osg/LOD>

#include <osg/GL>
#include <osg/Group>
#include <osg/MatrixTransform>
#include <osg/TexGen>
#include <osg/TexEnv>
#include <osg/TexMat>
#include <osg/TexEnvCombine>
#include <osg/Texture>
#include <osg/TextureCubeMap>
#include <osg/Texture2D>
#include <osg/Geode>
#include <osg/Switch>
#include <osg/Geometry>
#include <osg/PrimitiveSet>
#include <osg/Shape>
#include <osg/ShapeDrawable>
#include <osg/CullFace>
#include <osg/BlendFunc>
#include <osg/Light>
#include <osg/LightSource>
#include <osg/Depth>
#include <osg/Fog>
#include <osg/AlphaFunc>
#include <osg/ColorMask>
#include <osgDB/ReadFile>

#include <osgUtil/SmoothingVisitor>

#include <osgUtil/Optimizer>

#include "3dmPlugin.h"


#include <osgText/Font>
#include <osgText/Text>


#include <sys/types.h>
#include <string.h>



RhinoPlugin *RhinoPlugin::plugin = NULL;

static FileHandler handlers[] = {
    {NULL,
      RhinoPlugin::loadRhino,
      RhinoPlugin::unloadRhino,
      "Rhino"},
      {NULL,
      RhinoPlugin::loadRhino,
      RhinoPlugin::unloadRhino,
      "3dm"}
};

#include <osgUtil/SmoothingVisitor>
#include <osgUtil/DelaunayTriangulator>

const double TOLERANCE_EDGE = 1e-6;
const double TOLERANCE_FACE = 1e-6;



int RhinoPlugin::loadRhino(const char* filename, osg::Group* loadParent, const char*)
{
	return plugin->loadFile(filename, loadParent);
}

int RhinoPlugin::unloadFile(const std::string& fileName)
{
	for (const auto& s : switches)
	{
		if (s->getName() == fileName)
		{
			switches.remove(s);
			while(s->getParent(0))
				s->removeChild(s.get());
			return 1;
		}
	}
	return 0;
}

int RhinoPlugin::loadFile(const std::string&fileName, osg::Group *loadParent)
{
	if (loadParent == nullptr)
	{
		loadParent = RhinoRoot;
	}
	RhinoSwitch = new osg::Switch();
	loadParent->addChild(RhinoSwitch);
	RhinoSwitch->setName(fileName);
	switches.push_back(RhinoSwitch);
    Read3DM(fileName);
    RhinoSwitch->addChild(mRhinoNode);
    return 0;
}

int RhinoPlugin::unloadRhino(const char *filename, const char *)
{
	return plugin->unloadFile(filename);
}


RhinoPlugin::RhinoPlugin()
: coVRPlugin(COVER_PLUGIN_NAME)
{
   
    //scaleGeometry = coCoviseConfig::isOn("COVER.Plugin.Rhino.ScaleGeometry", true);
    //lodScale = coCoviseConfig::getFloat("COVER.Plugin.Rhino.LodScale", 2000.0f);
	RhinoRoot = new osg::MatrixTransform();
	RhinoRoot->setName("RhinoRoot");
	cover->getObjectsRoot()->addChild(RhinoRoot);
    plugin = this;
}

bool RhinoPlugin::init()
{
    for(int i=0;i<2;i++)
    coVRFileManager::instance()->registerFileHandler(&handlers[i]);
    return true;
}

// this is called if the plugin is removed at runtime
// which currently never happens
RhinoPlugin::~RhinoPlugin()
{
    for (int i = 0; i < 3; i++)
    coVRFileManager::instance()->unregisterFileHandler(&handlers[i]);
	cover->getObjectsRoot()->removeChild(RhinoRoot);
}

bool
RhinoPlugin::update()
{
	return false;
}



osg::Node* RhinoPlugin::GetRhinoModel()
{
    return mRhinoNode;
}

void RhinoPlugin::Read3DM(const std::string& theFileName)
{
    if (!mRhinoModel.Read(theFileName.c_str()))
    {
        return;
    }

    osg::Group* aRoot = new osg::Group();

    ONX_ModelComponentIterator it(mRhinoModel, ON_ModelComponent::Type::ModelGeometry);
    const ON_ModelComponent* model_component = nullptr;
    for (model_component = it.FirstComponent(); nullptr != model_component; model_component = it.NextComponent())
    {
        const ON_ModelGeometryComponent* model_geometry = ON_ModelGeometryComponent::Cast(model_component);
        if (nullptr != model_geometry)
        {
            const ON_3dmObjectAttributes* attributes = model_geometry->Attributes(nullptr);
            if (nullptr != attributes && !attributes->IsInstanceDefinitionObject())
            {
                const ON_Geometry* geometry = model_geometry->Geometry(nullptr);
                if (nullptr != geometry)
                {
                    const ON_Brep* aBrep = dynamic_cast<const ON_Brep*> (geometry);

                    if (aBrep)
                    {
                        aRoot->addChild(BuildBrep(aBrep));
                    }
                }
            }
        }
    }


    mRhinoNode = aRoot;
}

osg::Node* RhinoPlugin::BuildBrep(const ON_Brep* theBrep)
{
    osg::ref_ptr<osg::Group> aGroup = new osg::Group();

    //aGroup->addChild(BuildEdge(theBrep));

    for (int i = 0; i < theBrep->m_F.Count(); ++i)
    {
        ON_BrepFace* aFace = theBrep->Face(i);

        //aGroup->addChild(BuildWireFrameFace(aFace));
        osg::Node* n = BuildShadedFace(aFace);
        if (n != nullptr)
        {
            aGroup->addChild(n);
        }
    }

    //theBrep->Dump(ON_TextLog());

    return aGroup.release();
}

osg::Node* RhinoPlugin::BuildEdge(const ON_Brep* theBrep)
{
    osg::ref_ptr<osg::Geode> aGeode = new osg::Geode();

    for (int i = 0; i < theBrep->m_E.Count(); ++i)
    {
        osg::ref_ptr<osg::Geometry> aGeometry = new osg::Geometry();
        osg::ref_ptr<osg::Vec3Array> aVertices = new osg::Vec3Array();

        ON_BrepEdge* anEdge = theBrep->Edge(i);

        double t0 = 0.0;
        double t1 = 0.0;
        double d = 0.0;

        anEdge->GetDomain(&t0, &t1);

        d = (t1 - t0) / 5.0;

        for (double t = t0; (t - t1) < TOLERANCE_EDGE; t += d)
        {
            ON_3dPoint aPoint = anEdge->PointAt(t);

            aVertices->push_back(osg::Vec3(aPoint.x, aPoint.y, aPoint.z));
        }

        aGeometry->setVertexArray(aVertices);
        aGeometry->addPrimitiveSet(new osg::DrawArrays(osg::PrimitiveSet::LINE_LOOP, 0, aVertices->size()));

        aGeode->addDrawable(aGeometry);
    }

    return aGeode.release();
}

osg::Node* RhinoPlugin::BuildWireFrameFace(const ON_BrepFace* theFace)
{
    osg::ref_ptr<osg::Geode> aGeode = new osg::Geode();

    ON_NurbsSurface aSurface;

    if (theFace->GetNurbForm(aSurface) == 0)
    {
        return NULL;
    }

    double u0 = aSurface.Domain(0).Min();
    double u1 = aSurface.Domain(0).Max();
    double v0 = aSurface.Domain(1).Min();
    double v1 = aSurface.Domain(1).Max();

    double d0 = 0.0;
    double d1 = 0.0;

    d0 = (u1 - u0) / 10.0;
    d1 = (v1 - v0) / 10.0;

    for (double u = u0; (u - u1) < TOLERANCE_FACE; u += d0)
    {
        osg::ref_ptr<osg::Geometry> aGeometry = new osg::Geometry();
        osg::ref_ptr<osg::Vec3Array> aVertices = new osg::Vec3Array();

        for (double v = v0; (v - v1) < TOLERANCE_FACE; v += d1)
        {
            ON_3dPoint aPoint = aSurface.PointAt(u, v);

            aVertices->push_back(osg::Vec3(aPoint.x, aPoint.y, aPoint.z));
        }

        aGeometry->setVertexArray(aVertices);
        aGeometry->addPrimitiveSet(new osg::DrawArrays(osg::PrimitiveSet::LINE_STRIP, 0, aVertices->size()));

        aGeode->addDrawable(aGeometry);
    }

    for (double v = v0; (v - v1) < TOLERANCE_FACE; v += d1)
    {
        osg::ref_ptr<osg::Geometry> aGeometry = new osg::Geometry();
        osg::ref_ptr<osg::Vec3Array> aVertices = new osg::Vec3Array();

        for (double u = u0; (u - u1) < TOLERANCE_FACE; u += d0)
        {
            ON_3dPoint aPoint = aSurface.PointAt(u, v);

            aVertices->push_back(osg::Vec3(aPoint.x, aPoint.y, aPoint.z));
        }

        aGeometry->setVertexArray(aVertices);
        aGeometry->addPrimitiveSet(new osg::DrawArrays(osg::PrimitiveSet::LINE_STRIP, 0, aVertices->size()));

        aGeode->addDrawable(aGeometry);
    }

    return aGeode.release();
}

osg::Node* RhinoPlugin::BuildShadedFace(const ON_BrepFace* theFace)
{
    osg::ref_ptr<osg::Geode> aGeode = new osg::Geode();

    ON_NurbsSurface aSurface;

    if (theFace->GetNurbForm(aSurface) == 0)
    {
        return NULL;
    }

    osg::ref_ptr<osg::Geometry> aGeometry = new osg::Geometry();

    osg::ref_ptr<osg::Vec3Array> aUVPoints = new osg::Vec3Array();
    osg::ref_ptr<osg::Vec3Array> aPoints = new osg::Vec3Array();
    osg::ref_ptr<osg::Vec3Array> aBounds = new osg::Vec3Array();

    osg::ref_ptr<osgUtil::DelaunayTriangulator> dt = new osgUtil::DelaunayTriangulator();
    osg::ref_ptr<osgUtil::DelaunayConstraint> dc = new osgUtil::DelaunayConstraint();

    // add loop for the face.
    for (int i = 0; i < theFace->LoopCount(); ++i)
    {
        ON_BrepLoop* aLoop = theFace->Loop(i);

        if (aLoop->m_type == ON_BrepLoop::outer)
        {
            for (int j = 0; j < aLoop->TrimCount(); ++j)
            {
                ON_BrepTrim* aTrim = aLoop->Trim(j);

                const ON_Curve* aPCurve = aTrim->TrimCurveOf();
                if (aPCurve)
                {
                    ON_3dPoint aStartPoint = aPCurve->PointAtStart();
                    ON_3dPoint aEndPoint = aPCurve->PointAtEnd();

                    aUVPoints->push_back(osg::Vec3(aStartPoint.x, aStartPoint.y, 0.0));
                    aUVPoints->push_back(osg::Vec3(aEndPoint.x, aEndPoint.y, 0.0));
                }
            }
        }
        else if (aLoop->m_type == ON_BrepLoop::inner)
        {
            for (int j = 0; j < aLoop->TrimCount(); ++j)
            {
            }
        }
    }
    if (aUVPoints->getNumElements() < 3)
    {
        return nullptr; // not enough points to create a triangle
    }
    dc->setVertexArray(aBounds);
    dc->addPrimitiveSet(new osg::DrawArrays(osg::PrimitiveSet::LINE_LOOP, 0, aBounds->size()));

    // triangulate the parametric space.
    //dt->addInputConstraint(dc);
    dt->setInputPointArray(aUVPoints);
    dt->triangulate();
    //dt->removeInternalTriangles(dc);

    for (osg::Vec3Array::const_iterator j = aUVPoints->begin(); j != aUVPoints->end(); ++j)
    {
        // evaluate the point on the surface
        ON_3dPoint aPoint = aSurface.PointAt((*j).x(), (*j).y());

        aPoints->push_back(osg::Vec3(aPoint.x, aPoint.y, aPoint.z));
    }

    //aGeometry->setVertexArray(aUVPoints);
    aGeometry->setVertexArray(aPoints);
    aGeometry->addPrimitiveSet(dt->getTriangles());

    aGeode->addDrawable(aGeometry);

    // use smoothing visitor to set the average normals
    //osgUtil::SmoothingVisitor sv;
    //sv.apply(*aGeode);

    return aGeode.release();
}

COVERPLUGIN(RhinoPlugin)
