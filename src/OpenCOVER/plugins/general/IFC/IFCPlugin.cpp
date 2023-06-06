/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/****************************************************************************\
**                                                            (C)2021 HLRS  **
**                                                                          **
** Description: IFC Plugin (read IFC files)                                 **
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

#include "IFCPlugin.h"
#include <ifcpp/model/BasicTypes.h>
#include <ifcpp/model/BuildingModel.h>
#include <ifcpp/model/BuildingException.h>
#include <ifcpp/model/BuildingGuid.h>
#include <ifcpp/model/StatusCallback.h>
#include <ifcpp/reader/ReaderSTEP.h>
#include <ifcpp/writer/WriterSTEP.h>
#include <ifcpp/writer/WriterUtil.h>

#ifdef GEOMETRY_DEBUG_CHECK
#include <ifcpp/geometry/Carve/GeomDebugDump.h>
#endif
#include <ifcpp/geometry/Carve/GeometryConverter.h>
#include <ifcpp/geometry/Carve/ConverterOSG.h>
#include <ifcpp/geometry/Carve/GeomUtils.h>



#include <ifcpp/geometry/SceneGraphUtils.h>
#include <ifcpp/geometry/GeometrySettings.h>
#include "IfcPlusPlusSystem.h"

#ifdef HAVE_OSGNV
#include <osgNVExt/RegisterCombiners>
#include <osgNVExt/CombinerInput>
#include <osgNVExt/CombinerOutput>
#include <osgNVExt/FinalCombinerInput>
#endif

#include <osgText/Font>
#include <osgText/Text>


#include <sys/types.h>
#include <string.h>

#include <boost/locale.hpp>


IFCPlugin *IFCPlugin::plugin = NULL;

static FileHandler handlers[] = {
    {NULL,
      IFCPlugin::loadIFC,
      IFCPlugin::unloadIFC,
      "ifc"},
      {NULL,
      IFCPlugin::loadIFC,
      IFCPlugin::unloadIFC,
      "stp"},
      {NULL,
      IFCPlugin::loadIFC,
      IFCPlugin::unloadIFC,
      "step"}
};



int IFCPlugin::loadIFC(const char* filename, osg::Group* loadParent, const char*)
{
#ifdef WIN32
	std::string utf8_filename = boost::locale::conv::to_utf<char>(filename, "ISO-8859-1"); // we hope  the system locale is Latin1
#else
	std::string utf8_filename(filename); // we hope it is utf8 already
#endif
	return plugin->loadFile(utf8_filename, loadParent);
}

int IFCPlugin::unloadFile(const std::string& fileName)
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

int IFCPlugin::loadFile(const std::string&fileName, osg::Group *loadParent)
{
	if (loadParent == nullptr)
	{
		loadParent = IFCRoot;
	}
	IFCSwitch = new osg::Switch();
	loadParent->addChild(IFCSwitch);
	IFCSwitch->setName(fileName);
	switches.push_back(IFCSwitch);

	std::wstring_convert<std::codecvt_utf8_utf16<wchar_t>> converter;

	m_system->clearSelection();

	// reset the IFC model
	shared_ptr<GeometryConverter> geometry_converter = m_system->getGeometryConverter();
	geometry_converter->clearMessagesCallback();
	geometry_converter->resetModel();
	geometry_converter->getGeomSettings()->setNumVerticesPerCircle(16);
	geometry_converter->getGeomSettings()->setMinNumVerticesPerArc(4);
	std::stringstream err;

	// load file to IFC model
	shared_ptr<ReaderSTEP> step_reader(new ReaderSTEP());
	//step_reader->setMessageCallBack(this, &TabReadWrite::messageTarget);
	step_reader->loadModelFromFile(converter.from_bytes(fileName), geometry_converter->getBuildingModel());

	// convert IFC geometric representations into Carve geometry
	const double length_in_meter = geometry_converter->getBuildingModel()->getUnitConverter()->getLengthInMeterFactor();
	geometry_converter->setCsgEps(1.5e-08 * length_in_meter);
	geometry_converter->convertGeometry();

	// convert Carve geometry to OSG
	shared_ptr<ConverterOSG> converter_osg(new ConverterOSG(geometry_converter->getGeomSettings()));
	converter_osg->setMessageTarget(geometry_converter.get());
	converter_osg->convertToOSG(geometry_converter->getShapeInputData(), IFCSwitch);

	// in case there are IFC entities that are not in the spatial structure
	const std::map<std::string, shared_ptr<BuildingObject> >& objects_outside_spatial_structure = geometry_converter->getObjectsOutsideSpatialStructure();
	if (objects_outside_spatial_structure.size() > 0 && false)
	{
		osg::ref_ptr<osg::Switch> sw_objects_outside_spatial_structure = new osg::Switch();
		sw_objects_outside_spatial_structure->setName("IfcProduct objects outside spatial structure");

		converter_osg->addNodes(objects_outside_spatial_structure, sw_objects_outside_spatial_structure);
		if (sw_objects_outside_spatial_structure->getNumChildren() > 0)
		{
			IFCSwitch->addChild(sw_objects_outside_spatial_structure);
		}
	}

	/*if (IFCSwitch)
	{
		bool optimize = true;
		if (optimize)
		{
			osgUtil::Optimizer opt;
			opt.optimize(IFCSwitch);
		}

		// if model bounding sphere is far from origin, move to origin
		const osg::BoundingSphere& bsphere = IFCSwitch->getBound();
		if (bsphere.center().length() > 10000)
		{
			if (bsphere.center().length() / bsphere.radius() > 100)
			{
				//std::unordered_set<osg::Node*> set_applied;
				//SceneGraphUtils::translateGroup(loadParent, , set_applied, length_in_meter*0.001);

				osg::MatrixTransform* mt = new osg::MatrixTransform();
				mt->setMatrix(osg::Matrix::translate(-bsphere.center() * 0.98));

				int num_children = IFCSwitch->getNumChildren();
				for (int i = 0; i < num_children; ++i)
				{
					osg::Node* node = IFCSwitch->getChild(i);
					if (!node)
					{
						continue;
					}
					mt->addChild(node);
				}
				SceneGraphUtils::removeChildren(IFCSwitch);
				IFCSwitch->addChild(mt);
			}
		}
	}*/
    return 0;
}

int IFCPlugin::unloadIFC(const char *filename, const char *)
{
	return plugin->unloadFile(filename);
}


IFCPlugin::IFCPlugin()
: coVRPlugin(COVER_PLUGIN_NAME)
{
   
    //scaleGeometry = coCoviseConfig::isOn("COVER.Plugin.IFC.ScaleGeometry", true);
    //lodScale = coCoviseConfig::getFloat("COVER.Plugin.IFC.LodScale", 2000.0f);
	IFCRoot = new osg::MatrixTransform();
	IFCRoot->setName("IFCRoot");
	m_system = new IfcPlusPlusSystem();
	cover->getObjectsRoot()->addChild(IFCRoot);
    plugin = this;
}

bool IFCPlugin::init()
{
    for(int i=0;i<3;i++)
    coVRFileManager::instance()->registerFileHandler(&handlers[i]);
    return true;
}

// this is called if the plugin is removed at runtime
// which currently never happens
IFCPlugin::~IFCPlugin()
{
    for (int i = 0; i < 3; i++)
    coVRFileManager::instance()->unregisterFileHandler(&handlers[i]);
	cover->getObjectsRoot()->removeChild(IFCRoot);
	delete m_system;
}

void
IFCPlugin::preFrame()
{
}

COVERPLUGIN(IFCPlugin)
