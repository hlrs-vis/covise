/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "GeneralGeometryInteraction.h"

#include <cover/coVRPluginSupport.h>
#include <cover/OpenCOVER.h>
#include <net/message_types.h>
#include <net/message.h>

#include <PluginUtil/colors/ColorBar.h>

#include <OpenVRUI/coCheckboxMenuItem.h>
#include <OpenVRUI/coPotiMenuItem.h>

#include <osg/Geometry>
#include <osg/Material>
#include <osg/Depth>

#include <cover/VRSceneGraph.h>
#include <grmsg/coGRObjSetTransparencyMsg.h>
#include <grmsg/coGRMsg.h>

using namespace osg;
using namespace vrui;
using namespace grmsg;
using namespace covise;
using namespace opencover;

GeneralGeometryInteraction::GeneralGeometryInteraction(const RenderObject *container, const RenderObject *geomObject, const char *pluginName)
    : ModuleFeedbackManager(container, geomObject, pluginName)
{
    if (cover->debugLevel(2))
        fprintf(stderr, "GeneralGeometryInteraction::GeneralGeometryInteraction\n");

    // check if the parameters are ok
    if (!paramsOk())
    {
        return;
    }
    newObject_ = false;
    firsttime_ = true;
    //    fluxedColors_=NULL;
    transparency_ = 1.0;

    transparencyPoti_ = new coPotiMenuItem("Transparency", 0.0, 1.0, 1.0);
    transparencyPoti_->setMenuListener(this);
    menu_->add(transparencyPoti_);
}

GeneralGeometryInteraction::~GeneralGeometryInteraction()
{
    if (cover->debugLevel(2))
        fprintf(stderr, "\nGeneralGeometryInteraction::~GeneralGeometryInteraction\n");
    delete transparencyPoti_;
}

void
GeneralGeometryInteraction::update(const RenderObject *container, const RenderObject *obj)
{
    (void)container;
    (void)obj;

    if (cover->debugLevel(3))
        fprintf(stderr, "\nGeneralGeometryInteraction::update\n");
    // base class updates the item in the COVISE menu
    // and the title of the Tracer menu
    ModuleFeedbackManager::update(container, obj);
    newObject_ = true;
    firsttime_ = true;

    if (cover->debugLevel(3))
        fprintf(stderr, "\nGeneralGeometryInteraction::update done\n");
}

void
GeneralGeometryInteraction::preFrame()
{
    if (cover->debugLevel(5))
        fprintf(stderr, "\nGeneralGeometryInteraction::preFrame newObject=%d\n", newObject_);

    // update visibility of new traces
    // in update the new geometry is not in the sg, either use addNode or delay it to preFrame
    if (newObject_)
    {
        ModuleFeedbackManager::hideGeometry(hideCheckbox_->getState());
        newObject_ = false;
    }
}

bool
GeneralGeometryInteraction::paramsOk()
{
    if (cover->debugLevel(2))
        fprintf(stderr, "\nGeneralGeometryInteraction::paramsOk\n");

    return (true);
}

void
GeneralGeometryInteraction::menuEvent(coMenuItem *item)
{
    //    fprintf(stderr,"GeneralGeometryInteraction::menuEvent(%s)\n", item->getName());
    ModuleFeedbackManager::menuEvent(item);

    if (item == transparencyPoti_)
    {
        float transparency = transparencyPoti_->getValue();
        //check if we can use shaders

        setTransparencyWithoutShader(transparency);

        //send to GUI
        //       std::vector<osg::Geode*> geodes = findMyGeode();
        //       for( std::vector<osg::Geode*>::iterator it= geodes.begin(); it != geodes.end(); it++)
        //       {
        coGRObjSetTransparencyMsg vMsg(coGRMsg::SET_TRANSPARENCY, initialObjectName_.c_str(), transparency);
        cover->sendGrMessage(vMsg);
        //       }
    }
}

void GeneralGeometryInteraction::setTransparencyWithoutShader(float transparency)
{
    //fprintf(stderr, "GeneralGeometryInteraction::setTransparencyWithoutShader(%f)\n",  transparency);
    osg::ref_ptr<osg::Vec4Array> oldColors;

    std::vector<osg::Geode *> geodes = findMyGeode();
    if (geodes.size() > 0)
    {
        for (size_t j = 0; j < geodes.size(); j++)
        {
            osg::ref_ptr<osg::Geode> geode = geodes[j];
            geode->ref();
            osg::StateAttribute::GLModeValue attr = geode->getStateSet()->getMode(osg::StateAttribute::MATERIAL);

            // set transparency for all drawables
            for (size_t i = 0; i < geode->getNumDrawables(); i++)
            {
                geoset_ = geode->getDrawable(i);
                geoset_->ref();
                osg::Geometry::AttributeBinding bind = geoset_->asGeometry()->getColorBinding();
                oldColors = (osg::Vec4Array *)(geoset_->asGeometry()->getColorArray());
                if (!oldColors.get())
                {
                    if (geoset_->getStateSet())
                        attr = geoset_->asGeometry()->getStateSet()->getMode(osg::StateAttribute::MATERIAL);
                    if (geoset_->getStateSet() && ((attr == osg::StateAttribute::ON) || (attr == osg::StateAttribute::INHERIT)))
                    {
                        osg::ref_ptr<osg::Material> mtl = (osg::Material *)geoset_->asGeometry()->getStateSet()->getAttribute(osg::StateAttribute::MATERIAL);
                        if (mtl.get())
                        {
                            mtl->ref();
                            osg::Vec4 color;
                            color = mtl->getAmbient(osg::Material::FRONT_AND_BACK);
                            mtl->setAmbient(osg::Material::FRONT_AND_BACK, osg::Vec4(color[0], color[1], color[2], transparency));
                            color = mtl->getDiffuse(osg::Material::FRONT_AND_BACK);
                            mtl->setDiffuse(osg::Material::FRONT_AND_BACK, osg::Vec4(color[0], color[1], color[2], transparency));
                            color = mtl->getSpecular(osg::Material::FRONT_AND_BACK);
                            mtl->setSpecular(osg::Material::FRONT_AND_BACK, osg::Vec4(color[0], color[1], color[2], transparency));
                            color = mtl->getEmission(osg::Material::FRONT_AND_BACK);
                            mtl->setEmission(osg::Material::FRONT_AND_BACK, osg::Vec4(color[0], color[1], color[2], transparency));
                            geoset_->getStateSet()->setAttributeAndModes(mtl.get(), osg::StateAttribute::ON);
                            mtl->unref();
                        }
                    }
                    else
                    {
                        oldColors = new osg::Vec4Array();
                        oldColors->push_back(osg::Vec4(1.0, 0.2, 0.2, transparency));
                        oldColors->ref();
                        bind = osg::Geometry::BIND_OVERALL;
                        geoset_->asGeometry()->setColorArray(oldColors.get());
                        geoset_->asGeometry()->setColorBinding(bind);
                        oldColors->unref();
                    }
                }
                else
                {
                    oldColors->ref();
                    for (osg::Vec4Array::iterator it = oldColors->begin(); it != oldColors->end(); it++)
                        (*it)[3] = transparency;
                    geoset_->asGeometry()->setColorArray(oldColors.get());
                    geoset_->asGeometry()->setColorBinding(bind);
                    oldColors->unref();
                }

                if (transparency < 1.0)
                {
                    geoset_->getOrCreateStateSet()->setRenderingHint(StateSet::TRANSPARENT_BIN);

                    geoset_->getOrCreateStateSet()->setMode(GL_BLEND, osg::StateAttribute::ON);

                    // Disable writing to depth buffer.
                    osg::Depth *depth = new osg::Depth;
                    depth->setWriteMask(false);
                    geoset_->getOrCreateStateSet()->setAttributeAndModes(depth, osg::StateAttribute::ON);
                }
                else
                {
                    geoset_->getOrCreateStateSet()->setRenderingHint(StateSet::OPAQUE_BIN);

                    geoset_->getOrCreateStateSet()->setMode(GL_BLEND, osg::StateAttribute::OFF);

                    // Enable writing to depth buffer.
                    osg::Depth *depth = new osg::Depth;
                    depth->setWriteMask(true);
                    geoset_->getOrCreateStateSet()->setAttributeAndModes(depth, osg::StateAttribute::ON);
                }

                geoset_->unref();
            }
            geode->dirtyBound();
            geode->unref();
        }
    }
    transparency_ = transparency;
}

void
GeneralGeometryInteraction::setColor(int *color)
{
    //fprintf(stderr,"GeneralGeometryInteraction::setColor of object \n");
    std::vector<osg::Geode *> geodes = findMyGeode();

    if (geodes.size() == 0)
    {
        if (cover->debugLevel(2))
            fprintf(stderr, "GeneralGeometryInteraction::setColor ERROR didn't find my geode \n");
        return;
    }
    firsttime_ = true;
    for (size_t i = 0; i < geodes.size(); i++)
    {
        osg::Geode *geode = geodes[i];
        VRSceneGraph::instance()->setColor(geode, color, transparency_);
    }
}

void
GeneralGeometryInteraction::setTransparency(float transparency)
{
    std::vector<osg::Geode *> geodes = findMyGeode();

    if (geodes.size() == 0)
    {
        if (cover->debugLevel(2))
            fprintf(stderr, "GeneralGeometryInteraction::setTransparency ERROR didn't find my geode \n");
        return;
    }

    for (size_t i = 0; i < geodes.size(); i++)
    {
        osg::Geode *geode = geodes[i];
        VRSceneGraph::instance()->setTransparency(geode, transparency);
    }
}

// not implemented
//void
//GeneralGeometryInteraction::setShader(const char* shaderName, const char* paraFloat, const char* paraVec2, const char* paraVec3, const char* paraVec4, const char* paraInt, const char* paraBool, const char* paraMat2, const char* paraMat3, const char* paraMat4)
//{
//   //fprintf(stderr, "GeneralGeometryInteraction::setShader()\n");
//   (void) shaderName;
//   (void)paraFloat ;
//   (void) paraVec2;
//   (void) paraVec3;
//   (void) paraVec4;
//   (void)paraInt ;
//   (void) paraBool;
//   (void) paraMat2;
//   (void) paraMat3;
//   (void) paraMat4;
//
//
//   std::vector<osg::Geode *>geodes = findMyGeode();
//
//   if (geodes.size()==0)
//   {
//      if (cover->debugLevel(2))
//         fprintf(stderr,"GeneralGeometryInteraction::setShader ERROR didn't find my geode \n");
//      return;
//   }
//   firsttime_=true;
//
//   // look for the transparency in the shader
//   std::string param;
//   std::string varName;
//   float fvalue;
//   //the float variable
//   param = paraFloat;
//   std::string::size_type erg = param.find(':') ;
//   std::string::size_type erg2;
//   //found one variable
//   while (erg != std::string::npos)
//   {
//      varName =param.substr(0, erg);
//      erg2 = param.find(' ', erg+1);
//      fvalue = atof( param.substr(erg+1, erg2).c_str() );
//      if (varName == "transparency")
//      {
//         for (int i=0;i<geodes.size(); i++)
//         {
//            osg::Geode* geode = geodes[i];
//            VRSceneGraph::instance()->setTransparency(geode, fvalue);
//         }
//         break;
//      }
//      //there are more variables
//      if (erg2!=std::string::npos)
//      {
//         param = param.substr(erg2+1);
//         erg = param.find(':',0) ;
//      } else
//         erg = erg2;
//   }
//
//
////    VRSceneGraph::instance()->setShader(geode->getName(), shaderName, paraFloat, paraVec2, paraVec3, paraVec4, paraInt, paraBool, paraMat2, paraMat3, paraMat4);
//}

void
GeneralGeometryInteraction::setMaterial(const int *ambient, const int *diffuse, const int *specular, float shininess, float transparency)
{
    //fprintf(stderr,"GeneralGeometryInteraction::setMaterial of object \n");

    std::vector<osg::Geode *> geodes = findMyGeode();

    if (geodes.size() == 0)
    {
        if (cover->debugLevel(2))
            fprintf(stderr, "GeneralGeometryInteraction::setMaterial ERROR didn't find my geode\n");
        return;
    }
    firsttime_ = true;
    for (size_t i = 0; i < geodes.size(); i++)
    {
        osg::Geode *geode = geodes[i];
        VRSceneGraph::instance()->setMaterial(geode, ambient, diffuse, specular, shininess, transparency);
        transparencyPoti_->setValue(transparency);
    }
}
