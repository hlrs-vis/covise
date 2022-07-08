/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */


#include "GeneralGeometryPlugin.h"

#include "GeneralGeometryInteraction.h"

#include <cover/coVRPluginSupport.h>
#include <cover/coVRPluginList.h>
#include <cover/coVRMSController.h>
#include <config/CoviseConfig.h>
#include <cover/OpenCOVER.h>
#include <cover/RenderObject.h>
#include <grmsg/coGRObjVisMsg.h>
#include <grmsg/coGRObjSetCaseMsg.h>
#include <grmsg/coGRObjSetNameMsg.h>
#include <grmsg/coGRObjColorObjMsg.h>
#include <grmsg/coGRObjSetTransparencyMsg.h>
//#include <grmsg/coGRObjShaderObjMsg.h>
#include <grmsg/coGRObjMaterialObjMsg.h>
#include <grmsg/coGRKeyWordMsg.h>
#include <grmsg/coGRObjTransformMsg.h>
#include <grmsg/coGRAnimationOnMsg.h>
#include <grmsg/coGRSetAnimationSpeedMsg.h>
#include <grmsg/coGRSetTimestepMsg.h>

#include <map>

using namespace covise;
using namespace grmsg;
using namespace opencover;

// void
// coVRAddObject(DO_Geometry *container,
// DistributedObject* geomobj,
// DistributedObject* /*normObj*/,
// DistributedObject* /*colorObj*/,
// DistributedObject* /*texObj*/,
// const char* /*root*/,
// int /*numCol*/,
// int /*colorBinding*/,
// int /*colorPacking*/,
// float* /*r*/,
// float* /*g*/,
// float* /*b*/,
// int* /*packedCol*/,
// int /*numNormals*/,
// int /*normalBinding*/,
// float* /*xn*/,
// float* /*yn*/,
// float* /*zn*/,
// float /*transparency*/)
// {
//    if (cover->debugLevel(3))
//    {
//       if (container)
//          fprintf(stderr,"\n--- Plugin GeneralGeometry coVRAddObject container=%s geomobject=%s\n", container->getName(), geomobj->getName());
//       else
//          fprintf(stderr,"\n--- Plugin GeneralGeometry coVRAddObject geomobject=%s\n", geomobj->getName());
//    }
//    const char *moduleName;
//    if (container)
//       moduleName = container->getName();
//    else
//       moduleName = geomobj->getName();
//    if (msgForGeneralGeometry(moduleName))
//    {
//       add(container, geomobj);
//    }
// }

//-----------------------------------------------------------------------------

GeneralGeometryPlugin::GeneralGeometryPlugin()
    : ModuleFeedbackPlugin()
{
    if (cover->debugLevel(3))
        fprintf(stderr, "\nGeneralGeometryPlugin::GeneralGeometryPlugin\n");
}

GeneralGeometryPlugin::~GeneralGeometryPlugin()
{
    if (cover->debugLevel(3))
        fprintf(stderr, "\nGeneralGeometryPlugin::~GeneralGeometryPlugin\n");
}

bool GeneralGeometryPlugin::init()
{
    return true;
}

void GeneralGeometryPlugin::addNode(osg::Node *node, const RenderObject *obj)
{
    if (obj)
    {
        addNodeToCase(obj->getName(), node);
    }
}

void GeneralGeometryPlugin::removeObject(const char *objName, bool replaceFlag)
{
    if (!replaceFlag && objName != NULL)
    {
        if (msgForGeneralGeometry(objName))
        {
            remove(objName);
        }
    }
}

void GeneralGeometryPlugin::addObject(const RenderObject *baseObj, osg::Group *, const RenderObject *geometry, const RenderObject *, const RenderObject *, const RenderObject *)
{
    ModuleFeedbackPlugin::add(baseObj, geometry);
}

void GeneralGeometryPlugin::guiToRenderMsg(const grmsg::coGRMsg &msg) 
{
    if (cover->debugLevel(3))
        fprintf(stderr, "\n--- Plugin GeneralGeometry coVRGuiToRenderMsg [%s]\n", msg.getString().c_str());

    if (msg.isValid())
    {
        switch (msg.getType())
        {
        case coGRMsg::GEO_VISIBLE:
        {
            auto &geometryVisibleMsg = msg.as<coGRObjVisMsg>();
            const char *objectName = geometryVisibleMsg.getObjName();
            if (msgForGeneralGeometry(objectName))
            {
                if (cover->debugLevel(3))
                    fprintf(stderr, "in GeneralGeometry coGRMsg::GEO_VISIBLE object=%s visible=%d\n", objectName, geometryVisibleMsg.isVisible());

                handleGeoVisibleMsg(objectName, geometryVisibleMsg.isVisible());
            }
        }
        break;
        case coGRMsg::SET_CASE:
        {
            auto &setCaseMsg = msg.as<coGRObjSetCaseMsg>();
            const char *objectName = setCaseMsg.getObjName();
            if (msgForGeneralGeometry(objectName))
            {
                const char *caseName = setCaseMsg.getCaseName();
                if (cover->debugLevel(3))
                    fprintf(stderr, "in GeneralGeometry coGRMsg::SET_CASE object=%s case=%s\n", objectName, caseName);
                handleSetCaseMsg(objectName, caseName);
            }
        }
        break;
        case coGRMsg::SET_NAME:
        {
            auto &setNameMsg = msg.as<coGRObjSetNameMsg>();
            const char *coviseObjectName = setNameMsg.getObjName();
            const char *newName = setNameMsg.getNewName();
            if (cover->debugLevel(3))
                fprintf(stderr, "in GeneralGeometry coGRMsg::SET_NAME object=%s name=%s\n", coviseObjectName, newName);
            handleSetNameMsg(coviseObjectName, newName);
        }
        break;
        case coGRMsg::COLOR_OBJECT:
        {
            auto &colorObjMsg = msg.as<coGRObjColorObjMsg>();
            const char *objectName = colorObjMsg.getObjName();
            if (msgForGeneralGeometry(objectName))
            {
                if (cover->debugLevel(3))
                    fprintf(stderr, "in GeneralGeometry  coGRMsg::COLOR_OBJECT object=%s\n", objectName);
                int *color = new int[3];
                color[0] = colorObjMsg.getR();
                color[1] = colorObjMsg.getG();
                color[2] = colorObjMsg.getB();
                setColor(objectName, color);

                delete[] color;
            }
        }
        break;
        case coGRMsg::SET_TRANSPARENCY:
        {
            auto &setTransparencyMsg = msg.as<coGRObjSetTransparencyMsg>();
            const char *objectName = setTransparencyMsg.getObjName();

            if (msgForGeneralGeometry(objectName))
            {
                if (cover->debugLevel(3))
                    fprintf(stderr, "in GeneralGeometry  coGRMsg::SET_TRANSPARENCY object=%s\n", objectName);

                setTransparency(objectName, setTransparencyMsg.getTransparency());
            }
        }
        break;
        case coGRMsg::MATERIAL_OBJECT:
        {
            auto &materialObjMsg = msg.as<coGRObjMaterialObjMsg>();
            const char *objectName = materialObjMsg.getObjName();

            if (msgForGeneralGeometry(objectName))
            {
                if (cover->debugLevel(3))
                    fprintf(stderr, "in GeneralGeometry coGRMsg::MATERIAL_OBJECT object=%s\n", objectName);

                const int *ambient = materialObjMsg.getAmbient();
                const int *diffuse = materialObjMsg.getDiffuse();
                const int *specular = materialObjMsg.getSpecular();
                float shininess = materialObjMsg.getShininess();
                float transparency = materialObjMsg.getTransparency();
                if (cover->debugLevel(3))
                    fprintf(stderr, "coGRMsg::MATERIAL_OBJECT object=%s\n", objectName);
                setMaterial(objectName, ambient, diffuse, specular, shininess, transparency);
            }
        }
        break;
        case coGRMsg::TRANSFORM_OBJECT:
        {
            auto &transformMsg = msg.as<coGRObjTransformMsg>();
            const char *objectName = transformMsg.getObjName();

            if (msgForGeneralGeometry(objectName))
            {
                if (cover->debugLevel(3))
                    fprintf(stderr, "in GeneralGeometry coGRMsg::TRANSFORM_OBJECT object=%s\n", objectName);

                float row0[4];
                float row1[4];
                float row2[4];
                float row3[4];
                if (cover->debugLevel(3))
                    fprintf(stderr, "coGRMsg::TRANSFORM_OBJECT object=%s\n", objectName);
                for (int i = 0; i < 4; i++)
                {
                    row0[i] = transformMsg.getMatrix(0, i);
                    row1[i] = transformMsg.getMatrix(1, i);
                    row2[i] = transformMsg.getMatrix(2, i);
                    row3[i] = transformMsg.getMatrix(3, i);
                }

                setMatrix(objectName, row0, row1, row2, row3);
            }
        }
        break;
        default:
        {
            if (cover->debugLevel(3))
                fprintf(stderr, "NOT-USED\n");
        }
        break;
        }
    }
}

opencover::ModuleFeedbackManager *
GeneralGeometryPlugin::NewModuleFeedbackManager(const RenderObject *container, coInteractor *, const RenderObject *geomObject, const char *pluginName)
{
    return new GeneralGeometryInteraction(container, geomObject, pluginName);
}

void
GeneralGeometryPlugin::setColor(const char *objectName, int *color)
{
    //fprintf(stderr,"*****GeneralGeometryPlugin::setColor(%s, %d %d %d)\n", objectName, color[0], color[1], color[2]);

    myInteractions_.reset();
    while (myInteractions_.current())
    {
        if (myInteractions_.current()->compare(objectName))
        {
            ((GeneralGeometryInteraction *)myInteractions_.current())->setColor(color);
            break;
        }
        myInteractions_.next();
    }
}

//void
//GeneralGeometryPlugin::setShader(const char *objectName, const char* shaderName, const char* paraFloat, const char* paraVec2, const char* paraVec3, const char* paraVec4, const char* paraInt, const char* paraBool, const char* paraMat2, const char* paraMat3, const char* paraMat4)
//{
//   //fprintf(stderr,"******GeneralGeometryPlugin::setShader(%s, %s)\n", objectName, shaderName);
//
//   myInteractions_.reset();
//   while (myInteractions_.current())
//   {
//      if (myInteractions_.current()->compare(objectName))
//      {
//         ((GeneralGeometryInteraction*)myInteractions_.current())->setShader(shaderName,paraFloat,paraVec2,paraVec3,paraVec4,paraInt,paraBool, paraMat2, paraMat3, paraMat4);
//         break;
//      }
//      myInteractions_.next();
//   }
//}

void
GeneralGeometryPlugin::setTransparency(const char *objectName, float transparency)
{
    myInteractions_.reset();
    while (myInteractions_.current())
    {
        if (myInteractions_.current()->compare(objectName))
        {
            ((GeneralGeometryInteraction *)myInteractions_.current())->setTransparency(transparency);
            break;
        }
        myInteractions_.next();
    }
}

void
GeneralGeometryPlugin::setMaterial(const char *objectName, const int *ambient, const int *diffuse, const int *specular, float shininess, float transparency)
{
    //fprintf(stderr,"******GeneralGeometryPlugin::setMaterial to %s\n", objectName);

    myInteractions_.reset();
    while (myInteractions_.current())
    {
        if (myInteractions_.current()->compare(objectName))
        {
            ((GeneralGeometryInteraction *)myInteractions_.current())->setMaterial(ambient, diffuse, specular, shininess, transparency);
            break;
        }
        myInteractions_.next();
    }
}

bool GeneralGeometryPlugin::msgForGeneralGeometry(const char *moduleName)
{
    if (strncmp(moduleName, "CuttingSurfaceComp", 18) != 0 && strncmp(moduleName, "CuttingSurface", 14) != 0 && strncmp(moduleName, "IsoSurfaceComp", 14) != 0 && strncmp(moduleName, "IsoSurface", 10) != 0 && strncmp(moduleName, "TracerComp", 10) != 0)
        return true;
    return false;
}

COVERPLUGIN(GeneralGeometryPlugin)
