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

void GeneralGeometryPlugin::addNode(osg::Node *node, RenderObject *obj)
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

void GeneralGeometryPlugin::addObject(RenderObject *baseObj,
                                      RenderObject *geomObj, RenderObject *,
                                      RenderObject *, RenderObject *,
                                      osg::Group *,
                                      int, int, int,
                                      float *, float *, float *, int *,
                                      int, int,
                                      float *, float *, float *,
                                      float)
{
    ModuleFeedbackPlugin::add(baseObj, geomObj);
}

void GeneralGeometryPlugin::guiToRenderMsg(const char *msg)
{
    if (cover->debugLevel(3))
        fprintf(stderr, "\n--- Plugin GeneralGeometry coVRGuiToRenderMsg [%s]\n", msg);

    string fullMsg(string("GRMSG\n") + msg);
    coGRMsg grMsg(fullMsg.c_str());
    if (grMsg.isValid())
    {
        if (grMsg.getType() == coGRMsg::GEO_VISIBLE)
        {
            coGRObjVisMsg geometryVisibleMsg(fullMsg.c_str());
            const char *objectName = geometryVisibleMsg.getObjName();
            if (msgForGeneralGeometry(objectName))
            {
                if (cover->debugLevel(3))
                    fprintf(stderr, "in GeneralGeometry coGRMsg::GEO_VISIBLE object=%s visible=%d\n", objectName, geometryVisibleMsg.isVisible());

                handleGeoVisibleMsg(objectName, geometryVisibleMsg.isVisible());
            }
        }
        else if (grMsg.getType() == coGRMsg::SET_CASE) // die visMsg ist eigentlich eine boolean msg
        {

            coGRObjSetCaseMsg setCaseMsg(fullMsg.c_str());
            const char *objectName = setCaseMsg.getObjName();
            if (msgForGeneralGeometry(objectName))
            {
                const char *caseName = setCaseMsg.getCaseName();
                if (cover->debugLevel(3))
                    fprintf(stderr, "in GeneralGeometry coGRMsg::SET_CASE object=%s case=%s\n", objectName, caseName);
                handleSetCaseMsg(objectName, caseName);
            }
        }
        else if (grMsg.getType() == coGRMsg::SET_NAME)
        {
            coGRObjSetNameMsg setNameMsg(fullMsg.c_str());
            const char *coviseObjectName = setNameMsg.getObjName();
            const char *newName = setNameMsg.getNewName();
            if (cover->debugLevel(3))
                fprintf(stderr, "in GeneralGeometry coGRMsg::SET_NAME object=%s name=%s\n", coviseObjectName, newName);
            handleSetNameMsg(coviseObjectName, newName);
        }

        else if (grMsg.getType() == coGRMsg::COLOR_OBJECT)
        {
            coGRObjColorObjMsg colorObjMsg(fullMsg.c_str());
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

        //      else if (grMsg.getType()==coGRMsg::SHADER_OBJECT)
        //      {
        //         coGRObjShaderObjMsg shaderObjMsg(fullMsg.c_str());
        //         const char* objectName = shaderObjMsg.getObjName();
        //
        //         if (msgForGeneralGeometry(objectName))
        //         {
        //            if (cover->debugLevel(3))
        //               fprintf(stderr, "in GeneralGeometry  coGRMsg::SHADER_OBJECT object=%s\n",objectName);
        //
        //            const char* shaderName = shaderObjMsg.getShaderName();
        //            const char* mapFloat = shaderObjMsg.getParaFloatName();
        //            const char* mapVec2 = shaderObjMsg.getParaVec2Name();
        //            const char* mapVec3 = shaderObjMsg.getParaVec3Name();
        //            const char* mapVec4 = shaderObjMsg.getParaVec4Name();
        //            const char* mapBool = shaderObjMsg.getParaBoolName();
        //            const char* mapInt = shaderObjMsg.getParaIntName();
        //            const char* mapMat2 = shaderObjMsg.getParaMat2Name();
        //            const char* mapMat3 = shaderObjMsg.getParaMat3Name();
        //            const char* mapMat4 = shaderObjMsg.getParaMat4Name();
        //            if (cover->debugLevel(3))
        //               fprintf(stderr, "coGRMsg::SHADER_OBJECT object=%s\n",objectName);
        //            setShader(objectName, shaderName,mapFloat,mapVec2,mapVec3,mapVec4,mapInt,mapBool, mapMat2, mapMat3, mapMat4);
        //         }
        //      }

        else if (grMsg.getType() == coGRMsg::SET_TRANSPARENCY)
        {
            coGRObjSetTransparencyMsg setTransparencyMsg(fullMsg.c_str());
            const char *objectName = setTransparencyMsg.getObjName();

            if (msgForGeneralGeometry(objectName))
            {
                if (cover->debugLevel(3))
                    fprintf(stderr, "in GeneralGeometry  coGRMsg::SET_TRANSPARENCY object=%s\n", objectName);

                setTransparency(objectName, setTransparencyMsg.getTransparency());
            }
        }

        else if (grMsg.getType() == coGRMsg::MATERIAL_OBJECT)
        {
            coGRObjMaterialObjMsg materialObjMsg(fullMsg.c_str());
            const char *objectName = materialObjMsg.getObjName();

            if (msgForGeneralGeometry(objectName))
            {
                if (cover->debugLevel(3))
                    fprintf(stderr, "in GeneralGeometry coGRMsg::MATERIAL_OBJECT object=%s\n", objectName);

                int *ambient = materialObjMsg.getAmbient();
                int *diffuse = materialObjMsg.getDiffuse();
                int *specular = materialObjMsg.getSpecular();
                float shininess = materialObjMsg.getShininess();
                float transparency = materialObjMsg.getTransparency();
                if (cover->debugLevel(3))
                    fprintf(stderr, "coGRMsg::MATERIAL_OBJECT object=%s\n", objectName);
                setMaterial(objectName, ambient, diffuse, specular, shininess, transparency);
            }
        }

        else if (grMsg.getType() == coGRMsg::TRANSFORM_OBJECT)
        {
            coGRObjTransformMsg transformMsg(fullMsg.c_str());
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
        else
        {
            if (cover->debugLevel(3))
                fprintf(stderr, "NOT-USED\n");
        }
    }
}

opencover::ModuleFeedbackManager *
GeneralGeometryPlugin::NewModuleFeedbackManager(RenderObject *container, coInteractor *, RenderObject *geomObject, const char *pluginName)
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
GeneralGeometryPlugin::setMaterial(const char *objectName, int *ambient, int *diffuse, int *specular, float shininess, float transparency)
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
