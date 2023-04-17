/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

//
//  Vrml 97 library
//  Copyright (C) Uwe Woessner
//
//  %W% %G%
//  ViewerOsg.cpp
//  Display of VRML models using Performer/COVER.
//

#ifdef _WIN32
#if (_MSC_VER >= 1300) && !(defined(MIDL_PASS) || defined(RC_INVOKED))
#define POINTER_64 __ptr64
#else
#define POINTER_64
#endif
#include <winsock2.h>
#include <windows.h>
#endif
#include <util/unixcompat.h>
#include <vrml97/vrml/config.h>

#include <vrml97/vrml/MathUtils.h>
#include <vrml97/vrml/System.h>
#include <vrml97/vrml/VrmlScene.h>
#include <vrml97/vrml/VrmlNodeNavigationInfo.h>
#include <vrml97/vrml/VrmlNode.h>
#include <vrml97/vrml/Player.h>

#include <osg/Group>
#include <osg/MatrixTransform>
#include <osg/TexGen>
#include <osg/TexEnv>
#include <osg/TexMat>
#include <osg/TexEnvCombine>
#include <osg/Texture>
#include <osg/Texture2D>
#include <osg/TextureCubeMap>
#include <osg/Geode>
#include <osg/Switch>
#include <osg/Geometry>
#include <osg/PrimitiveSet>
#include <osg/Shape>
#include <osg/ShapeDrawable>
#include <osg/BlendFunc>
#include <osg/AlphaFunc>
#include <osg/TexEnvCombine>
#include <osg/GL>
#include <osg/Depth>
#include <osg/LightSource>

#include "ViewerObject.h"
#include "ViewerOsg.h"
#include <cover/coVRPluginSupport.h>
#include <cover/coVRLighting.h>

extern int Blended;
extern int AlphaTest;

#include <config/CoviseConfig.h>
/*
#include <cover/coVRPluginSupport.h>
#include <cover/coCubeMap.h>
#include <cover/coCgWave.h>
#include <cover/coEffectHandler.h>
#include <cover/coCgBumpMapping.h>
#include <cover/coEffectHandler.h>
#include <cover/coTrackerButtonInteraction.h>
#include <Performer/pr/pfCombiner.h>
#include "Vrml97Plugin.h"
*/

using namespace osg;

bool almostEqualMat(const Matrix &m1, const Matrix &m2, double tolerance)
{
    for (int i = 0; i < 16; i++)
    {
        if (fabs(m1.ptr()[i] - m2.ptr()[i]) > tolerance)
            return false;
    }

    return true;
}

TexEnvCombine *combiner = NULL;
TexEnvCombine *combinerEnv = NULL;

#if 0
int
CmbPostDraw( pfTraverser *, void * /*_userData*/ )
{
   pfDisable( PFEN_COMBINER );
   return PFTRAV_CONT;
}


int
CmbPreDraw( pfTraverser *, void * /*_userData*/ )
{
   pfEnable( PFEN_COMBINER );
   return PFTRAV_CONT;
}


int NoFramebufferPreDrawCallback(pfTraverser *, void *)
{
   glColorMask(false,false,false,false);
   return PFTRAV_CONT;
}


int NoFramebufferPostDrawCallback(pfTraverser *, void *)
{

   glColorMask(true,true,true,true);
   return PFTRAV_CONT;
}


int NoDepthbufferPreDrawCallback(pfTraverser *, void *)
{
   glDepthMask(false);
   return PFTRAV_CONT;
}


int NoDepthbufferPostDrawCallback(pfTraverser *, void *)
{
   glDepthMask(true);
   return PFTRAV_CONT;
}


//billboardList bblist;
//int noViewpoints=0;

class MoveInfo;
#endif

osgViewerObject *osgViewerObject::getChild(VrmlNode *n)
{
    auto iter = childIndex.find(n);
    if (iter != childIndex.end())
    {
        return iter->second;
    }

    return nullptr;
}

bool osgViewerObject::hasChild(osgViewerObject *o)
{
    auto it = std::find(children.begin(), children.end(), o);
    return it != children.end();
}

void osgViewerObject::setRootNode(Node *n)
{
    rootNode = n;

    for (auto c: children)
    {
        if (c->rootNode != n)
            c->setRootNode(n);
    }
}

void osgViewerObject::addChild(osgViewerObject *n)
{
    //cerr << "osgViewerObject::addChild, num: " << children.num() << ", node:" << n->node << endl;
    n->parent = this;
    children.push_back(n);
    if (n->node)
        childIndex[n->node] = n;
    n->ref();
    if (rootNode.get())
        n->setRootNode(rootNode.get());
}

void osgViewerObject::addChildrensNodes()
{
    for (auto c: children)
    {
        if (c->pNode.get() && c->haveToAdd > 0)
        {
            if (((Group *)pNode.get())->containsNode(c->getNode()))
            {
                if (whichChoice > -1)
                {
                    //cerr << "searchChild\n";
                    if (cover->debugLevel(1))
                        cerr << "_sch_";
                    choiceMap[whichChoice] = ((Group *)pNode.get())->getChildIndex(c->getNode());
                    //choiceMap[whichChoice]=((Group *)pNode.get())->searchChild(c->getNode());
                    //fprintf(stderr,"addChildrenNodes() 1: choiceMap: %d -> %d\n", whichChoice, choiceMap[whichChoice]);
                }
                c->haveToAdd--;
            }
            else
            {
                if ((c->node) && (strncmp(c->node->name(), "StaticCave", 10) == 0))
                {
                    //ViewerOsg::VRMLCaveRoot->addChild(c->getNode());
                    viewer->addObj(c, ViewerOsg::VRMLCaveRoot);
                    c->setRootNode(ViewerOsg::VRMLCaveRoot);
                    //cerr << "add node " << c->node->name() << " to Cave root!" << endl;
                    if (cover->debugLevel(1))
                        cerr << "_SC_";
                }
                else
                {
                    //((Group *)pNode.get())->addChild(c->getNode());
                    viewer->addObj(c, (Group *)pNode.get());
                }
                if (whichChoice > -1)
                {
                    if (cover->debugLevel(1))
                        cerr << "_sch_2";
                    choiceMap[whichChoice] = ((Group *)pNode.get())->getChildIndex(c->getNode());
                    //fprintf(stderr, "addChildrenNodes() 2: choiceMap: %d -> %d\n", whichChoice, choiceMap[whichChoice]);
                }
                c->haveToAdd--;
            }
            //cerr << "addch\n";
        }
    }
}

osgViewerObject *osgViewerObject::getParent()
{
    if (level > 0)
    {
        level--;
        return this;
    }
    //cerr << "osgViewerObject::back" << endl;

    return parent;
}

void osgViewerObject::ref()
{
    refcount++;
}

void osgViewerObject::deref()
{
    refcount--;
    if (refcount == 0)
        delete this;
}

osgViewerObject::osgViewerObject(VrmlNode *n)
{
    viewer = NULL;
    MyDoc =NULL;
    node = n;
    refcount = 0;
    level = 0;
    parent = NULL;
    rootNode = NULL;
    pNode = NULL;
    lightedNode = NULL;
    modeNames = NULL;
    haveToAdd = 0;
    aI = shin = trans = 0;
    dC[0] = dC[1] = dC[2] = 0;
    eC[0] = eC[1] = eC[2] = 0;
    sC[0] = sC[1] = sC[2] = 0;
    mtl = NULL;
    sensor = NULL;
    billBoard = NULL;
    whichChoice = -2;
    numSwitchChildren = 0;
    parentTransform.makeIdentity();
    nodeType = NODE_GENERAL;
    sensorObjectToAdd = NULL;
    int i;
    for (i = 0; i < 100; i++)
        choiceMap[i] = -1;
    transparent = false;
    cubeMap = NULL;
    numTextures = 0;
    ntc = 0;

    // XXX das geht so nicht, da braucht man nv combiner glaube ich...
    Vec4 white(1, 1, 1, 1);
    combinerEnv = new TexEnvCombine();
    combinerEnv->setConstantColor(white);
    combinerEnv->setSource0_RGB(TexEnvCombine::TEXTURE0);
    combinerEnv->setSource0_Alpha(TexEnvCombine::TEXTURE0);
    combinerEnv->setSource1_RGB(TexEnvCombine::CONSTANT);
    combinerEnv->setSource1_Alpha(TexEnvCombine::CONSTANT);
    combinerEnv->setSource2_RGB(TexEnvCombine::TEXTURE0);
    combinerEnv->setSource2_Alpha(TexEnvCombine::TEXTURE0);

    combiner = new TexEnvCombine();
    combiner->setSource0_RGB(TexEnvCombine::TEXTURE0);
    combiner->setSource0_Alpha(TexEnvCombine::TEXTURE0);
    combiner->setSource1_RGB(TexEnvCombine::TEXTURE1);
    combiner->setSource1_Alpha(TexEnvCombine::TEXTURE1);
    combiner->setSource2_RGB(TexEnvCombine::TEXTURE2);
    combiner->setSource2_Alpha(TexEnvCombine::TEXTURE2);

    /*
    combinerEnv->setGeneralInput( GL_COMBINER0_NV, GL_RGB, GL_VARIABLE_A_NV, GL_TEXTURE0_ARB, GL_UNSIGNED_IDENTITY_NV,GL_RGB);
    combinerEnv->setGeneralInput( GL_COMBINER0_NV, GL_ALPHA, GL_VARIABLE_A_NV, GL_TEXTURE0_ARB, GL_UNSIGNED_IDENTITY_NV,GL_ALPHA);

    combinerEnv->setGeneralInput( GL_COMBINER0_NV, GL_RGB, GL_VARIABLE_B_NV, GL_CONSTANT_COLOR0_NV, GL_UNSIGNED_IDENTITY_NV,GL_RGB);
    combinerEnv->setGeneralInput( GL_COMBINER0_NV, GL_ALPHA, GL_VARIABLE_B_NV, GL_CONSTANT_COLOR0_NV, GL_UNSIGNED_IDENTITY_NV,GL_ALPHA);

    combinerEnv->setGeneralInput( GL_COMBINER0_NV, GL_RGB, GL_VARIABLE_C_NV, GL_TEXTURE2_ARB, GL_UNSIGNED_IDENTITY_NV,GL_RGB);
    combinerEnv->setGeneralInput( GL_COMBINER0_NV, GL_ALPHA, GL_VARIABLE_C_NV, GL_ZERO, GL_UNSIGNED_IDENTITY_NV,GL_ALPHA);

    combinerEnv->setGeneralInput( GL_COMBINER0_NV, GL_RGB, GL_VARIABLE_D_NV, GL_TEXTURE1_ARB, GL_UNSIGNED_IDENTITY_NV,GL_RGB);
   combinerEnv->setGeneralInput( GL_COMBINER0_NV, GL_ALPHA, GL_VARIABLE_D_NV, GL_ZERO, GL_UNSIGNED_IDENTITY_NV,GL_ALPHA);

   combinerEnv->setGeneralOutput( GL_COMBINER0_NV, GL_RGB,GL_DISCARD_NV, GL_DISCARD_NV, GL_SPARE0_NV, GL_NONE, GL_NONE, GL_FALSE, GL_FALSE, GL_FALSE);
   combinerEnv->setGeneralOutput( GL_COMBINER0_NV, GL_ALPHA,GL_SPARE0_NV, GL_DISCARD_NV, GL_DISCARD_NV, GL_NONE, GL_NONE, GL_FALSE, GL_FALSE, GL_FALSE);

   combinerEnv->setFinalInput( GL_VARIABLE_A_NV, GL_PRIMARY_COLOR_NV, GL_UNSIGNED_IDENTITY_NV, GL_RGB);

   combinerEnv->setFinalInput( GL_VARIABLE_B_NV, GL_SPARE0_NV, GL_UNSIGNED_IDENTITY_NV, GL_RGB);

   combinerEnv->setFinalInput( GL_VARIABLE_C_NV, GL_ZERO, GL_UNSIGNED_IDENTITY_NV, GL_RGB);

   combinerEnv->setFinalInput( GL_VARIABLE_D_NV, GL_ZERO, GL_UNSIGNED_IDENTITY_NV, GL_RGB);
   combinerEnv->setFinalInput( GL_VARIABLE_E_NV, GL_ZERO, GL_UNSIGNED_IDENTITY_NV, GL_RGB);
   combinerEnv->setFinalInput( GL_VARIABLE_F_NV, GL_ZERO, GL_UNSIGNED_IDENTITY_NV, GL_RGB);
   combinerEnv->setFinalInput( GL_VARIABLE_G_NV, GL_SPARE0_NV, GL_UNSIGNED_IDENTITY_NV, GL_ALPHA);

   combiner->setActiveCombiners(1);
   combiner->setActiveConstColors(0);

   combiner->setGeneralInput( GL_COMBINER0_NV, GL_RGB, GL_VARIABLE_A_NV, GL_TEXTURE0_ARB, GL_UNSIGNED_IDENTITY_NV,GL_RGB);

   combiner->setGeneralInput( GL_COMBINER0_NV, GL_RGB, GL_VARIABLE_B_NV, GL_TEXTURE1_ARB, GL_UNSIGNED_IDENTITY_NV,GL_RGB);

   combiner->setGeneralInput( GL_COMBINER0_NV, GL_RGB, GL_VARIABLE_C_NV, GL_TEXTURE2_ARB, GL_UNSIGNED_IDENTITY_NV,GL_RGB);

   combiner->setGeneralInput( GL_COMBINER0_NV, GL_RGB, GL_VARIABLE_D_NV, GL_TEXTURE1_ARB, GL_UNSIGNED_INVERT_NV,GL_RGB);

   combiner->setGeneralOutput( GL_COMBINER0_NV, GL_RGB,GL_DISCARD_NV, GL_DISCARD_NV, GL_SPARE0_NV, GL_NONE, GL_NONE, GL_FALSE, GL_FALSE, GL_FALSE);

   combiner->setFinalInput( GL_VARIABLE_A_NV, GL_PRIMARY_COLOR_NV, GL_UNSIGNED_IDENTITY_NV, GL_RGB);

   combiner->setFinalInput( GL_VARIABLE_B_NV, GL_SPARE0_NV, GL_UNSIGNED_IDENTITY_NV, GL_RGB);

   combiner->setFinalInput( GL_VARIABLE_C_NV, GL_ZERO, GL_UNSIGNED_IDENTITY_NV, GL_RGB);

   combiner->setFinalInput( GL_VARIABLE_D_NV, GL_ZERO, GL_UNSIGNED_IDENTITY_NV, GL_RGB);
   combiner->setFinalInput( GL_VARIABLE_E_NV, GL_ZERO, GL_UNSIGNED_IDENTITY_NV, GL_RGB);
   combiner->setFinalInput( GL_VARIABLE_F_NV, GL_ZERO, GL_UNSIGNED_IDENTITY_NV, GL_RGB);
   combiner->setFinalInput( GL_VARIABLE_G_NV, GL_SPARE0_NV, GL_UNSIGNED_IDENTITY_NV, GL_ALPHA);*/
}

osgViewerObject::~osgViewerObject()
{
    if (auto group = pNode->asGroup())
    {
        group->removeChildren(0, group->getNumChildren());
    }

    if (sensor)
    {
        sensor->remove(); // can't be deleted here, because it could be active right now
    }
    delete[] modeNames;

    for (auto c: children)
    {
        c->deref();
    }
    children.clear();
    childIndex.clear();

    if (pNode.get())
    {
        Group *parentNode;
        while ((pNode->getNumParents()) && (parentNode = pNode->getParent(0)))
        {
            parentNode->removeChild(pNode.get());
        }
    }
    //if(pNode != cover->getScene())
    //pNode->unref();
    if (billBoard.get())
    {
        Group *parentNode;
        while ((billBoard->getNumParents()) && (parentNode = billBoard->getParent(0)))
        {
            parentNode->removeChild(billBoard.get());
        }
        // billBoard->unref();
    }

    if (nodeType == NODE_LIGHT)
    {
        LightSource *ls = dynamic_cast<LightSource *>(pNode.get());
        if (ls)
            coVRLighting::instance()->removeLight(ls);
    }
}

void osgViewerObject::removeChild(osgViewerObject *rmObj)
{
    auto iter = childIndex.find(rmObj->node);
    if (iter != childIndex.end())
        childIndex.erase(iter);

    auto it = std::find(children.begin(), children.end(), rmObj);
    if (it == children.end())
        return;

    children.erase(it);
}

void osgViewerObject::updateBin()
{
    if (Sorted)
    {
        Geode *pGeode = dynamic_cast<Geode *>(pNode.get());
        if (pGeode)
        {
            for (unsigned int i = 0; i < pGeode->getNumDrawables(); i++)
            {
                Drawable *geoset = pGeode->getDrawable(i);
                if (geoset)
                {
                    StateSet *stateset = geoset->getOrCreateStateSet();
                    stateset->setNestRenderBins(false);
                    if (strncmp(pNode->getName().c_str(), "coTransparent", 13) == 0)
                    {
                        //geoset->setDrawBin(PFSORT_TRANSP_BIN);
                        stateset->setRenderingHint(StateSet::TRANSPARENT_BIN);
                        stateset->setNestRenderBins(false);
                    }
                    else if (strncmp(pNode->getName().c_str(), "coOpaque", 8) == 0)
                    {
                        //geoset->setDrawBin(PFSORT_OPAQUE_BIN);
                        stateset->setRenderingHint(StateSet::OPAQUE_BIN);
                        stateset->setNestRenderBins(false);
                    }
                    else if (strncmp(pNode->getName().c_str(), "coDepthOnly", 11) == 0)
                    {
                        //geoset->setDrawBin(DEPTH_BIN);
                        // XXX: DEPTH -> TRANSP ?
                        //stateset->setRenderingHint(StateSet::TRANSPARENT_BIN);
                        stateset->setRenderBinDetails(-1, "RenderBin");
                        stateset->setNestRenderBins(false);
                    }
                    else if (transparent)
                    {
                        //geoset->setDrawBin(PFSORT_TRANSP_BIN);
                        stateset->setRenderingHint(StateSet::TRANSPARENT_BIN);
                        stateset->setNestRenderBins(false);
#if 0
                  stateset->setMode(GL_TEXTURE_2D, StateAttribute::OFF);
                  mtl = new Material;
                  //mtl->setSide(PFMTL_BOTH);
                  mtl->setColorMode( Material::OFF);
                  mtl->setAmbient( Material::FRONT_AND_BACK, Vec4(1.0, 0.0, 0.0, 0.5));
                  mtl->setDiffuse( Material::FRONT_AND_BACK, Vec4(1.0, 0.0, 0.0, 0.5));
                  mtl->setSpecular( Material::FRONT_AND_BACK, Vec4(0.8, 0.8, 0.8, 0.5));
                  mtl->setEmission( Material::FRONT_AND_BACK, Vec4(0.0, 0.0, 0.0, 0.5));
                  //mtl->setShininess( Material::FRONT_AND_BACK, shininess*128);
                  mtl->setAlpha( Material::FRONT_AND_BACK, 0.5);
                  stateset->setAttributeAndModes(mtl,StateAttribute::ON);
#endif

#if 0
                  osg::Depth *NoDepthBuffer = new osg::Depth(osg::Depth::LESS, 0.0, 1.0, false);
                  stateset->setAttributeAndModes(NoDepthBuffer,StateAttribute::ON);
#endif
                        //fprintf(stderr,"\ntransp %s\n",pNode->getName().c_str());
                    }
                    else
                    {
                        //geoset->setDrawBin(PFSORT_OPAQUE_BIN);
                        stateset->setRenderingHint(StateSet::OPAQUE_BIN);
                        //fprintf(stderr,"\nopaque");
                    }
                    geoset->setStateSet(stateset);
                }
            }
        }
    }
}

void osgViewerObject::updateTexGen()
{
    for (int i = 0; i < numTextures; i++)
    {
        if (texData[i].texGen == NULL)
        {
            texData[i].texGen = new TexGen;
        }
        texData[i].texGen->setMode(TexGen::OBJECT_LINEAR);
        //d_currentObject->texData[i].texGen->setMode(PF_T, TexGen::OBJECT_LINEAR);
    }
}

void osgViewerObject::updateTexData(int numActiveTextures)
{
    if (numActiveTextures > texData.size())
    {
        texData.resize(numActiveTextures);
        numTextures = numActiveTextures;
    }
}

void osgViewerObject::updateTexture()
{
    Geode *pGeode = dynamic_cast<Geode *>(pNode.get());
    //fprintf(stderr, "updateTexture: pGeode=%p\n", pGeode);
    if (pGeode)
    {
        Drawable *geoset = pGeode->getDrawable(0);
        if (geoset)
        {
            StateSet *geostate = geoset->getOrCreateStateSet();
            geostate->setNestRenderBins(false);
 
            //fprintf(stderr, "updateTexture: numTextures=%d\n", numTextures);
            for (int i = 0; i < numTextures; i++)
            {
                //fprintf(stderr, "updateTexture:: texImage=%p\n", texData[i].texImage);
                if (texData[i].texture.get())
                {
                    //fprintf(stderr, "updateTexture: unit %d\n", i);
                    // XXX
                    geostate->setTextureAttributeAndModes(i, texData[i].texture.get(), StateAttribute::ON);
                    //                   cerr << "geostate " << geostate << " numTextures:" << geostate->getNumTextures() <<" i: "<< i<< "texture"<< texData[i].texture << endl;
                    geostate->setTextureAttributeAndModes(i, texData[i].tEnv.get(), StateAttribute::ON);
                    if (texData[i].texGen.get())
                    {
                        //fprintf(stderr, "updateTexture: texGen on for unit %d\n", i);
                        geostate->setTextureAttributeAndModes(i, texData[i].texGen.get(), StateAttribute::ON);
                    }
                    geoset->setStateSet(geostate);
                }
                else
                {
                    //fprintf(stderr, "updateTexture: texture==NULL for unit %d\n", i);
                }
            }
            for(unsigned int i=1;i<pGeode->getNumDrawables();i++) // copy the same geostate to the rest of the drawables
            {
                osg::Drawable *d = pGeode->getDrawable(i);
                d->setStateSet(geostate);
            }
        }
    }

// XXX
#if 0
   if((pNode)&&(cubeMap))
   {
      if(pfIsOfType(pNode,Geode::getClassType()))
      {
         pfGeoSet *geoset = ((Geode*)pNode)->getGSet(0);
         if(geoset)
         {
            pfGeoState *geostate = geoset->getGState();
            if (geostate)
            {
               geostate->setMultiMode(PFSTATE_ENTEXTURE, cubeMap->unit, PF_ON);
               if(numTextures > cubeMap->unit)
               {
                  geostate->setMultiAttr(PFSTATE_TEXENV, cubeMap->unit, texData[cubeMap->unit]->tEnv);
                  if(texData[cubeMap->unit]->texGen)
                  {
                     geostate->setMultiMode(PFSTATE_ENTEXGEN, cubeMap->unit, PF_ON);
                     geostate->setMultiAttr(PFSTATE_TEXGEN, cubeMap->unit, texData[cubeMap->unit]->texGen);
                  }
               }
               geoset->setGState(geostate);
            }
         }
         cubeMap->addToNode(pNode);
         //cerr << "updateTexture::added CubeMap2!!!!!!!!!!" << endl;
      }
   }
#endif
}

void osgViewerObject::updateTMat()
{
    bool needTransform=false;
    for (int i = 0; i < numTextures; i++)
    {
        if(texData[i].mirror !=0)
        {
            needTransform=true;
            break;
        }
    }
    if (texData.size() > 0 && ((texData[0].newTMat.compare(osg::Matrix::identity()) != 0) || (needTransform))) // 1 == vertical 2 = horizontal
    {
        Geode *pGeode = dynamic_cast<Geode *>(pNode.get());
        if (pGeode)
        {
            Drawable *drawable = pGeode->getDrawable(0);
            for (int i = 0; i < numTextures; i++)
            {
                if (texData[i].mirror!=0 && drawable)
                {
                    StateSet *geostate = drawable->getOrCreateStateSet();
                    geostate->setNestRenderBins(false);

                    Matrixd multMat;
                    multMat.makeIdentity();
                    int texWidth = 1;
                    int texHeight = 1;
                    if(texData[i].texImage)
                    {
                        while ((texWidth <<= 1) < texData[i].texImage->s())
                        {
                        }
                        while ((texHeight <<= 1) < texData[i].texImage->t())
                        {
                        }
                    }
                    if(texData[i].mirror == 1)
                    {
                        
                            multMat(1, 1) = -1;
                            multMat(3, 1) = 1;
                        
                    }
                    else
                    {
                        
                            multMat(0, 0) = -1;
                            multMat(1, 1) = 1;
                            multMat(3, 0) = 1;
                        
                    }
                    TexMat *texMat = new TexMat();
                    Matrix tmpMat;
                    tmpMat.mult(texData[i].newTMat, multMat);
                    texMat->setMatrix(tmpMat);
                    geostate->setTextureAttributeAndModes(i, texMat, StateAttribute::ON);
                    drawable->setStateSet(geostate);
                }
                else if (drawable)
                {
                    StateSet *geostate = drawable->getOrCreateStateSet();
                    geostate->setNestRenderBins(false);

                    TexMat *texMat = new TexMat();
                    texMat->setMatrix(texData[i].newTMat);
                    geostate->setTextureAttributeAndModes(i, texMat, StateAttribute::ON);
                    drawable->setStateSet(geostate);
                }
            }
        }
    }
}

void osgViewerObject::setMaterial(float ambientIntensity,
                                  float diffuseColor[],
                                  float emissiveColor[],
                                  float shininess,
                                  float specularColor[],
                                  float transparency)
{
    float a = 1.0 - transparency;

    /* if(numTextures==0)                             // if texture available, texture defines transparency, otherwise we could set this to opaque
    {*/
    if (transparency != 0.0)
    {
        if (!transparent)
        {
            transparent = true;
            // updateBin();
        }
    }
    /* else
    {
       if(transparent)
       {
          transparent = false;
          updateBin();
       }
    }*/
    /* }
    else*/
    {
        updateBin();
    }

    if (mtl == NULL)
    {
        mtl = new Material;
        //mtl->setSide(PFMTL_BOTH);
        mtl->setColorMode(Material::AMBIENT_AND_DIFFUSE);
        mtl->setAmbient(Material::FRONT_AND_BACK, Vec4(ambientIntensity, ambientIntensity, ambientIntensity, a));
        mtl->setDiffuse(Material::FRONT_AND_BACK, Vec4(diffuseColor[0], diffuseColor[1], diffuseColor[2], a));
        mtl->setSpecular(Material::FRONT_AND_BACK, Vec4(specularColor[0], specularColor[1], specularColor[2], a));
        mtl->setEmission(Material::FRONT_AND_BACK, Vec4(emissiveColor[0], emissiveColor[1], emissiveColor[2], a));
        mtl->setShininess(Material::FRONT_AND_BACK, shininess * 128);
        mtl->setAlpha(Material::FRONT_AND_BACK, a);
        updateMaterial();
    }
    else
    {
        if ((ambientIntensity != aI)
            || (diffuseColor[0] != dC[0]) || (diffuseColor[1] != dC[1]) || (diffuseColor[2] != dC[2])
            || (emissiveColor[0] != eC[0]) || (emissiveColor[1] != eC[1]) || (emissiveColor[2] != eC[2])
            || (specularColor[0] != sC[0]) || (specularColor[1] != sC[1]) || (specularColor[2] != sC[2])
            || (shininess != shin) || (transparency != trans))
        {
            aI = ambientIntensity;
            trans = transparency;
            shin = shininess;
            dC[0] = diffuseColor[0];
            dC[1] = diffuseColor[1];
            dC[2] = diffuseColor[2];
            eC[0] = emissiveColor[0];
            eC[1] = emissiveColor[1];
            eC[2] = emissiveColor[2];
            sC[0] = specularColor[0];
            sC[1] = specularColor[1];
            sC[2] = specularColor[2];
            mtl->setAmbient(Material::FRONT_AND_BACK, Vec4(ambientIntensity, ambientIntensity, ambientIntensity, a));
            mtl->setDiffuse(Material::FRONT_AND_BACK, Vec4(diffuseColor[0], diffuseColor[1], diffuseColor[2], a));
            mtl->setSpecular(Material::FRONT_AND_BACK, Vec4(specularColor[0], specularColor[1], specularColor[2], a));
            mtl->setEmission(Material::FRONT_AND_BACK, Vec4(emissiveColor[0], emissiveColor[1], emissiveColor[2], a));
            mtl->setShininess(Material::FRONT_AND_BACK, shininess * 128);
            mtl->setAlpha(Material::FRONT_AND_BACK, a);
            updateMaterial(); ///????
        }
    }
}

void osgViewerObject::updateMaterial()
{
    if (modeNames && pNode.get())
    {
        if (modeNames[0] == 'M' && modeNames[1] == '_')
        {
            if (strncmp(modeNames, "M_baked_", 8) == 0)
                ViewerOsg::viewer->setModesByName(modeNames + 8);
            else
                ViewerOsg::viewer->setModesByName(modeNames + 2);
        }
        else
            ViewerOsg::viewer->setModesByName(modeNames);
    }
    if (mtl.get())
    {
        Geode *pGeode = dynamic_cast<Geode *>(pNode.get());
        if (pGeode && pGeode->getNumDrawables())
        {
            Drawable *drawable = pGeode->getDrawable(0);
            if (drawable)
            {
                StateSet *geostate = drawable->getOrCreateStateSet();
                geostate->setNestRenderBins(false);
                geostate->setAttributeAndModes(mtl.get(), StateAttribute::ON);
                if (!transparent)
                {
                    //geostate->setMode(StateAttribute::BLEND, StateAttribute::OFF);
                    geostate->setMode(GL_BLEND, StateAttribute::OFF);
                }
                else
                {
                    if (Blended)
                    {
                        BlendFunc *blendFunc = new BlendFunc();
                        blendFunc->setFunction(BlendFunc::SRC_ALPHA, BlendFunc::ONE_MINUS_SRC_ALPHA);
                        geostate->setAttributeAndModes(blendFunc, StateAttribute::ON);

                        AlphaFunc *alphaFunc = new AlphaFunc(AlphaFunc::GREATER, 0.0);
                        if (AlphaTest)
                        {
                            geostate->setAttributeAndModes(alphaFunc, StateAttribute::ON);
                        }
                        else
                        {
                            geostate->setAttributeAndModes(alphaFunc, StateAttribute::OFF);
                        }
                    }
                }
                drawable->setStateSet(geostate);
                
                for(unsigned int i=1;i<pGeode->getNumDrawables();i++) // copy the same geostate to the rest of the drawables
                {
                    osg::Drawable *d = pGeode->getDrawable(i);
                    d->setStateSet(geostate);
                }
            }
        }
    }
}

void osgViewerObject::setTexGen(int environment, int textureNumber, int /*blendMode*/)
{
    if (environment == 1 && texData[textureNumber].texGen == NULL)
    {
        texData[textureNumber].texGen = new TexGen();
        texData[textureNumber].texGen->setMode(TexGen::SPHERE_MAP);
    }
    if (environment == 2 && texData[textureNumber].texGen == NULL)
    {
        texData[textureNumber].texGen = new TexGen();
        texData[textureNumber].texGen->setMode(TexGen::REFLECTION_MAP);
    }
    if (environment == 3 && texData[textureNumber].texGen == NULL)
    {
        texData[textureNumber].texGen = new TexGen();
        texData[textureNumber].texGen->setMode(TexGen::OBJECT_LINEAR);
        Plane sPlane(1, 0, 0, 0);
        Plane tPlane(0, 1, 0, 0);
        Plane rPlane(0, 0, 1, 0);
        texData[textureNumber].texGen->setPlane(TexGen::S, sPlane);
        texData[textureNumber].texGen->setPlane(TexGen::T, tPlane);
        texData[textureNumber].texGen->setPlane(TexGen::R, rPlane);
    }
    if (environment == 4 && texData[textureNumber].texGen == NULL)
    {
        texData[textureNumber].texGen = new TexGen();
        texData[textureNumber].texGen->setMode(TexGen::NORMAL_MAP);
    }
    if (environment == 5 && texData[textureNumber].texGen == NULL)
    {
        texData[textureNumber].texGen = new TexGen();
        texData[textureNumber].texGen->setMode(TexGen::EYE_LINEAR);
        Plane sPlane(1, 0, 0, 0);
        Plane tPlane(0, 1, 0, 0);
        Plane rPlane(0, 0, 1, 0);
        texData[textureNumber].texGen->setPlane(TexGen::S, sPlane);
        texData[textureNumber].texGen->setPlane(TexGen::T, tPlane);
        texData[textureNumber].texGen->setPlane(TexGen::R, rPlane);
    }
    updateTexture();
}

int osgViewerObject::getBlendModeForVrmlNode(const char *modeString)
{
    int blendMode = 1;

    if (strncasecmp(modeString, "MODULATE", 8) == 0)
        blendMode = 1;
    else if (strncasecmp(modeString, "BLEND", 5) == 0)
        blendMode = 2;
    else if (strncasecmp(modeString, "DECAL", 5) == 0)
        blendMode = 3;
    else if (strncasecmp(modeString, "REPLACE", 7) == 0)
        blendMode = 4;
    else if (strncasecmp(modeString, "ADD", 3) == 0)
        blendMode = 5;
    else if (strncasecmp(modeString, "ALPHA", 5) == 0)
        blendMode = TexEnv::BLEND;
    else if (strncasecmp(modeString, "SELECTARG1", 10) == 0)
        blendMode = 0;
    else if (strncasecmp(modeString, "SELECTARG2", 10) == 0)
        blendMode = 0;
    else
    {
        blendMode = 1;
        cerr << "blendMode " << modeString << " is not supported by OpenSceneGraph. Using TexEnv::MODULATE instead\n" << endl;
    }
    return blendMode;
}

void osgViewerObject::setTexEnv(int environment, int textureNumber, int blendMode, int nc)
{
    if (texData[textureNumber].tEnv == NULL)
    {
        texData[textureNumber].tEnv = tEnvModulate.get();
        if (blendMode)
        {
            if (blendMode == 1)
                texData[textureNumber].tEnv = tEnvModulate.get();
            else if (blendMode == 2)
                texData[textureNumber].tEnv = tEnvBlend.get();
            else if (blendMode == 3)
                texData[textureNumber].tEnv = tEnvDecal.get();
            else if (blendMode == 4)
                texData[textureNumber].tEnv = tEnvReplace.get();
            else if (blendMode == 5)
                texData[textureNumber].tEnv = tEnvAdd.get();
        }
        else if ((nc == 3) && (textureMode == TexEnv::DECAL))
        {
            texData[textureNumber].tEnv = tEnvDecal.get();
        }
        else
        {
            if (environment)
            {
                texData[textureNumber].tEnv = tEnvAdd.get();
            }
            else
            {
                if (textureMode == TexEnv::MODULATE)
                    texData[textureNumber].tEnv = tEnvModulate.get();
                else if (textureMode == TexEnv::BLEND)
                    texData[textureNumber].tEnv = tEnvBlend.get();
                else if (textureMode == TexEnv::DECAL)
                    texData[textureNumber].tEnv = tEnvDecal.get();
                else if (textureMode == TexEnv::REPLACE)
                    texData[textureNumber].tEnv = tEnvReplace.get();
                else if (textureMode == TexEnv::ADD)
                    texData[textureNumber].tEnv = tEnvAdd.get();
            }
        }
        updateTexture();
    }
}

textureData::textureData()
    : ntc(0)
    , mirror(false)
{
    texture = NULL;
    tEnv = NULL;
    texImage = NULL;
    newTMat.makeIdentity();
    texGen = NULL;
}

textureData::~textureData()
{
}
