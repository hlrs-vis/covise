/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/************************************************************************
 *                                                                      *
 *                                                                      *
 *                            (C) 1996                                  *
 *              Computer Centre University of Stuttgart                 *
 *                         Allmandring 30                               *
 *                       D-70550 Stuttgart                              *
 *                            Germany                                   *
 *                                                                      *
 *                                                                      *
 *   File         VRCoviseGeometryManager.C                             *
 *                                                                      *
 *   Description      add/delete objects to/from the scene graph        *
 *                                                                      *
 *   Author         D. Rainer                                           *
 *                                                                      *
 *   Date         20.08.97                                              *
 *            24.02.98 large indexlists for tristrips                   *
 *            27.02.98 large indexlists for polygons                    *
 *                                                                      *
 ************************************************************************/
#include <util/common.h>

#include <config/CoviseConfig.h>
#include <osg/Sequence>
#include <osg/Geode>
#include <osg/Array>
#include <osg/Depth>
#include <osg/Material>
#include <osg/Group>
#include <osg/Array>
#include <osg/MatrixTransform>
#include <osg/TexEnv>
#include <osg/Texture>
#include <osg/Texture2D>
#include <osg/TextureRectangle>
#include <osg/Geometry>
#include <osg/PrimitiveSet>
#include <osg/CullFace>
#include <osg/BlendFunc>
#include <osg/LineWidth>
#include <osg/Point>
#include <osg/AlphaFunc>
#include <osg/ColorMask>
#include <osg/PolygonOffset>
#include <osgUtil/TriStripVisitor>
#include <cover/coVRFileManager.h>
#include "VRCoviseGeometryManager.h"
#include <cover/coVRLighting.h>
#include <cover/VRSceneGraph.h>
#include <cover/coVRMSController.h>
#include <cover/coVRPluginSupport.h>
#include <cover/coVRConfig.h>
#include <PluginUtil/coSphere.h>
#include <cover/RenderObject.h>
#include <do/coDoData.h>

#include <osg/Program>
#include <osg/Shader>
#include <osg/Uniform>

using namespace opencover;
using namespace covise;
void createXYPlane(int xmin, int xmax, int ymin, int ymax, int zmin, int xsize, int ysize,
                   osg::Vec3 *coordArray);

GeometryManager *GeometryManager::instance()
{
    static GeometryManager *singleton = NULL;
    if (!singleton)
        singleton = new GeometryManager();
    return singleton;
}

GeometryManager::GeometryManager()
{
    backfaceCulling = coCoviseConfig::isOn("COVER.BackfaceCulling", false);

    d_stripper = new osgUtil::TriStripVisitor;
    genStrips = coCoviseConfig::isOn("COVER.GenStrips", false);

    float r = coCoviseConfig::getFloat("r", "COVER.CoviseGeometryDefaultColor", 1.0f);
    float g = coCoviseConfig::getFloat("g", "COVER.CoviseGeometryDefaultColor", 1.0f);
    float b = coCoviseConfig::getFloat("b", "COVER.CoviseGeometryDefaultColor", 1.0f);
    coviseGeometryDefaultColor = osg::Vec4(r, g, b, 1.0f);
}

osg::Group *
GeometryManager::addGroup(const char *object, bool is_timestep)
{
    osg::Group *group = NULL;
    if (is_timestep) // set is a sequence
    {
#ifdef DBGPRINT
        printf("add sequence\n");
        printf("\t sequence name  = %s\n", object);
#endif

        group = new osg::Sequence;
    }
    else
    {
#ifdef DBGPRINT
        printf("adding a set\n");
        printf("\t set name  = %s\n", object);
#endif

        group = new osg::Group;
    }

    group->setName(object);

    return group;
}

osg::Node *GeometryManager::addUGrid(const char *object,
        int xsize, int ysize, int zsize,
        float xmin, float xmax, float ymin, float ymax, float zmin, float zmax,
        int no_c, int colorbinding, int colorpacking,
        float *rc, float *gc, float *bc, int *pc,
        int no_n, int normalbinding,
        float *xn, float *yn, float *zn, float &transparency)
{
    //std::cerr << "Adding a uniform grid..." << std::endl;
    osg::Geode *geode = NULL;

    if (xsize>1 && ysize>1 && zsize>1)
        return geode;

    if (colorbinding != Bind::PerVertex)
        return geode;

    if (no_c != xsize*ysize*zsize)
        return geode;

    geode = new osg::Geode();
    geode->setName(object);
    osg::Geometry *geom = new osg::Geometry();
    geom->setUseDisplayList(coVRConfig::instance()->useDisplayLists());
    geom->setUseVertexBufferObjects(coVRConfig::instance()->useVBOs());

    // set up geometry
    osg::Vec3Array *vert = new osg::Vec3Array;
    osg::Vec3Array *normal = new osg::Vec3Array;
    osg::DrawArrayLengths *primitives = new osg::DrawArrayLengths(osg::PrimitiveSet::QUADS);
    primitives->push_back(4);

    int flatDim = 0;
    if (xsize == 1)
        flatDim = 0;
    else if (ysize == 1)
        flatDim = 1;
    else if (zsize == 1)
        flatDim = 2;

    int w=0, h=0;
    switch (flatDim)
    {
        case 0:
        {
            w = ysize;
            h = zsize;
            float x = (xmin+xmax)*.5;
            vert->push_back(osg::Vec3(x, ymin, zmin));
            vert->push_back(osg::Vec3(x, ymin, zmax));
            vert->push_back(osg::Vec3(x, ymax, zmax));
            vert->push_back(osg::Vec3(x, ymax, zmin));
            normal->push_back(osg::Vec3(1., 0., 0.));
            break;
        }
        case 1:
        {
            w = xsize;
            h = zsize;
            float y = (ymin+ymax)*.5;
            vert->push_back(osg::Vec3(xmin, y, zmin));
            vert->push_back(osg::Vec3(xmin, y, zmax));
            vert->push_back(osg::Vec3(xmax, y, zmax));
            vert->push_back(osg::Vec3(xmax, y, zmin));
            normal->push_back(osg::Vec3(0., 1., 0.));
            break;
        }
        case 2:
        {
            w = xsize;
            h = ysize;
            float z = (zmin+zmax)*.5;
            vert->push_back(osg::Vec3(xmin, ymin, z));
            vert->push_back(osg::Vec3(xmin, ymax, z));
            vert->push_back(osg::Vec3(xmax, ymax, z));
            vert->push_back(osg::Vec3(xmax, ymin, z));
            normal->push_back(osg::Vec3(0., 0., 1.));
            break;
        }
    }

    geom->setVertexArray(vert);
    geom->addPrimitiveSet(primitives);

    osg::Vec4Array *color = new osg::Vec4Array(1);
    (*color)    [0].set(1, 1, 0, 1.0f);
    geom->setColorArray(color);
    geom->setColorBinding(osg::Geometry::BIND_OVERALL);

    geom->setNormalArray(normal);
    geom->setNormalBinding(osg::Geometry::BIND_OVERALL);


    osg::TextureRectangle *tex = new osg::TextureRectangle;
    osg::Image *img = new osg::Image();
#ifndef BYTESWAP
    if (colorpacking == Pack::RGBA)
    {
        img->setImage(w, h, 1, GL_RGBA, GL_RGBA, GL_UNSIGNED_BYTE, (unsigned char *)pc, osg::Image::NO_DELETE);
        no_c = 0;
    }
    else
#endif
    {
        img->allocateImage(w, h, 1, GL_RGBA, GL_UNSIGNED_BYTE);
    }
    tex->setImage(img);
    img->setPixelBufferObject(new osg::PixelBufferObject(img));
    tex->setInternalFormat( GL_RGBA );
    tex->setBorderWidth( 0 );
    tex->setFilter( osg::Texture::MIN_FILTER, osg::Texture::LINEAR );
    tex->setFilter( osg::Texture::MAG_FILTER, osg::Texture::LINEAR );
    tex->setWrap(osg::Texture2D::WRAP_S, osg::Texture::CLAMP_TO_EDGE);
    tex->setWrap(osg::Texture2D::WRAP_T, osg::Texture::CLAMP_TO_EDGE);

    // associate colors
    bool transparent = false;
    if (no_c > 0)
    {
        switch (colorbinding)
        {
        case Bind::PerVertex:
        {
            int dims[] = { xsize, ysize, zsize };
            unsigned char *rgba = img->data();
            for (int z=0; z<zsize; ++z)
            {
                for (int y=0; y<ysize; ++y)
                {
                    for (int x=0; x<xsize; ++x)
                    {
                        int idx = covise::coIndex(x, y, z, dims);
                        if (colorpacking == Pack::RGBA)
                        {
                            float r, g, b, a;
                            unpackRGBA(pc, idx, &r, &g, &b, &a);
                            if (a < 1.0f)
                            {
                                transparent = true;
                            }
                            rgba[0] = r*255.99f;
                            rgba[1] = g*255.99f;
                            rgba[2] = b*255.99f;
                            rgba[3] = a*255.99f;
                        }
                        else
                        {
                            if (transparency > 0.f)
                                transparent = true;
                            rgba[0] = rc[idx]*255.99f;
                            rgba[1] = gc[idx]*255.99f;
                            rgba[2] = bc[idx]*255.99f;
                            rgba[3] = (1.f-transparency)*255.99f;
                        }
                        ++idx;
                        rgba += 4;
                    }
                }
            }
            //fprintf(stderr,"COVER INFO: colorbinding per vertex\n");

        }
        break;
        }
    }
    img->dirty();

   osg::Vec2Array *texcoord  = new osg::Vec2Array(4);
   (*texcoord)[0].set(0.0,0.0);
   (*texcoord)[1].set(0.0,h);
   (*texcoord)[2].set(w,h);
   (*texcoord)[3].set(w,0.0);
   geom->setTexCoordArray(0, texcoord);

   osg::TexEnv * texEnv = new osg::TexEnv();
   texEnv->setMode(osg::TexEnv::REPLACE);

   osg::StateSet *stateSet = geode->getOrCreateStateSet();
   stateSet->setMode(GL_LIGHTING, osg::StateAttribute::OFF);
   stateSet->setTextureAttributeAndModes(0, tex, osg::StateAttribute::ON);
   stateSet->setTextureAttribute(0, texEnv);
   geode->setStateSet(stateSet);

    geode->addDrawable(geom);

    return geode;
}

osg::Node *
GeometryManager::addRGrid(const char *, int, int,
                          int, float *, float *,
                          float *,
                          int, int, int,
                          float *, float *, float *, int *, int,
                          int,
                          float *, float *, float *, float &)

{

    if (coVRMSController::instance()->isMaster())
    {
        //CoviseBase::sendInfo("Adding a rectilinear grid... not implemented");
    }

    return ((osg::Node *)NULL);
}

osg::Node *
GeometryManager::addSGrid(const char *object,
                          int xsize, int ysize, int zsize,
                          float *x_c, float *y_c, float *z_c,
                          int no_of_colors, int colorbinding, int colorpacking,
                          float *r, float *g, float *b, int *pc,
                          int no_of_normals, int normalbinding,
                          float *nx, float *ny, float *nz,
                          float &transparency)
{
    //cerr << "addSGrid: not implemented" << endl;
    (void)object;
    (void)xsize;
    (void)ysize;
    (void)zsize;
    (void)x_c;
    (void)y_c;
    (void)z_c;
    (void)no_of_colors;
    (void)colorbinding;
    (void)colorpacking;
    (void)r;
    (void)g;
    (void)b;
    (void)pc;
    (void)no_of_normals;
    (void)normalbinding;
    (void)nx;
    (void)ny;
    (void)nz;
    (void)transparency;

    return NULL;

#if 0
   osg::Geode         *sgridGeode;
   osg::osg::Geometry        *sgridGeoSet;
   static osg::osg::Vec3Array   *coordArray;
   static osg::osg::Vec4Array   *colorArray;
   static osg::osg::Vec3Array *normalArray;
   static int      *primLengthsArray;
   static osg::osg::Vec4Array   *defaultColorArray;
   static ushort   *defaultColorIndexArray;
   long            i, j, k,l,no_of_coords,i_s,j_s;
   static ushort   *v_l_short, *colorIndexArray;
   // osg::Group    *parent;
   osg::Vec3     v1,  v2,  n,ns;
   //BoundingSphere   bsphere;
   long    warningFlag = false;

   if(zsize==1)
   {
      i_s=xsize;
      j_s=ysize;
   }
   else if(ysize==1)
   {
      i_s=xsize;
      j_s=zsize;
   }
   else
   {
      i_s=zsize;
      j_s=ysize;
   }
#ifdef DBGPRINT
   printf("GeometryManager::addStructuredGrid\n");
   printf("\t geode name  = %s\n", object);
   if(parent!=NULL)
      printf("\t root name = %s\n", rootName);
   else
      printf("\t root name = NULL\n");
#endif

   sgridGeode = new osg::Geode;
   sgridGeode->setName(object);

   sgridGeoSet = new osg::osg::Geometry;

   // coordinates
   no_of_coords=i_s*j_s;
   coordArray =  new osg::osg::Vec3Array(no_of_coords);

   for (i = 0; i < no_of_coords; i++)
   {
      (*coordArray)[i].set(x_c[i], y_c[i], z_c[i]);

   }

   // index into coordArray
   k=0;
   l=0;
   v_l_short = new ushort(2*j_s*(i_s-1));
   for (i = 0; i<(i_s-1); i++)
   {
      for (j = 0; j<j_s; j++)
      {
         if ((l+j_s >= 65535) && (warningFlag == false) )
         {
            printf("\n\t ********* WARNING: geoset index contains to many elements **********\n");
            printf("\t *********          geometry will be corrupted         **********\n\n");
            warningFlag = true;
            break;
         }
         v_l_short[k] =  l;
         k++;
         v_l_short[k] =  l+j_s;
         k++;
         l++;

      }
   }

   if(colorbinding == Bind::PerVertex || colorbinding == Bind::OverAll || colorbinding == Bind::None)
   {
      osg::Vec3Array *vert = new osg::Vec3Array;
      osg::DrawArrayLengths *primitives = new osg::DrawArrayLengths(osg::PrimitiveSet::QUAD_STRIP);
      for(int i=0; i<i_s; i++)
      {
         for(int j=0; j<j_s; j++)
         {
            vert->push_back(osg::Vec3(x_c[l], y_c[l], z_c[l]));
            vert->push_back(osg::Vec3(x_c[l+j_s], y_c[l+j_s], z_c[l+j_s]));
         }
         primitives->push_back(j_s);
      }
      sgridGeoSet->setVertexArray(vert);
      sgridGeoSet->addPrimitiveSet(primitives);
   }
   else
   {
   }
   //sgridGeoSet->setAttr(PFGS_COORD3, PFGS_PER_VERTEX, coordArray, v_l_short);

   // Array of primitive lengths

#if 0
   primLengthsArray = new int[(i_s-1)];

   for (i = 0; i< (i_s-1); i++)
   {
      primLengthsArray[i] = j_s*2;
   }

   sgridGeoSet->setPrimLengths(primLengthsArray);
#endif

   // colors

   switch (colorbinding)
   {

      case Bind::PerVertex:

#ifdef DBGPRINT
         printf("\t colorbinding = Bind::PerVertex\n");
#endif
         colorArray = new osg::osg::Vec4Array(no_of_colors);

         for (i = 0;i < no_of_colors; i++)
         {
            if (colorpacking == Pack::RGBA)
            {
               float           r,g,b,a;

               unpackRGBA(pc, i, &r, &g, &b, &a);
               (*colorArray)[i].set(r, g , b , a);
               if (a<1.0)
                  transparency=1.0-a;
            }

            else
               (*colorArray)[i].set(r[i], g[i], b[i], 1-transparency);
         }

         sgridGeoSet->setAttr(PFGS_COLOR4, PFGS_PER_VERTEX, colorArray, (ushort*) v_l_short);
         break;

      case Bind::OverAll:
#ifdef DBGPRINT
         printf("\t colorbinding = Bind::OverAll\n");
#endif
         colorArray = new osg::osg::Vec4Array(1);

         if (colorpacking == Pack::RGBA)
         {
            float           r,g,b,a;
            unpackRGBA(pc, 0, &r, &g, &b, &a);
            (*colorArray)[0].set( r, g , b ,a);
            if (a<1.0)
               transparency=1.0-a;

         }

         else
            (*colorArray)[0].set(r[0], g[0], b[0], 1-transparency);

         colorIndexArray = new ushort(1);
         colorIndexArray[0]=0;
         sgridGeoSet->setAttr(PFGS_COLOR4, PFGS_OVERALL, colorArray, colorIndexArray);
         break;

      case Bind::PerFace:
         printf("\t colorbinding = Bind::PerFace not supported!\n");
         break;

      case Bind::None:
#ifdef DBGPRINT
         printf("\t colorbinding = Bind::None\n");
#endif
         defaultColorArray = new osg::osg::Vec4Array(1);
         defaultColorIndexArray = new ushort(1);
         defaultColorIndexArray[0]=0;
         (*defaultColorArray)[0].set(0.8f, 0.8f, 0.8f, 1-transparency);
         sgridGeoSet->setAttr(PFGS_COLOR4, PFGS_OVERALL, defaultColorArray, defaultColorIndexArray);
   }

   switch (normalbinding)
   {
   case Bind::PerVertex:

         normalArray = new osg::osg::Vec3Array(no_of_normals);

         for (i = 0;i < no_of_normals; i++)
         {
            (*normalArray)[i].set(nx[i], ny[i], nz[i]);
            (*normalArray)[i].normalize();
         }

         sgridGeoSet->setAttr(PFGS_NORMAL3, PFGS_PER_VERTEX, normalArray, (ushort*) v_l_short);

         break;

   case Bind::OverAll:
         printf("\t normalbinding = Bind::OverAll\n");

         // macht keinen Sinn ???

         break;

   case Bind::PerFace:
         printf("\t normalbinding = Bind::PerFace\n");

         break;

      case Bind::None:

#ifdef DBGPRINT
         printf("\t normalbinding = Bind::None...creating normals PER_VERTEX\n");
#endif
         //create normals PER_PRIM

         normalArray = new osg::osg::Vec3Array(2*j_s*(i_s-1));

         k=0;
         for (i = 0; i < i_s; i++)
         {
            for (j = 0; j < j_s; j++)
            {
               ns[0]=ns[1]=ns[2]=0.0;
               if((i>0)&&(j>0))
               {
                  v1 = (*coordArray))[(i-1)*j_s+j] - (*coordArray)[i*j_s+j];
                     v2 = (*coordArray)[i*j_s+j-1] - (*coordArray)[(i-1)*j_s+j];
                     n =v1^v2;
                     ns[0]+=n[0];
                     ns[1]+=n[1];
                     ns[2]+=n[2];
               }
               if((i>0)&&(j<(j_s-1)))
               {
                  v1 =  (*coordArray)[i*j_s+j+1] - (*coordArray)[i*j_s+j];
                     v2 =  (*coordArray)[(i-1)*j_s+j] - (*coordArray)[i*j_s+j+1];
                     n =v1^v2;
                     ns[0]+=n[0];
                     ns[1]+=n[1];
                     ns[2]+=n[2];
               }
               if((i<(i_s-1))&&(j>0))
               {
                  v1 = (*coordArray)[i*j_s+j-1] - (*coordArray)[i*j_s+j];
                     v2 =  (*coordArray)[(i+1)*j_s+j] - (*coordArray)[i*j_s+j-1];
                     n =v1^v2;
                     ns[0]+=n[0];
                     ns[1]+=n[1];
                     ns[2]+=n[2];
               }
               if((i<(i_s-1))&&(j<(j_s-1)))
               {
                  v1 = (*coordArray)[(i+1)*j_s+j] - (*coordArray)[i*j_s+j];
                     v2 = (*coordArray)[i*j_s+j+1] - (*coordArray)[(i+1)*j_s+j];
                     n =v1^v2;
                     ns[0]+=n[0];
                     ns[1]+=n[1];
                     ns[2]+=n[2];
               }
               ns.normalize();
                  if(i<(i_s-1))
                     (*normalArray)[i*2*j_s+j*2] = ns;
                     if(i>0)
                        (*normalArray)[(i-1)*2*j_s+j*2+1] = ns;
            }

         }
         //pfGSetAttr(triangleGeoSet, PFGS_NORMAL3, PFGS_PER_VERTEX, normalArray, (ushort*) v_l_short );
         break;
   }

   sgridGeoSet->setPrimType(PFGS_TRISTRIPS);
      sgridGeoSet->setNumPrims(i_s);

                                                  //pfGetSharedArena());
      osg::osg::StateSet *geostate = loadDefaultGeostate();
      sgridGeoSet->setGState(  geostate);

      if (sgridGeoSet->getNumPrims() > 0)
         sgridGeode->addDrawable(sgridGeoSet);

   //pfPrint(triangleGeode,PFTRAV_SELF|PFTRAV_DESCEND,PFPRINT_VB_DEBUG,NULL);

   //pfPrint(triangleGeode, PFTRAV_SELF|PFTRAV_DESCEND, PFPRINT_VB_DEBUG, NULL);
      VRSceneGraph::instance()->addGeode(sgridGeode, rootName);

   // scale scene to see all
   //ViewState->viewAll = true;

      return ((osg::Node *)sgridGeode);
#endif
}

void GeometryManager::setDefaultMaterial(osg::StateSet *geoState, bool transparent, coMaterial *material, bool isLightingOn)
{
    if (globalDefaultMaterial.get() == NULL)
    {
        globalDefaultMaterial = new osg::Material;
        globalDefaultMaterial->setColorMode(osg::Material::AMBIENT_AND_DIFFUSE);
        globalDefaultMaterial->setAmbient(osg::Material::FRONT_AND_BACK, osg::Vec4(0.2f, 0.2f, 0.2f, 1.0));
        globalDefaultMaterial->setDiffuse(osg::Material::FRONT_AND_BACK, osg::Vec4(1.0f, 1.0f, 1.0f, 1.0));
        globalDefaultMaterial->setSpecular(osg::Material::FRONT_AND_BACK, osg::Vec4(0.4f, 0.4f, 0.4f, 1.0));
        globalDefaultMaterial->setEmission(osg::Material::FRONT_AND_BACK, osg::Vec4(0.0f, 0.0f, 0.0f, 1.0));
        globalDefaultMaterial->setShininess(osg::Material::FRONT_AND_BACK, 16.0f);
    }

    if (material)
    {
        osg::Material *mymtl = new osg::Material;
        mymtl->setColorMode(osg::Material::OFF);
        mymtl->setAmbient(osg::Material::FRONT_AND_BACK, osg::Vec4(material->ambientColor[0], material->ambientColor[1], material->ambientColor[2], 1.0 - material->transparency));
        mymtl->setDiffuse(osg::Material::FRONT_AND_BACK, osg::Vec4(material->diffuseColor[0], material->diffuseColor[1], material->diffuseColor[2], 1.0 - material->transparency));
        mymtl->setSpecular(osg::Material::FRONT_AND_BACK, osg::Vec4(material->specularColor[0], material->specularColor[1], material->specularColor[2], 1.0 - material->transparency));
        mymtl->setEmission(osg::Material::FRONT_AND_BACK, osg::Vec4(material->emissiveColor[0], material->emissiveColor[1], material->emissiveColor[2], 1.0 - material->transparency));
        mymtl->setShininess(osg::Material::FRONT_AND_BACK, material->shininess * 128);
        geoState->setAttributeAndModes(mymtl, osg::StateAttribute::ON);
        transparent = transparent || (material->transparency > 0.0f && material->transparency < 1.0);
    }
    else
    {
        geoState->setAttributeAndModes(globalDefaultMaterial.get(), osg::StateAttribute::ON);
    }

    if (transparent)
    {
        geoState->setRenderingHint(osg::StateSet::TRANSPARENT_BIN);
        geoState->setMode(GL_BLEND, osg::StateAttribute::ON);
    }
    else
    {
        geoState->setRenderingHint(osg::StateSet::OPAQUE_BIN);
        geoState->setMode(GL_BLEND, osg::StateAttribute::OFF);
    }
    geoState->setNestRenderBins(false);

    if (isLightingOn)
    {
        geoState->setMode(GL_LIGHTING, osg::StateAttribute::ON);
    }
    else
    {
        geoState->setMode(GL_LIGHTING, osg::StateAttribute::OFF);
    }
}
//--------------------------------------------------------------------------------------
// create a new geode contraining polygons and add it to the scene
//--------------------------------------------------------------------------------------

osg::Node *
GeometryManager::addPolygon(const char *object_name,
                            int no_of_polygons, int no_of_vertices, int no_of_coords,
                            float *x_c, float *y_c, float *z_c,
                            int *v_l, int *i_l,
                            int no_of_colors, int colorbinding, int colorpacking,
                            float *r, float *g, float *b, int *pc,
                            int no_of_normals, int normalbinding,
                            float *nx, float *ny, float *nz,
                            float &transparency, int, coMaterial *material, int texWidth, int texHeight, int pixelSize, unsigned char *image,
                            int no_of_texCoords, float *tx, float *ty, osg::Texture::WrapMode wm, osg::Texture::FilterMode minfm, osg::Texture::FilterMode magfm,
                            int no_of_vertexAttributes,
                            float *vax, float *vay, float *vaz, bool cullBackfaces)
{
    if ((no_of_polygons == 0) || (no_of_coords == 0) || (no_of_vertices == 0))
    {
        osg::Group *g = new osg::Group(); // add a dummy object so that we don`t have missing timesteps if object is empty
        g->setName(object_name);
        return g;
    }

    osg::Geode *geode = new osg::Geode();
    geode->setName(object_name);
    osg::Geometry *geom = new osg::Geometry();
    geom->setUseDisplayList(coVRConfig::instance()->useDisplayLists());
    geom->setUseVertexBufferObjects(coVRConfig::instance()->useVBOs());

    // set up geometry
    int numv;
    osg::Vec3Array *vert = new osg::Vec3Array;
    osg::DrawArrayLengths *primitives = new osg::DrawArrayLengths(osg::PrimitiveSet::POLYGON);
    for (int i = 0; i < no_of_polygons; i++)
    {
        if (i == no_of_polygons - 1)
            numv = no_of_vertices - i_l[i];
        else
            numv = i_l[i + 1] - i_l[i];
        primitives->push_back(numv);
        for (int n = 0; n < numv; n++)
        {
            int v = v_l[i_l[i] + n];
            vert->push_back(osg::Vec3(x_c[v], y_c[v], z_c[v]));
        }
    }
    geom->setVertexArray(vert);
    geom->addPrimitiveSet(primitives);

    // associate colors
    bool transparent = false;
    if (no_of_colors && material == NULL && image == NULL) // material should overwrite object colors, so ignore them if a material is present
    {

        switch (colorbinding)
        {
        case Bind::PerVertex:
        {
            //fprintf(stderr,"COVER INFO: colorbinding per vertex\n");

            osg::Vec4Array *colArr = new osg::Vec4Array();

            for (int i = 0; i < no_of_polygons; i++)
            {
                if (i == no_of_polygons - 1)
                    numv = no_of_vertices - i_l[i];
                else
                    numv = i_l[i + 1] - i_l[i];
                for (int n = 0; n < numv; n++)
                {
                    int v = v_l[i_l[i] + n];
                    if (colorpacking == Pack::RGBA)
                    {
                        float r, g, b, a;
                        unpackRGBA(pc, v, &r, &g, &b, &a);
                        if (a < 1.0)
                        {
                            transparent = true;
                        }
                        colArr->push_back(osg::Vec4(r, g, b, a));
                    }
                    else
                    {
                        if (transparency > 0.f)
                            transparent = true;
                        colArr->push_back(osg::Vec4(r[v], g ? g[v] : r[v], b ? b[v] : r[v], 1.0f - transparency));
                    }
                }
            }
            geom->setColorArray(colArr);
            geom->setColorBinding(osg::Geometry::BIND_PER_VERTEX);
        }
        break;

        case Bind::OverAll:
        {
            if (colorpacking == Pack::RGBA)
            {
                osg::Vec4Array *colArr = new osg::Vec4Array();
                float r, g, b, a;
                unpackRGBA(pc, 0, &r, &g, &b, &a);
                if (a < 1.0)
                {
                    transparent = true;
                }
                colArr->push_back(osg::Vec4(r, g, b, a));

                geom->setColorArray(colArr);
            }
            else
            {
                osg::Vec4Array *colArr = new osg::Vec4Array();
                if (transparency > 0.f)
                    transparent = true;
                colArr->push_back(osg::Vec4(r[0], g ? g[0] : r[0], b ? b[0] : r[0], 1.0f - transparency));
                geom->setColorArray(colArr);
            }
            geom->setColorBinding(osg::Geometry::BIND_OVERALL);
        }
        break;

        case Bind::PerFace:
        {
            osg::Vec4Array *colArr = new osg::Vec4Array();
            if (colorpacking == Pack::RGBA)
            {
                for (int i = 0; i < no_of_polygons; i++)
                {
                    float r, g, b, a;
                    unpackRGBA(pc, i, &r, &g, &b, &a);
                    if (a < 1.0)
                    {
                        transparent = true;
                    }
                    int numv;
                    if (i == no_of_polygons - 1)
                        numv = no_of_vertices - i_l[i];
                    else
                        numv = i_l[i + 1] - i_l[i];
                    for (int j = 0; j < numv; ++j)
                        colArr->push_back(osg::Vec4(r, g, b, a));
                }
            }
            else
            {
                for (int i = 0; i < no_of_polygons; i++)
                {
                    if (transparency > 0.f)
                        transparent = true;

                    int numv;
                    if (i == no_of_polygons - 1)
                        numv = no_of_vertices - i_l[i];
                    else
                        numv = i_l[i + 1] - i_l[i];
                    for (int j = 0; j < numv; ++j)
                        colArr->push_back(osg::Vec4(r[i], g ? g[i] : r[i], b ? b[i] : r[i], 1.0f - transparency));
                }
            }
            geom->setColorArray(colArr);
            geom->setColorBinding(osg::Geometry::BIND_PER_VERTEX);
        }
        break;
        }
    }
    else
    {
        if (material != NULL)
        {
            geom->setColorBinding(osg::Geometry::BIND_OFF);
        }
        else
        {
            osg::Vec4Array *colArr = new osg::Vec4Array();
            colArr->push_back(coviseGeometryDefaultColor);

            geom->setColorArray(colArr);
            geom->setColorBinding(osg::Geometry::BIND_OVERALL);
        }
    }

    if (no_of_normals)
    {
        switch (normalbinding)
        {
        case Bind::PerVertex:
        {
            //fprintf(stderr,"COVER INFO: colorbinding per vertex\n");

            osg::Vec3Array *normalArray = new osg::Vec3Array();

            for (int i = 0; i < no_of_polygons; i++)
            {
                if (i == no_of_polygons - 1)
                    numv = no_of_vertices - i_l[i];
                else
                    numv = i_l[i + 1] - i_l[i];
                for (int n = 0; n < numv; n++)
                {
                    int v = v_l[i_l[i] + n];
                    osg::Vec3 norm = osg::Vec3(nx[v], ny[v], nz[v]);
                    norm.normalize();
                    normalArray->push_back(norm);
                }
            }
            geom->setNormalArray(normalArray);
            geom->setNormalBinding(osg::Geometry::BIND_PER_VERTEX);
        }
        break;

        case Bind::OverAll:
        {
            osg::Vec3Array *normalArray = new osg::Vec3Array();

            osg::Vec3 n = osg::Vec3(nx[0], ny[0], nz[0]);
            n.normalize();
            normalArray->push_back(n);
            geom->setNormalArray(normalArray);
            geom->setNormalBinding(osg::Geometry::BIND_OVERALL);
        }
        break;

        case Bind::PerFace:
        {
            osg::Vec3Array *normalArray = new osg::Vec3Array();
            for (int i = 0; i < no_of_polygons; i++)
            {
                osg::Vec3 n = osg::Vec3(nx[i], ny[i], nz[i]);
                n.normalize();

                if (i == no_of_polygons - 1)
                    numv = no_of_vertices - i_l[i];
                else
                    numv = i_l[i + 1] - i_l[i];
                for (int j = 0; j < numv; ++j)
                {
                    normalArray->push_back(n);
                }
            }
            geom->setNormalArray(normalArray);
            geom->setNormalBinding(osg::Geometry::BIND_PER_VERTEX);
        }
        break;
        }
    }
    else
    {
        osg::Vec3Array *normalArray = new osg::Vec3Array();

        // create one normal per polygon and use it for all vertices
        for (int i = 0; i < no_of_polygons; i++)
        {
            if (i == no_of_polygons - 1)
                numv = no_of_vertices - i_l[i];
            else
                numv = i_l[i + 1] - i_l[i];

            int v = v_l[i_l[i] + 0];
            osg::Vec3 p0 = osg::Vec3(x_c[v], y_c[v], z_c[v]);
            v = v_l[i_l[i] + 1];
            osg::Vec3 p1 = osg::Vec3(x_c[v], y_c[v], z_c[v]);
            osg::Vec3 v1 = p1 - p0;
            int vert = 2;
            while (v1.length2() < 1E-16 && vert < numv - 1)
            {
                v = v_l[i_l[i] + vert];
                p1 = osg::Vec3(x_c[v], y_c[v], z_c[v]);
                v1 = p1 - p0;
                vert++;
            }
            v = v_l[i_l[i] + vert];
            osg::Vec3 p2 = osg::Vec3(x_c[v], y_c[v], z_c[v]);
            osg::Vec3 v2 = p2 - p0;
            while (v2.length2() < 1E-16 && vert < numv)
            {
                v = v_l[i_l[i] + vert];
                p2 = osg::Vec3(x_c[v], y_c[v], z_c[v]);
                v2 = p2 - p0;
                vert++;
            }
            v1.normalize();
            v2.normalize();
            osg::Vec3 vn = v1 ^ v2;
            while (vn.length2() < 1E-2 && vert < numv)
            {
                v = v_l[i_l[i] + vert];
                p2 = osg::Vec3(x_c[v], y_c[v], z_c[v]);
                v2 = p2 - p0;
                vert++;
                if (v2.length2() < 1E-16 && vert < numv)
                {
                    continue;
                }
                v2.normalize();
                vn = v1 ^ v2;
            }
            vn.normalize();
            for (int n = 0; n < numv; n++)
            {
                normalArray->push_back(vn);
            }
        }

        geom->setNormalArray(normalArray);
        geom->setNormalBinding(osg::Geometry::BIND_PER_VERTEX);
    }

    if (no_of_texCoords)
    {
        osg::Vec2Array *tcArray = new osg::Vec2Array();

        for (int i = 0; i < no_of_polygons; i++)
        {
            if (i == no_of_polygons - 1)
                numv = no_of_vertices - i_l[i];
            else
                numv = i_l[i + 1] - i_l[i];
            for (int n = 0; n < numv; n++)
            {
                int v = v_l[i_l[i] + n];
                tcArray->push_back(osg::Vec2(tx[v], ty[v]));
            }
        }
        geom->setTexCoordArray(0, tcArray);
    }

    osg::StateSet *geoState = geode->getOrCreateStateSet();

    if (no_of_vertexAttributes > 0)
    {
        osg::Vec3Array *vertArray = new osg::Vec3Array;
        for (int i = 0; i < no_of_polygons; i++)
        {
            if (i == no_of_polygons - 1)
                numv = no_of_vertices - i_l[i];
            else
                numv = i_l[i + 1] - i_l[i];
            for (int n = 0; n < numv; n++)
            {
                int v = v_l[i_l[i] + n];
                vertArray->push_back(osg::Vec3(vax[v], vay[v], vaz[v]));
            }
        }
        geom->setVertexAttribArray(6, vertArray);
    }
    // geoState->setGlobalDefaults();

    if (pixelSize == 4)
        transparent = true;

    setDefaultMaterial(geoState, transparent, material);

    if (backfaceCulling || cullBackfaces) // backfaceCulling nur dann, wenn es im CoviseConfig enabled ist
    {
        osg::CullFace *cullFace = new osg::CullFace();
        cullFace->setMode(osg::CullFace::BACK);
        geoState->setAttributeAndModes(cullFace, osg::StateAttribute::ON);
    }

    osg::PolygonOffset *po = new osg::PolygonOffset(1., 1.);
    geoState->setAttributeAndModes(po, osg::StateAttribute::ON);

    osg::BlendFunc *blendFunc = new osg::BlendFunc();
    blendFunc->setFunction(osg::BlendFunc::SRC_ALPHA, osg::BlendFunc::ONE_MINUS_SRC_ALPHA);
    geoState->setAttributeAndModes(blendFunc, osg::StateAttribute::ON);
    osg::AlphaFunc *alphaFunc = new osg::AlphaFunc();
    alphaFunc->setFunction(osg::AlphaFunc::ALWAYS, 1.0);
    //blendFunc->setFunction(osg::BlendFunc::SRC_ALPHA, osg::BlendFunc::ONE_MINUS_SRC_ALPHA);
    geoState->setAttributeAndModes(alphaFunc, osg::StateAttribute::OFF);

    setTexture(image, pixelSize, texWidth, texHeight, geoState, wm, minfm, magfm);

    geode->setStateSet(geoState);

    if (genStrips)
    {
        d_stripper->stripify(*geom);
    }

    geode->addDrawable(geom);

    return ((osg::Node *)geode);
}

//--------------------------------------------------------------------------------------
// create a new geode contraining triangles
//--------------------------------------------------------------------------------------

osg::Node *
GeometryManager::addTriangles(const char *object_name,
                              int no_of_vertices, int no_of_coords,
                              float *x_c, float *y_c, float *z_c,
                              int *v_l,
                              int no_of_colors, int colorbinding, int colorpacking,
                              float *r, float *g, float *b, int *pc,
                              int no_of_normals, int normalbinding,
                              float *nx, float *ny, float *nz,
                              float &transparency, int, coMaterial *material, int texWidth, int texHeight, int pixelSize, unsigned char *image,
                              int no_of_texCoords, float *tx, float *ty, osg::Texture::WrapMode wm, osg::Texture::FilterMode minfm, osg::Texture::FilterMode magfm,
                              int no_of_vertexAttributes,
                              float *vax, float *vay, float *vaz, bool cullBackfaces)
{
    int no_of_triangles = no_of_vertices / 3;
    if ((no_of_triangles == 0) || (no_of_coords == 0) || (no_of_vertices == 0))
    {
        osg::Group *g = new osg::Group(); // add a dummy object so that we don`t have missing timesteps if object is empty
        g->setName(object_name);
        return g;
    }

    osg::Geode *geode = new osg::Geode();
    geode->setName(object_name);
    osg::Geometry *geom = new osg::Geometry();
    geom->setUseDisplayList(coVRConfig::instance()->useDisplayLists());
    geom->setUseVertexBufferObjects(coVRConfig::instance()->useVBOs());

    // set up geometry
    osg::Vec3Array *vert = new osg::Vec3Array;
    osg::DrawArrays *primitives = new osg::DrawArrays(osg::PrimitiveSet::TRIANGLES, 0, no_of_vertices);
    int vn = 0;
    for (int i = 0; i < no_of_triangles; i++)
    {
        for (int n = 0; n < 3; n++)
        {
            int v = v_l[vn];
            vert->push_back(osg::Vec3(x_c[v], y_c[v], z_c[v]));
            vn++;
        }
    }
    geom->setVertexArray(vert);
    geom->addPrimitiveSet(primitives);

    // associate colors
    bool transparent = false;
    if (no_of_colors && material == NULL && image == NULL) // material should overwrite object colors, so ignore them if a material is present
    {

        switch (colorbinding)
        {
        case Bind::PerVertex:
        {
            //fprintf(stderr,"COVER INFO: colorbinding per vertex\n");

            osg::Vec4Array *colArr = new osg::Vec4Array();
            vn = 0;
            for (int i = 0; i < no_of_triangles; i++)
            {
                for (int n = 0; n < 3; n++)
                {
                    int v = v_l[vn];
                    vn++;
                    if (colorpacking == Pack::RGBA)
                    {
                        float r, g, b, a;
                        unpackRGBA(pc, v, &r, &g, &b, &a);
                        if (a < 1.0)
                        {
                            transparent = true;
                        }
                        colArr->push_back(osg::Vec4(r, g, b, a));
                    }
                    else
                    {
                        if (transparency > 0.f)
                            transparent = true;
                        colArr->push_back(osg::Vec4(r[v], g ? g[v] : r[v], b ? b[v] : r[v], 1.0f - transparency));
                    }
                }
            }
            geom->setColorArray(colArr);
            geom->setColorBinding(osg::Geometry::BIND_PER_VERTEX);
        }
        break;

        case Bind::OverAll:
        {
            if (colorpacking == Pack::RGBA)
            {
                osg::Vec4Array *colArr = new osg::Vec4Array();
                float r, g, b, a;
                unpackRGBA(pc, 0, &r, &g, &b, &a);
                if (a < 1.0)
                {
                    transparent = true;
                }
                colArr->push_back(osg::Vec4(r, g, b, a));

                geom->setColorArray(colArr);
            }
            else
            {
                osg::Vec4Array *colArr = new osg::Vec4Array();
                if (transparency > 0.f)
                    transparent = true;
                colArr->push_back(osg::Vec4(r[0], g ? g[0] : r[0], b ? b[0] : r[0], 1.0f - transparency));
                geom->setColorArray(colArr);
            }
            geom->setColorBinding(osg::Geometry::BIND_OVERALL);
        }
        break;

        case Bind::PerFace:
        {
            osg::Vec4Array *colArr = new osg::Vec4Array();
            if (colorpacking == Pack::RGBA)
            {
                for (int i = 0; i < no_of_triangles; i++)
                {
                    float r, g, b, a;
                    unpackRGBA(pc, i, &r, &g, &b, &a);
                    if (a < 1.0)
                    {
                        transparent = true;
                    }
                    for (int j = 0; j < 3; ++j)
                        colArr->push_back(osg::Vec4(r, g, b, a));
                }
            }
            else
            {
                for (int i = 0; i < no_of_triangles; i++)
                {
                    if (transparency > 0.f)
                        transparent = true;

                    for (int j = 0; j < 3; ++j)
                        colArr->push_back(osg::Vec4(r[i], g ? g[i] : r[i], b ? b[i] : r[i], 1.0f - transparency));
                }
            }
            geom->setColorArray(colArr);
            geom->setColorBinding(osg::Geometry::BIND_PER_VERTEX);
        }
        break;
        }
    }
    else
    {
        if (material != NULL)
        {
            geom->setColorBinding(osg::Geometry::BIND_OFF);
        }
        else
        {
            osg::Vec4Array *colArr = new osg::Vec4Array();
            colArr->push_back(coviseGeometryDefaultColor);

            geom->setColorArray(colArr);
            geom->setColorBinding(osg::Geometry::BIND_OVERALL);
        }
    }

    if (no_of_normals)
    {
        switch (normalbinding)
        {
        case Bind::PerVertex:
        {
            //fprintf(stderr,"COVER INFO: colorbinding per vertex\n");

            osg::Vec3Array *normalArray = new osg::Vec3Array();
            vn = 0;
            for (int i = 0; i < no_of_triangles; i++)
            {
                for (int n = 0; n < 3; n++)
                {
                    int v = v_l[vn];
                    vn++;
                    osg::Vec3 norm = osg::Vec3(nx[v], ny[v], nz[v]);
                    norm.normalize();
                    normalArray->push_back(norm);
                }
            }
            geom->setNormalArray(normalArray);
            geom->setNormalBinding(osg::Geometry::BIND_PER_VERTEX);
        }
        break;

        case Bind::OverAll:
        {
            osg::Vec3Array *normalArray = new osg::Vec3Array();

            osg::Vec3 n = osg::Vec3(nx[0], ny[0], nz[0]);
            n.normalize();
            normalArray->push_back(n);
            geom->setNormalArray(normalArray);
            geom->setNormalBinding(osg::Geometry::BIND_OVERALL);
        }
        break;

        case Bind::PerFace:
        {
            osg::Vec3Array *normalArray = new osg::Vec3Array();
            for (int i = 0; i < no_of_triangles; i++)
            {
                osg::Vec3 n = osg::Vec3(nx[i], ny[i], nz[i]);
                n.normalize();

                for (int j = 0; j < 3; ++j)
                    normalArray->push_back(n);
            }
            geom->setNormalArray(normalArray);
            geom->setNormalBinding(osg::Geometry::BIND_PER_VERTEX);
        }
        break;
        }
    }
    else
    {
        osg::Vec3Array *normalArray = new osg::Vec3Array();
        vn = 0;
        for (int i = 0; i < no_of_triangles; i++)
        {
            int v = v_l[vn];
            vn++;
            osg::Vec3 p0 = osg::Vec3(x_c[v], y_c[v], z_c[v]);
            v = v_l[vn];
            vn++;
            osg::Vec3 p1 = osg::Vec3(x_c[v], y_c[v], z_c[v]);
            v = v_l[vn];
            vn++;
            osg::Vec3 p2 = osg::Vec3(x_c[v], y_c[v], z_c[v]);
            osg::Vec3 v1 = p1 - p0;
            osg::Vec3 v2 = p2 - p1;
            osg::Vec3 vn = v1 ^ v2;
            vn.normalize();
            for (int j = 0; j < 3; ++j)
                normalArray->push_back(vn);
        }
        geom->setNormalArray(normalArray);
        geom->setNormalBinding(osg::Geometry::BIND_PER_VERTEX);
    }

    if (no_of_texCoords)
    {
        osg::Vec2Array *tcArray = new osg::Vec2Array();
        vn = 0;
        for (int i = 0; i < no_of_triangles; i++)
        {
            for (int n = 0; n < 3; n++)
            {
                int v = v_l[vn];
                vn++;
                tcArray->push_back(osg::Vec2(tx[v], ty[v]));
            }
        }
        geom->setTexCoordArray(0, tcArray);
    }

    osg::StateSet *geoState = geode->getOrCreateStateSet();

    if (no_of_vertexAttributes > 0)
    {
        osg::Vec3Array *vertArray = new osg::Vec3Array;
        vn = 0;
        for (int i = 0; i < no_of_triangles; i++)
        {
            for (int n = 0; n < 3; n++)
            {
                int v = v_l[vn];
                vn++;
                vertArray->push_back(osg::Vec3(vax[v], vay[v], vaz[v]));
            }
        }
        geom->setVertexAttribArray(6, vertArray);
    }
    // geoState->setGlobalDefaults();
    if (pixelSize == 4)
        transparent = true;
    setDefaultMaterial(geoState, transparent, material);

    if (backfaceCulling || cullBackfaces) // backfaceCulling nur dann, wenn es im CoviseConfig enabled ist
    {
        osg::CullFace *cullFace = new osg::CullFace();
        cullFace->setMode(osg::CullFace::BACK);
        geoState->setAttributeAndModes(cullFace, osg::StateAttribute::ON);
    }

    osg::PolygonOffset *po = new osg::PolygonOffset(1., 1.);
    geoState->setAttributeAndModes(po, osg::StateAttribute::ON);

    osg::BlendFunc *blendFunc = new osg::BlendFunc();
    blendFunc->setFunction(osg::BlendFunc::SRC_ALPHA, osg::BlendFunc::ONE_MINUS_SRC_ALPHA);
    geoState->setAttributeAndModes(blendFunc, osg::StateAttribute::ON);
    osg::AlphaFunc *alphaFunc = new osg::AlphaFunc();
    alphaFunc->setFunction(osg::AlphaFunc::ALWAYS, 1.0);
    //blendFunc->setFunction(osg::BlendFunc::SRC_ALPHA, osg::BlendFunc::ONE_MINUS_SRC_ALPHA);
    geoState->setAttributeAndModes(alphaFunc, osg::StateAttribute::OFF);

    setTexture(image, pixelSize, texWidth, texHeight, geoState, wm, minfm, magfm);

    geode->setStateSet(geoState);

    if (genStrips)
    {
        d_stripper->stripify(*geom);
    }

    geode->addDrawable(geom);

    return ((osg::Node *)geode);
}

//--------------------------------------------------------------------------------------
// create a new geode contraining quads
//--------------------------------------------------------------------------------------

osg::Node *
GeometryManager::addQuads(const char *object_name,
                          int no_of_vertices, int no_of_coords,
                          float *x_c, float *y_c, float *z_c,
                          int *v_l,
                          int no_of_colors, int colorbinding, int colorpacking,
                          float *r, float *g, float *b, int *pc,
                          int no_of_normals, int normalbinding,
                          float *nx, float *ny, float *nz,
                          float &transparency, int, coMaterial *material, int texWidth, int texHeight, int pixelSize, unsigned char *image,
                          int no_of_texCoords, float *tx, float *ty, osg::Texture::WrapMode wm, osg::Texture::FilterMode minfm, osg::Texture::FilterMode magfm,
                          int no_of_vertexAttributes,
                          float *vax, float *vay, float *vaz, bool cullBackfaces)
{
    int no_of_quads = no_of_vertices / 4;
    if ((no_of_quads == 0) || (no_of_coords == 0) || (no_of_vertices == 0))
    {
        osg::Group *g = new osg::Group(); // add a dummy object so that we don`t have missing timesteps if object is empty
        g->setName(object_name);
        return g;
    }

    osg::Geode *geode = new osg::Geode();
    geode->setName(object_name);
    osg::Geometry *geom = new osg::Geometry();
    geom->setUseDisplayList(coVRConfig::instance()->useDisplayLists());
    geom->setUseVertexBufferObjects(coVRConfig::instance()->useVBOs());

    // set up geometry
    osg::Vec3Array *vert = new osg::Vec3Array;
    osg::DrawArrays *primitives = new osg::DrawArrays(osg::PrimitiveSet::QUADS, 0, no_of_vertices);
    int vn = 0;
    for (int i = 0; i < no_of_quads; i++)
    {
        for (int n = 0; n < 4; n++)
        {
            int v = v_l[vn];
            vert->push_back(osg::Vec3(x_c[v], y_c[v], z_c[v]));
            vn++;
        }
    }
    geom->setVertexArray(vert);
    geom->addPrimitiveSet(primitives);

    // associate colors
    bool transparent = false;
    if (no_of_colors && material == NULL && image == NULL) // material should overwrite object colors, so ignore them if a material is present
    {

        switch (colorbinding)
        {
        case Bind::PerVertex:
        {
            //fprintf(stderr,"COVER INFO: colorbinding per vertex\n");

            osg::Vec4Array *colArr = new osg::Vec4Array();
            vn = 0;
            for (int i = 0; i < no_of_quads; i++)
            {
                for (int n = 0; n < 4; n++)
                {
                    int v = v_l[vn];
                    vn++;
                    if (colorpacking == Pack::RGBA)
                    {
                        float r, g, b, a;
                        unpackRGBA(pc, v, &r, &g, &b, &a);
                        if (a < 1.0)
                        {
                            transparent = true;
                        }
                        colArr->push_back(osg::Vec4(r, g, b, a));
                    }
                    else
                    {
                        if (transparency > 0.f)
                            transparent = true;
                        colArr->push_back(osg::Vec4(r[v], g ? g[v] : r[v], b ? b[v] : r[v], 1.0f - transparency));
                    }
                }
            }
            geom->setColorArray(colArr);
            geom->setColorBinding(osg::Geometry::BIND_PER_VERTEX);
        }
        break;

        case Bind::OverAll:
        {
            //fprintf(stderr,"COVER INFO: colorbinding over all\n");

            if (colorpacking == Pack::RGBA)
            {
                osg::Vec4Array *colArr = new osg::Vec4Array();
                float r, g, b, a;
                unpackRGBA(pc, 0, &r, &g, &b, &a);
                if (a < 1.0)
                {
                    transparent = true;
                }
                colArr->push_back(osg::Vec4(r, g, b, a));

                geom->setColorArray(colArr);
            }
            else
            {
                osg::Vec4Array *colArr = new osg::Vec4Array();
                if (transparency > 0.f)
                    transparent = true;
                colArr->push_back(osg::Vec4(r[0], g ? g[0] : r[0], b ? b[0] : r[0], 1.0f - transparency));
                geom->setColorArray(colArr);
            }
            geom->setColorBinding(osg::Geometry::BIND_OVERALL);
        }
        break;

        case Bind::PerFace:
        {
            //fprintf(stderr,"COVER INFO: colorbinding per face\n");

            osg::Vec4Array *colArr = new osg::Vec4Array();
            if (colorpacking == Pack::RGBA)
            {
                for (int i = 0; i < no_of_quads; i++)
                {
                    float r, g, b, a;
                    unpackRGBA(pc, i, &r, &g, &b, &a);
                    if (a < 1.0)
                    {
                        transparent = true;
                    }
                    for (int j = 0; j < 4; ++j)
                        colArr->push_back(osg::Vec4(r, g, b, a));
                }
            }
            else
            {
                for (int i = 0; i < no_of_quads; i++)
                {
                    if (transparency > 0.f)
                        transparent = true;

                    for (int j = 0; j < 4; ++j)
                        colArr->push_back(osg::Vec4(r[i], g ? g[i] : r[i], b ? b[i] : r[i], 1.0f - transparency));
                }
            }
            geom->setColorArray(colArr);
            geom->setColorBinding(osg::Geometry::BIND_PER_VERTEX);
        }
        break;
        }
    }
    else
    {
        if (material != NULL)
        {
            geom->setColorBinding(osg::Geometry::BIND_OFF);
        }
        else
        {
            osg::Vec4Array *colArr = new osg::Vec4Array();
            colArr->push_back(coviseGeometryDefaultColor);

            geom->setColorArray(colArr);
            geom->setColorBinding(osg::Geometry::BIND_OVERALL);
        }
    }

    if (no_of_normals)
    {
        switch (normalbinding)
        {
        case Bind::PerVertex:
        {
            //fprintf(stderr,"COVER INFO: colorbinding per vertex\n");

            osg::Vec3Array *normalArray = new osg::Vec3Array();
            vn = 0;
            for (int i = 0; i < no_of_quads; i++)
            {
                for (int n = 0; n < 4; n++)
                {
                    int v = v_l[vn];
                    vn++;
                    osg::Vec3 norm = osg::Vec3(nx[v], ny[v], nz[v]);
                    norm.normalize();
                    normalArray->push_back(norm);
                }
            }
            geom->setNormalArray(normalArray);
            geom->setNormalBinding(osg::Geometry::BIND_PER_VERTEX);
        }
        break;

        case Bind::OverAll:
        {
            osg::Vec3Array *normalArray = new osg::Vec3Array();

            osg::Vec3 n = osg::Vec3(nx[0], ny[0], nz[0]);
            n.normalize();
            normalArray->push_back(n);
            geom->setNormalArray(normalArray);
            geom->setNormalBinding(osg::Geometry::BIND_OVERALL);
        }
        break;

        case Bind::PerFace:
        {
            osg::Vec3Array *normalArray = new osg::Vec3Array();
            for (int i = 0; i < no_of_quads; i++)
            {
                osg::Vec3 n = osg::Vec3(nx[i], ny[i], nz[i]);
                n.normalize();

                for (int j = 0; j < 4; ++j)
                    normalArray->push_back(n);
            }
            geom->setNormalArray(normalArray);
            geom->setNormalBinding(osg::Geometry::BIND_PER_VERTEX);
        }
        break;
        }
    }
    else
    {
        osg::Vec3Array *normalArray = new osg::Vec3Array();
        vn = 0;
        // create one normal per polygon and use it for all vertices
        for (int i = 0; i < no_of_quads; i++)
        {
            int v = v_l[vn];
            vn++;
            osg::Vec3 p0 = osg::Vec3(x_c[v], y_c[v], z_c[v]);
            v = v_l[vn];
            vn++;
            osg::Vec3 p1 = osg::Vec3(x_c[v], y_c[v], z_c[v]);
            v = v_l[vn];
            vn++;
            osg::Vec3 p2 = osg::Vec3(x_c[v], y_c[v], z_c[v]);
            osg::Vec3 v1 = p1 - p0;
            osg::Vec3 v2 = p2 - p1;
            osg::Vec3 vn = v1 ^ v2;
            vn.normalize();
            for (int j = 0; j < 4; ++j)
                normalArray->push_back(vn);
        }

        geom->setNormalArray(normalArray);
        geom->setNormalBinding(osg::Geometry::BIND_PER_VERTEX);
    }

    if (no_of_texCoords)
    {
        osg::Vec2Array *tcArray = new osg::Vec2Array();
        vn = 0;
        for (int i = 0; i < no_of_quads; i++)
        {
            for (int n = 0; n < 4; n++)
            {
                int v = v_l[vn];
                vn++;
                tcArray->push_back(osg::Vec2(tx[v], ty[v]));
            }
        }
        geom->setTexCoordArray(0, tcArray);
    }

    osg::StateSet *geoState = geode->getOrCreateStateSet();

    if (no_of_vertexAttributes > 0)
    {
        osg::Vec3Array *vertArray = new osg::Vec3Array;
        vn = 0;
        for (int i = 0; i < no_of_quads; i++)
        {
            for (int n = 0; n < 4; n++)
            {
                int v = v_l[vn];
                vn++;
                vertArray->push_back(osg::Vec3(vax[v], vay[v], vaz[v]));
            }
        }
        geom->setVertexAttribArray(6, vertArray);
    }
    // geoState->setGlobalDefaults();
    if (pixelSize == 4)
        transparent = true;
    setDefaultMaterial(geoState, transparent, material, false);

    if (backfaceCulling || cullBackfaces) // backfaceCulling nur dann, wenn es im CoviseConfig enabled ist
    {
        osg::CullFace *cullFace = new osg::CullFace();
        cullFace->setMode(osg::CullFace::BACK);
        geoState->setAttributeAndModes(cullFace, osg::StateAttribute::ON);
    }

    osg::PolygonOffset *po = new osg::PolygonOffset(1., 1.);
    geoState->setAttributeAndModes(po, osg::StateAttribute::ON);

    osg::BlendFunc *blendFunc = new osg::BlendFunc();
    blendFunc->setFunction(osg::BlendFunc::SRC_ALPHA, osg::BlendFunc::ONE_MINUS_SRC_ALPHA);
    geoState->setAttributeAndModes(blendFunc, osg::StateAttribute::ON);
    osg::AlphaFunc *alphaFunc = new osg::AlphaFunc();
    alphaFunc->setFunction(osg::AlphaFunc::ALWAYS, 1.0);
    //blendFunc->setFunction(osg::BlendFunc::SRC_ALPHA, osg::BlendFunc::ONE_MINUS_SRC_ALPHA);
    geoState->setAttributeAndModes(alphaFunc, osg::StateAttribute::OFF);

    setTexture(image, pixelSize, texWidth, texHeight, geoState, wm, minfm, magfm);

    geode->setStateSet(geoState);

    if (genStrips)
    {
        d_stripper->stripify(*geom);
    }

    geode->addDrawable(geom);

    return ((osg::Node *)geode);
}

void
GeometryManager::setTexture(const unsigned char *image, int pixelSize, int texWidth, int texHeight, osg::StateSet *geoState, osg::Texture::WrapMode wm, osg::Texture::FilterMode minfm, osg::Texture::FilterMode magfm)
{
    if (image)
    {
        osg::Texture2D *tex = new osg::Texture2D;
        osg::Image *texImage = new osg::Image;
        tex->setFilter(osg::Texture2D::MIN_FILTER, minfm);
        tex->setFilter(osg::Texture2D::MAG_FILTER, magfm);
        tex->setWrap(osg::Texture::WRAP_S, wm);
        tex->setWrap(osg::Texture::WRAP_T, wm);

        GLint internalFormat = pixelSize;
        switch (pixelSize)
        {
        case 3:
            internalFormat = GL_RGB8;
            break;
        case 4:
            internalFormat = GL_RGBA8;
            break;
        }
        GLint format = GL_LUMINANCE;
        switch (pixelSize)
        {
        case 1:
            format = GL_LUMINANCE;
            break;
        case 2:
            format = GL_LUMINANCE_ALPHA;
            break;
        case 3:
            format = GL_RGB;
            break;
        case 4:
            format = GL_RGBA;
            break;
        }

        unsigned char *it = NULL;
        if (texWidth == 255 && texHeight == 1)
        {
            // special case: osg::Texture map from ColorEdit module
            // scaling to texWidth+1=256 (->power of 2)
            it = new unsigned char[256 * pixelSize];
            for (int i = 0; i < 256; i++)
            {
                for (int j = 0; j < pixelSize; j++)
                {
                    switch (i)
                    {
                    case 0:
                        it[j] = image[j];
                        break;
                    case 255:
                        it[255 * pixelSize + j] = image[254 * pixelSize + j];
                        break;
                    default:
                        // linear interpolation
                        it[i * pixelSize + j] = image[(i - 1) * pixelSize + j] + (256 - i) / 256 * (image[i * pixelSize + j] - image[(i - 1) * pixelSize + j]);
                    }
                }
            }
            texImage->setImage(256, 1, 1, internalFormat, format, GL_UNSIGNED_BYTE, it, osg::Image::USE_NEW_DELETE);
        }
        else
        {
            it = new unsigned char[texWidth * texHeight * pixelSize];
            memcpy(it, image, texWidth * texHeight * pixelSize);
            /*   for(int i=0;i<texWidth*texHeight;i++)
         {
         for (int j=0; j<pixelSize; j++)
         it[i*pixelSize+j]=image[i*pixelSize+j];
         }*/
            texImage->setImage(texWidth, texHeight, 1, internalFormat, format, GL_UNSIGNED_BYTE, it, osg::Image::USE_NEW_DELETE);
        }

        tex->setImage(texImage);
        geoState->setTextureAttributeAndModes(0, tex, osg::StateAttribute::ON);
        osg::TexEnv *texEnv = new osg::TexEnv;
        texEnv->setMode(osg::TexEnv::MODULATE);
        geoState->setTextureAttributeAndModes(0, texEnv, osg::StateAttribute::ON);
    }
}

//--------------------------------------------------------------------------------------
// create a new geode contraining tristrips and add it to the scene
//--------------------------------------------------------------------------------------

osg::Node *GeometryManager::addTriangleStrip(const char *object_name,
                                             int no_of_strips, int no_of_vertices,
                                             int no_of_coords, float *x_c, float *y_c, float *z_c,
                                             int *v_l, int *i_l,
                                             int no_of_colors, int colorbinding, int colorpacking,
                                             float *r, float *g, float *b, int *pc,
                                             int no_of_normals, int normalbinding,
                                             float *nx, float *ny, float *nz, float &transparency,
                                             int /*vertexOrder*/, coMaterial *material, int texWidth, int texHeight, int pixelSize, unsigned char *image,
                                             int no_of_texCoords, float *tx, float *ty, osg::Texture::WrapMode wm, osg::Texture::FilterMode minfm, osg::Texture::FilterMode magfm,
                                             int no_of_vertexAttributes,
                                             float *vax, float *vay, float *vaz, bool cullBackfaces)
{
    if ((no_of_strips == 0) || (no_of_coords == 0) || (no_of_vertices == 0))
    {
        osg::Group *g = new osg::Group(); // add a dummy object so that we don`t have missing timesteps if object is empty
        g->setName(object_name);
        return g;
    }

    osg::Geode *geode = new osg::Geode();
    geode->setName(object_name);
    osg::Geometry *geom = new osg::Geometry();
    geom->setUseDisplayList(coVRConfig::instance()->useDisplayLists());
    geom->setUseVertexBufferObjects(coVRConfig::instance()->useVBOs());

    // set up geometry
    osg::Vec3Array *vert = new osg::Vec3Array;
    osg::DrawArrayLengths *primitives = new osg::DrawArrayLengths(osg::PrimitiveSet::TRIANGLE_STRIP);
    for (int i = 0; i < no_of_strips; i++)
    {
        int numv;
        if (i == no_of_strips - 1)
            numv = no_of_vertices - i_l[i];
        else
            numv = i_l[i + 1] - i_l[i];
        primitives->push_back(numv);
        for (int n = 0; n < numv; n++)
        {
            int v = v_l[i_l[i] + n];
            vert->push_back(osg::Vec3(x_c[v], y_c[v], z_c[v]));
        }
    }
    geom->setVertexArray(vert);
    geom->addPrimitiveSet(primitives);

    // associate colors
    bool transparent = false;
    if (no_of_colors && material == NULL && image == NULL) // material should overwrite object colors, so ignore them if a material is present
    {

        switch (colorbinding)
        {
        case Bind::PerVertex:
        {
            //fprintf(stderr,"COVER INFO: colorbinding per vertex\n");

            osg::Vec4Array *colArr = new osg::Vec4Array();

            for (int i = 0; i < no_of_strips; i++)
            {
                int numv;
                if (i == no_of_strips - 1)
                    numv = no_of_vertices - i_l[i];
                else
                    numv = i_l[i + 1] - i_l[i];
                for (int n = 0; n < numv; n++)
                {
                    int v = v_l[i_l[i] + n];
                    if (colorpacking == Pack::RGBA)
                    {
                        float r, g, b, a;
                        unpackRGBA(pc, v, &r, &g, &b, &a);
                        if (a < 1.0)
                        {
                            transparent = true;
                        }
                        colArr->push_back(osg::Vec4(r, g, b, a));
                    }
                    else
                    {
                        if (transparency > 0.f)
                            transparent = true;
                        colArr->push_back(osg::Vec4(r[v], g ? g[v] : r[v], b ? b[v] : r[v], 1.0f - transparency));
                    }
                }
            }
            geom->setColorArray(colArr);
            geom->setColorBinding(osg::Geometry::BIND_PER_VERTEX);
        }
        break;

        case Bind::OverAll:
        {
            osg::Vec4Array *colArr = new osg::Vec4Array();

            if (colorpacking == Pack::RGBA)
            {
                float r, g, b, a;
                unpackRGBA(pc, 0, &r, &g, &b, &a);
                if (a < 1.0)
                    transparent = true;
                colArr->push_back(osg::Vec4(r, g, b, a));
            }
            else
            {
                if (transparency > 0.f)
                    transparent = true;
                colArr->push_back(osg::Vec4(r[0], g ? g[0] : r[0], b ? b[0] : r[0], 1.0f - transparency));
            }
            geom->setColorArray(colArr);
            geom->setColorBinding(osg::Geometry::BIND_OVERALL);
        }
        break;

        case Bind::PerFace:
        {
            osg::Vec4Array *colArr = new osg::Vec4Array();
            for (int i = 0; i < no_of_strips; i++)
            {
                int numv;
                if (i == no_of_strips - 1)
                    numv = no_of_vertices - i_l[i];
                else
                    numv = i_l[i + 1] - i_l[i];
                for (int n = 0; n < numv; n++)
                {
                    if (colorpacking == Pack::RGBA)
                    {
                        float r, g, b, a;
                        unpackRGBA(pc, i, &r, &g, &b, &a);
                        if (a < 1.0)
                            transparent = true;
                        colArr->push_back(osg::Vec4(r, g, b, a));
                    }
                    else
                    {
                        if (transparency > 0.f)
                            transparent = true;
                        colArr->push_back(osg::Vec4(r[i], g ? g[i] : r[i], b ? b[i] : r[i], 1.0f - transparency));
                    }
                }
            }
            geom->setColorArray(colArr);
            geom->setColorBinding(osg::Geometry::BIND_PER_VERTEX);
        }
        break;
        }
    }
    else
    {
        if (material != NULL)
        {
            geom->setColorBinding(osg::Geometry::BIND_OFF);
        }
        else
        {
            osg::Vec4Array *colArr = new osg::Vec4Array();
            colArr->push_back(coviseGeometryDefaultColor);

            geom->setColorArray(colArr);
            geom->setColorBinding(osg::Geometry::BIND_OVERALL);
        }
    }

    if (no_of_normals)
    {
        switch (normalbinding)
        {
        case Bind::PerVertex:
        {
            //fprintf(stderr,"COVER INFO: colorbinding per vertex\n");

            osg::Vec3Array *normalArray = new osg::Vec3Array();

            for (int i = 0; i < no_of_strips; i++)
            {
                int numv;
                if (i == no_of_strips - 1)
                    numv = no_of_vertices - i_l[i];
                else
                    numv = i_l[i + 1] - i_l[i];
                for (int n = 0; n < numv; n++)
                {
                    int v = v_l[i_l[i] + n];

                    normalArray->push_back(osg::Vec3(nx[v], ny[v], nz[v]));
                }
            }
            geom->setNormalArray(normalArray);
            geom->setNormalBinding(osg::Geometry::BIND_PER_VERTEX);
        }
        break;

        case Bind::OverAll:
        {
            osg::Vec3Array *normalArray = new osg::Vec3Array();

            normalArray->push_back(osg::Vec3(nx[0], ny[0], nz[0]));
            geom->setNormalArray(normalArray);
            geom->setNormalBinding(osg::Geometry::BIND_OVERALL);
        }
        break;

        case Bind::PerFace:
        {
            osg::Vec3Array *normalArray = new osg::Vec3Array();
            for (int i = 0; i < no_of_strips; i++)
            {
                int numv;
                if (i == no_of_strips - 1)
                    numv = no_of_vertices - i_l[i];
                else
                    numv = i_l[i + 1] - i_l[i];
                for (int n = 0; n < numv; n++)
                    normalArray->push_back(osg::Vec3(nx[i], ny[i], nz[i]));
            }
            geom->setNormalArray(normalArray);
            geom->setNormalBinding(osg::Geometry::BIND_PER_VERTEX);
        }
        break;
        }
    }
    else
    {
        osg::Vec3Array *normalArray = new osg::Vec3Array();

        // create one normal per strip and use it for all vertices
        for (int i = 0; i < no_of_strips; i++)
        {
            int v = v_l[i_l[i] + 0];
            osg::Vec3 p0 = osg::Vec3(x_c[v], y_c[v], z_c[v]);
            v = v_l[i_l[i] + 1];
            osg::Vec3 p1 = osg::Vec3(x_c[v], y_c[v], z_c[v]);
            v = v_l[i_l[i] + 2];
            osg::Vec3 p2 = osg::Vec3(x_c[v], y_c[v], z_c[v]);
            osg::Vec3 v1 = p1 - p0;
            osg::Vec3 v2 = p2 - p1;
            osg::Vec3 vn = v1 ^ v2;
            vn.normalize();
            int numv;
            if (i == no_of_strips - 1)
                numv = no_of_vertices - i_l[i];
            else
                numv = i_l[i + 1] - i_l[i];
            for (int n = 0; n < numv; n++)
            {
                normalArray->push_back(vn);
            }
        }
        geom->setNormalArray(normalArray);
        geom->setNormalBinding(osg::Geometry::BIND_PER_VERTEX);
    }

    if (no_of_texCoords)
    {
        osg::Vec2Array *tcArray = new osg::Vec2Array();

        for (int i = 0; i < no_of_strips; i++)
        {
            int numv;
            if (i == no_of_strips - 1)
                numv = no_of_vertices - i_l[i];
            else
                numv = i_l[i + 1] - i_l[i];
            for (int n = 0; n < numv; n++)
            {
                int v = v_l[i_l[i] + n];
                tcArray->push_back(osg::Vec2(tx[v], ty[v]));
            }
        }
        geom->setTexCoordArray(0, tcArray);
    }

    geode->addDrawable(geom);

    osg::StateSet *geoState = geode->getOrCreateStateSet();

    if (no_of_vertexAttributes > 0)
    {
        osg::Vec3Array *vertArray = new osg::Vec3Array;
        for (int i = 0; i < no_of_strips; i++)
        {
            int numv;
            if (i == no_of_strips - 1)
                numv = no_of_vertices - i_l[i];
            else
                numv = i_l[i + 1] - i_l[i];
            for (int n = 0; n < numv; n++)
            {
                int v = v_l[i_l[i] + n];
                vertArray->push_back(osg::Vec3(vax[v], vay[v], vaz[v]));
            }
        }
        geom->setVertexAttribArray(6, vertArray);
    }
    // geoState->setGlobalDefaults();
    if (pixelSize == 4)
        transparent = true;
    setDefaultMaterial(geoState, transparent, material);

    osg::PolygonOffset *po = new osg::PolygonOffset(1., 1.);
    geoState->setAttributeAndModes(po, osg::StateAttribute::ON);

    if (backfaceCulling || cullBackfaces) // backfaceCulling nur dann, wenn es im CoviseConfig enabled ist
    {
        osg::CullFace *cullFace = new osg::CullFace();
        cullFace->setMode(osg::CullFace::BACK);
        geoState->setAttributeAndModes(cullFace, osg::StateAttribute::ON);
    }
    osg::BlendFunc *blendFunc = new osg::BlendFunc();
    blendFunc->setFunction(osg::BlendFunc::SRC_ALPHA, osg::BlendFunc::ONE_MINUS_SRC_ALPHA);
    geoState->setAttributeAndModes(blendFunc, osg::StateAttribute::ON);
    osg::AlphaFunc *alphaFunc = new osg::AlphaFunc();
    alphaFunc->setFunction(osg::AlphaFunc::ALWAYS, 1.0);
    //blendFunc->setFunction(osg::BlendFunc::SRC_ALPHA, osg::BlendFunc::ONE_MINUS_SRC_ALPHA);
    geoState->setAttributeAndModes(alphaFunc, osg::StateAttribute::OFF);

    setTexture(image, pixelSize, texWidth, texHeight, geoState, wm, minfm, magfm);

    geode->setStateSet(geoState);

    return ((osg::Node *)geode);
}

//--------------------------------------------------------------------------------------
// add a line object
//--------------------------------------------------------------------------------------
//--------------------------------------------------------------------------------------
//in-line GLSL source code for the "shader"
//--------------------------------------------------------------------------------------
//shader for two-sided lighting for lines

static const char *shaderVertSource = {
    "// shader \n"
    "  varying  vec4 ecPosition;\n"
    "  varying  vec3 ecPosition3;\n"
    "  varying  vec3 eye;\n"
    "  varying  vec3 normal;\n"
    "  varying  vec4 color;\n"

    "  uniform  int LightNum;\n"

    "void DirectionalLight ( in int LightNum, in vec3 normal,  inout vec4 ambient, inout vec4 diffuse,inout vec4 specular)\n"
    "{\n"
    "    float L,F,e;\n"
    // computation of cosine of angle between surface normal and light direction
    "    if(dot(normal, normalize(vec3 (gl_LightSource[LightNum].position)))>0.0)\n"
    // light source is in front of the surface
    // L is normal light direction
    "        L = dot(normal, normalize(vec3 (gl_LightSource[LightNum].position)));\n"
    "    else\n"
    "    {\n"
    // light source is behind the surface
    "        normal = -normal;\n"
    "        L = dot(normal, normalize(vec3 (gl_LightSource[LightNum].position)));\n"
    "    }\n"
    // computation of cosine of angle between surface normal and
    // half angle between light direction and viewing direction
    // F is normal light half vector
    "    F = max(0.0,dot(normal, vec3 (gl_LightSource[LightNum].halfVector)));\n"
    "    if (L == 0.0)\n"
    "        e = 0.0;\n"
    "    else\n"
    //power factor
    "        e = pow(F,gl_FrontMaterial.shininess);\n"
    "    ambient  += gl_LightSource[LightNum].ambient;\n"
    "    diffuse  += gl_LightSource[LightNum].diffuse * L;\n"
    "    specular += gl_LightSource[LightNum].specular * e;\n"
    "}\n"

    "void PointLight( in int LightNum, in vec3 eye, in vec3 ecPosition3, in vec3 normal, inout vec4 ambient, inout vec4 diffuse,inout vec4 specular)\n"
    "{\n"
    "    float L,F,e;\n"
    "    float attenuation;\n" //attenuation factor
    "    float d;\n" // distance from surface to light position
    "    vec3 VP;\n" // direction from surface to light position
    "    vec3 halfVector;\n"
    // vector from surface to light position
    "    VP = vec3(gl_LightSource[LightNum].position) - ecPosition3;\n"
    "    d = length(VP);\n"
    "    VP = normalize(VP);\n"
    "    attenuation = 1.0 / (gl_LightSource[LightNum].constantAttenuation +\n"
    "            gl_LightSource[LightNum].linearAttenuation * d + \n"
    "                 gl_LightSource[LightNum].quadraticAttenuation * d * d);\n"
    "    if(dot(normal,VP)>0.0)\n"
    // light source is in front of the surface
    "        L = dot(normal,VP);\n"
    "    else\n"
    "    {\n"
    // light source is behind the surface
    "        normal = -normal;\n"
    "        L = dot(normal,VP);\n"
    "    }\n"
    // direction of maximum highlights
    "    halfVector = normalize(VP +eye);\n"
    "    F = max(0.0,dot(normal,halfVector));\n"
    "    if (L == 0.0)\n"
    "        e = 0.0;\n"
    "    else\n"
    //power factor
    "        e = pow(F, gl_FrontMaterial.shininess);\n"
    "    ambient  += gl_LightSource[LightNum].ambient * attenuation;\n"
    "    diffuse  += gl_LightSource[LightNum].diffuse * L *attenuation;\n"
    "    specular += gl_LightSource[LightNum].specular *e* attenuation;\n"
    "}\n"

    "void SpotLight( in int LightNum, in vec3 eye,in vec3 ecPosition3,in vec3 normal,inout vec4 ambient,inout vec4 diffuse,inout vec4 specular)\n"
    "{\n"
    "    float L,F,e;\n"
    "    float spotDot;\n"
    "    float spotAttenuation;\n"
    "    float attenuation;\n"
    "    float d;\n"
    "    vec3 VP;\n"
    "    vec3 halfVector;\n"
    // vector from surface to light position
    "    VP = vec3(gl_LightSource[LightNum].position)- ecPosition3;\n"
    // distance from surface to light position
    "    d=length(VP);\n"
    "    VP=normalize(VP);\n"
    // attenuation factor
    "    attenuation = 1.0 /  (gl_LightSource[LightNum].constantAttenuation + \n"
    "            gl_LightSource[LightNum].linearAttenuation * d + \n"
    "                 gl_LightSource[LightNum].quadraticAttenuation * d * d);\n"
    // cosine of angle between spotlight
    "    spotDot = dot(-VP,normalize(gl_LightSource[LightNum].spotDirection));\n"
    "    if (spotDot < gl_LightSource[LightNum].spotCosCutoff )\n"
    "        spotAttenuation = 0.0;\n"
    "    else\n"
    "        spotAttenuation = pow(spotDot, gl_LightSource[LightNum].spotExponent);\n"
    "    attenuation *= spotAttenuation;\n"
    "    halfVector = normalize(VP + eye);\n"
    "    if(dot(normal,VP)>0.0)\n"
    // light source is in front of the surface
    "        L = dot(normal,VP);\n"
    "    else\n"
    "    {\n"
    // light source is behind the surface
    "        normal = -normal;\n"
    "        L = dot(normal,VP);\n"
    "    }\n"
    "    F = max(0.0, dot(normal, halfVector));\n"
    "    if (L == 0.0)\n"
    "        e = 0.0;\n"
    "    else\n"
    //power factor
    "        e = pow(F, gl_FrontMaterial.shininess);\n"
    "    ambient  += gl_LightSource[LightNum].ambient * attenuation;\n"
    "    diffuse  += gl_LightSource[LightNum].diffuse * L * attenuation;\n"
    "    specular += gl_LightSource[LightNum].specular * e * attenuation;\n"
    "}\n"

    "void main(void)\n"
    "{\n"
    // transform vertex to clip space
    "    gl_Position = gl_ModelViewProjectionMatrix * gl_Vertex;\n"
    // transform vertex to eye coordinates
    "    ecPosition = gl_ModelViewMatrix * gl_Vertex;  \n"
    "    ecPosition3 = (vec3(ecPosition)) / ecPosition.w;\n"
    "    eye = -normalize(ecPosition3);\n"
    // transformation of normal
    "    normal = gl_NormalMatrix * gl_Normal; \n"
    // normalization of normal
    "    normal = normalize(normal);\n"
    "    vec4 amb = vec4(0.0);\n"
    "    vec4 diff = vec4(0.0);\n"
    "    vec4 spec = vec4(0.0);\n"

    "    if (LightNum>=0)\n"
    "    {\n"
    "           if (gl_LightSource[LightNum].position.w == 0.0)\n"
    "               DirectionalLight(LightNum,normal,amb,diff,spec);\n"
    "           else if (gl_LightSource[LightNum].spotCutoff == 180.0)\n"
    "                    PointLight(LightNum, eye, ecPosition3, normal, amb, diff, spec);\n"
    "                else \n"
    "                    SpotLight(LightNum, eye, ecPosition3, normal, amb, diff, spec);\n"
    "    }\n"

    // surface color computation
    "    color = gl_FrontLightModelProduct.sceneColor + \n"
    "            amb * gl_FrontMaterial.ambient +\n"
    "            diff * gl_FrontMaterial.diffuse+\n"
    "            spec * gl_FrontMaterial.specular;\n"
    "    gl_FrontColor = color;\n"
    " }\n"
};

class ShaderCallback : public osg::Uniform::Callback
{
    virtual void operator()(osg::Uniform *LightNumUniform, osg::NodeVisitor *)
    {
        int lightNum = -1;
        for (int i = 0; i < int(coVRLighting::instance()->lightList.size()); i++)
        {
            if (coVRLighting::instance()->lightList[i].on)
            {
                lightNum = i;
                break;
            }
        }
        LightNumUniform->set(lightNum);
    }
};

osg::Node *GeometryManager::addLine(const char *object_name,
                                    int no_of_lines, // number of linestrip primitives
                                    int no_of_vertices, // length of v_l
                                    int no_of_coords, // length of x_c, y_c, z_c
                                    float *x_c, float *y_c, float *z_c,
                                    int *v_l, // indexlist for coord
                                    int *i_l,
                                    int no_of_colors, int colorbinding, int colorpacking,
                                    float *r, float *g, float *b, int *pc,
                                    int no_of_normals, int normalbinding,
                                    float *nx, float *ny, float *nz, int, coMaterial *material,
                                    int texWidth, int texHeight, int pixelSize, unsigned char *image,
                                    int no_of_texCoords, float *tx, float *ty, osg::Texture::WrapMode wm, osg::Texture::FilterMode minfm, osg::Texture::FilterMode magfm,
                                    float linewidth)

{
    //    if(material)
    //    {
    //       cerr << "addLine: material ignored" << endl;
    //    }

    if ((no_of_lines == 0) || (no_of_coords == 0) || (no_of_vertices == 0))
    {
        osg::Group *g = new osg::Group(); // add a dummy object so that we don`t have missing timesteps if object is empty
        g->setName(object_name);
        return g;
    }

    osg::Geode *geode = new osg::Geode();
    geode->setName(object_name);
    osg::Geometry *geom = new osg::Geometry();
    geom->setUseDisplayList(coVRConfig::instance()->useDisplayLists());
    geom->setUseVertexBufferObjects(coVRConfig::instance()->useVBOs());

    // set up geometry
    osg::Vec3Array *vert = new osg::Vec3Array;
    osg::DrawArrayLengths *primitives = new osg::DrawArrayLengths(osg::PrimitiveSet::LINE_STRIP);
    for (int i = 0; i < no_of_lines; i++)
    {
        int numv;
        if (i == no_of_lines - 1)
            numv = no_of_vertices - i_l[i];
        else
            numv = i_l[i + 1] - i_l[i];
        primitives->push_back(numv);
        for (int n = 0; n < numv; n++)
        {
            int v = v_l[i_l[i] + n];
            vert->push_back(osg::Vec3(x_c[v], y_c[v], z_c[v]));
        }
    }
    geom->setVertexArray(vert);
    geom->addPrimitiveSet(primitives);

    // associate colors
    bool transparent = false;
    if (no_of_colors && material == NULL && image == NULL) // material should overwrite object colors, so ignore them if a material is present
    {

        switch (colorbinding)
        {
        case Bind::PerVertex:
        {
            //fprintf(stderr,"COVER INFO: colorbinding per vertex\n");

            osg::Vec4Array *colArr = new osg::Vec4Array();

            for (int i = 0; i < no_of_lines; i++)
            {
                int numv;
                if (i == no_of_lines - 1)
                    numv = no_of_vertices - i_l[i];
                else
                    numv = i_l[i + 1] - i_l[i];
                for (int n = 0; n < numv; n++)
                {
                    int v = v_l[i_l[i] + n];
                    if (colorpacking == Pack::RGBA)
                    {
                        float r, g, b, a;
                        unpackRGBA(pc, v, &r, &g, &b, &a);
                        if (a < 1.f)
                            transparent = true;
                        colArr->push_back(osg::Vec4(r, g, b, a));
                    }
                    else
                        colArr->push_back(osg::Vec4(r[v], g ? g[v] : r[v], b ? b[v] : r[v], 1.0f));
                }
            }
            geom->setColorArray(colArr);
            geom->setColorBinding(osg::Geometry::BIND_PER_VERTEX);
        }
        break;

        case Bind::OverAll:
        {
            osg::Vec4Array *colArr = new osg::Vec4Array();

            if (colorpacking == Pack::RGBA)
            {
                float r, g, b, a;
                unpackRGBA(pc, 0, &r, &g, &b, &a);
                if (a < 1.f)
                    transparent = true;
                colArr->push_back(osg::Vec4(r, g, b, a));
            }
            else
                colArr->push_back(osg::Vec4(r[0], g[0], b[0], 1.0f));
            geom->setColorArray(colArr);
            geom->setColorBinding(osg::Geometry::BIND_OVERALL);
        }
        break;

        case Bind::PerFace:
        {
            osg::Vec4Array *colArr = new osg::Vec4Array();
            for (int i = 0; i < no_of_lines; i++)
            {
                int numv;
                if (i == no_of_lines - 1)
                    numv = no_of_vertices - i_l[i];
                else
                    numv = i_l[i + 1] - i_l[i];
                if (colorpacking == Pack::RGBA)
                {
                    float r, g, b, a;
                    unpackRGBA(pc, i, &r, &g, &b, &a);
                    if (a < 1.f)
                        transparent = true;
                    for (int n = 0; n < numv; n++)
                        colArr->push_back(osg::Vec4(r, g, b, a));
                }
                else
                {
                    for (int n = 0; n < numv; n++)
                        colArr->push_back(osg::Vec4(r[i], g ? g[i] : r[i], b ? b[i] : r[i], 1.0f));
                }
            }
            geom->setColorArray(colArr);
            geom->setColorBinding(osg::Geometry::BIND_PER_VERTEX);
        }
        break;
        }
    }
    else
    {
        if (material != NULL)
        {
            geom->setColorBinding(osg::Geometry::BIND_OFF);
        }
        else
        {
            osg::Vec4Array *colArr = new osg::Vec4Array();
            colArr->push_back(osg::Vec4(1.0f, 1.0f, 1.0f, 1.0f));
            geom->setColorArray(colArr);
            geom->setColorBinding(osg::Geometry::BIND_OVERALL);
        }
    }

    //fprintf(stderr,"no_of_normals=%d\n", no_of_normals);
    if (no_of_normals > 0)
    {
        switch (normalbinding)
        {
        case Bind::PerVertex:
        {
            //fprintf(stderr,"COVER INFO: normalbinding per vertex\n");

            osg::Vec3Array *normalArray = new osg::Vec3Array();

            int ind = 0;
            for (int i = 0; i < no_of_lines; i++)
            {
                int numv;
                if (i == no_of_lines - 1)
                    numv = no_of_vertices - i_l[i];
                else
                    numv = i_l[i + 1] - i_l[i];
                for (int n = 0; n < numv; n++)
                {
                    osg::Vec3 norm = osg::Vec3(nx[ind], ny[ind], nz[ind]);
                    norm.normalize();
                    normalArray->push_back(norm);
                    ind++;
                }
            }
            geom->setNormalArray(normalArray);
            geom->setNormalBinding(osg::Geometry::BIND_PER_VERTEX);
        }
        break;

        case Bind::OverAll:
        {
            osg::Vec3Array *normalArray = new osg::Vec3Array();

            osg::Vec3 n = osg::Vec3(nx[0], ny[0], nz[0]);
            n.normalize();
            normalArray->push_back(n);
            geom->setNormalArray(normalArray);
            geom->setNormalBinding(osg::Geometry::BIND_OVERALL);
        }
        break;

        case Bind::PerFace:
        {
            osg::Vec3Array *normalArray = new osg::Vec3Array();
            for (int i = 0; i < no_of_lines; i++)
            {
                osg::Vec3 n = osg::Vec3(nx[i], ny[i], nz[i]);
                n.normalize();

                int numv;
                if (i == no_of_lines - 1)
                    numv = no_of_vertices - i_l[i];
                else
                    numv = i_l[i + 1] - i_l[i];
                for (int j = 0; j < numv; ++j)
                    normalArray->push_back(n);
            }
            geom->setNormalArray(normalArray);
            geom->setNormalBinding(osg::Geometry::BIND_PER_VERTEX);
        }
        break;
        }
    }

    geode->addDrawable(geom);

    osg::StateSet *geoState = geode->getOrCreateStateSet();

    if (no_of_normals > 0)
    {

        osg::Uniform *LightNumUniform = new osg::Uniform("LightNum", -1);
        LightNumUniform->setUpdateCallback(new ShaderCallback());
        geoState->addUniform(LightNumUniform);
        osg::Program *program = new osg::Program;
        program->setName("Lightshader");
        program->addShader(new osg::Shader(osg::Shader::VERTEX, shaderVertSource));
        geoState->setAttributeAndModes(program, osg::StateAttribute::ON);
        setDefaultMaterial(geoState, transparent);
    }
    else
        setDefaultMaterial(geoState, transparent, material, false);

    if (no_of_texCoords > 0)
    {
        osg::Vec2Array *tcArray = new osg::Vec2Array();

        for (int i = 0; i < no_of_lines; i++)
        {
            int numv;
            if (i == no_of_lines - 1)
                numv = no_of_vertices - i_l[i];
            else
                numv = i_l[i + 1] - i_l[i];
            for (int n = 0; n < numv; n++)
            {
                int v = v_l[i_l[i] + n];
                tcArray->push_back(osg::Vec2(tx[v], ty[v]));
            }
        }
        geom->setTexCoordArray(0, tcArray);
    }

    osg::BlendFunc *blendFunc = new osg::BlendFunc();
    blendFunc->setFunction(osg::BlendFunc::SRC_ALPHA,
                           osg::BlendFunc::ONE_MINUS_SRC_ALPHA);
    geoState->setAttributeAndModes(blendFunc,
                                   osg::StateAttribute::ON);
    osg::AlphaFunc *alphaFunc = new osg::AlphaFunc();
    alphaFunc->setFunction(osg::AlphaFunc::ALWAYS, 1.0);

    geoState->setAttributeAndModes(alphaFunc,
                                   osg::StateAttribute::OFF);

    osg::LineWidth *lineWidth = new osg::LineWidth(linewidth);
    geoState->setAttributeAndModes(lineWidth, osg::StateAttribute::ON);

    setTexture(image, pixelSize, texWidth, texHeight, geoState, wm, minfm, magfm);

    geode->setStateSet(geoState);
    return ((osg::Node *)geode);
}

osg::Node *
GeometryManager::addPoint(const char *object_name, int no_of_points,
                          float *x_c, float *y_c, float *z_c,
                          int colorbinding, int colorpacking,
                          float *r, float *g, float *b, int *pc, coMaterial *material,
                          float pointsize)

{
    if (no_of_points == 0)
    {
        osg::Group *g = new osg::Group(); // add a dummy object so that we don`t have missing timesteps if object is empty
        g->setName(object_name);
        return g;
    }

    //    if(material)
    //    {
    //       cerr << "addPoint: material ignored" << endl;
    //    }

    osg::Geode *geode = new osg::Geode();
    geode->setName(object_name);
    osg::Geometry *geom = new osg::Geometry();
    geom->setUseDisplayList(coVRConfig::instance()->useDisplayLists());
    geom->setUseVertexBufferObjects(coVRConfig::instance()->useVBOs());

    // set up geometry
    osg::Vec3Array *vert = new osg::Vec3Array;
    osg::DrawArrayLengths *primitives = new osg::DrawArrayLengths(osg::PrimitiveSet::POINTS);
    primitives->push_back(no_of_points);
    for (int i = 0; i < no_of_points; i++)
    {
        vert->push_back(osg::Vec3(x_c[i], y_c[i], z_c[i]));
    }
    geom->setVertexArray(vert);
    geom->addPrimitiveSet(primitives);

    bool transparent = false;
    switch (colorbinding)
    {
    case Bind::PerVertex:
    {
        //fprintf(stderr,"COVER INFO: colorbinding per vertex\n");

        osg::Vec4Array *colArr = new osg::Vec4Array();

        for (int i = 0; i < no_of_points; i++)
        {
            if (colorpacking == Pack::RGBA)
            {
                float r, g, b, a;
                unpackRGBA(pc, i, &r, &g, &b, &a);
                if (a < 1.f)
                    transparent = true;
                colArr->push_back(osg::Vec4(r, g, b, a));
            }
            else
                colArr->push_back(osg::Vec4(r[i], g[i], b[i], 1.0f));
        }
        geom->setColorArray(colArr);
        geom->setColorBinding(osg::Geometry::BIND_PER_VERTEX);
    }
    break;

    case Bind::OverAll:
    {
        osg::Vec4Array *colArr = new osg::Vec4Array();

        if (colorpacking == Pack::RGBA)
        {
            float r, g, b, a;
            unpackRGBA(pc, 0, &r, &g, &b, &a);
            if (a < 1.f)
                transparent = true;
            colArr->push_back(osg::Vec4(r, g, b, a));
        }
        else
            colArr->push_back(osg::Vec4(r[0], g[0], b[0], 1.0f));
        geom->setColorArray(colArr);
        geom->setColorBinding(osg::Geometry::BIND_OVERALL);
    }
    break;
    default:
        if (material != NULL)
        {
            geom->setColorBinding(osg::Geometry::BIND_OFF);
        }
        else
        {
            osg::Vec4Array *colArr = new osg::Vec4Array();
            colArr->push_back(osg::Vec4(1.0f, 1.0f, 1.0f, 1.0f));
            geom->setColorArray(colArr);
            geom->setColorBinding(osg::Geometry::BIND_OVERALL);
        }
        break;
    }

    geode->addDrawable(geom);

    osg::StateSet *geoState = geode->getOrCreateStateSet();
    // geoState->setGlobalDefaults();
    setDefaultMaterial(geoState, transparent, NULL, false);

    osg::BlendFunc *blendFunc = new osg::BlendFunc();
    blendFunc->setFunction(osg::BlendFunc::SRC_ALPHA, osg::BlendFunc::ONE_MINUS_SRC_ALPHA);
    geoState->setAttributeAndModes(blendFunc, osg::StateAttribute::ON);
    osg::AlphaFunc *alphaFunc = new osg::AlphaFunc();
    alphaFunc->setFunction(osg::AlphaFunc::ALWAYS, 1.0);
    //blendFunc->setFunction(osg::BlendFunc::SRC_ALPHA, osg::BlendFunc::ONE_MINUS_SRC_ALPHA);
    geoState->setAttributeAndModes(alphaFunc, osg::StateAttribute::OFF);

    osg::Point *point = new osg::Point();
    point->setSize(pointsize);
    geoState->setAttributeAndModes(point, osg::StateAttribute::ON);

    geode->setStateSet(geoState);

    return ((osg::Node *)geode);
}

osg::Node *
GeometryManager::addSphere(const char *object_name, int no_of_points,
                           float *x_c, float *y_c, float *z_c,
                           int iRenderMethod,
                           float *radii_c, int colorbinding, float *rgbR_c, float *rgbG_c, float *rgbB_c, int *pc,
                           int no_of_normals, float *nx, float *ny, float *nz, int no_of_vertexAttributes,
                           float *vax, float *vay, float *vaz,
                           coMaterial *material)

{
    osg::Geode *geode = new osg::Geode();
    geode->setName(object_name);

    bool transparent = false;
    if (iRenderMethod == coSphere::RENDER_METHOD_PARTICLE_CLOUD)
        transparent = true;
    if (!transparent && pc)
    {
        for (int i = 0; i < no_of_points; ++i)
        {
            float r, g, b, a;
            unpackRGBA(pc, i, &r, &g, &b, &a);
            if (a != 0.f && a != 1.f)
            {
                transparent = true;
                break;
            }
        }
    }

    osg::StateSet *geoState = geode->getOrCreateStateSet();
    // geoState->setGlobalDefaults();
    setDefaultMaterial(geoState, transparent);

    osg::BlendFunc *blendFunc = new osg::BlendFunc();
    blendFunc->setFunction(osg::BlendFunc::SRC_ALPHA, osg::BlendFunc::ONE_MINUS_SRC_ALPHA);
    geoState->setAttributeAndModes(blendFunc, osg::StateAttribute::ON);
    osg::AlphaFunc *alphaFunc = new osg::AlphaFunc();
    alphaFunc->setFunction(osg::AlphaFunc::ALWAYS, 1.0);
    //blendFunc->setFunction(osg::BlendFunc::SRC_ALPHA, osg::BlendFunc::ONE_MINUS_SRC_ALPHA);
    geoState->setAttributeAndModes(alphaFunc, osg::StateAttribute::OFF);
    if (iRenderMethod == coSphere::RENDER_METHOD_TEXTURE)
    {
        osg::Texture2D *tex = coVRFileManager::instance()->loadTexture("share/covise/materials/textures/Sphere.tiff");
        if (tex)
        {
            tex->setFilter(osg::Texture2D::MIN_FILTER, osg::Texture2D::LINEAR_MIPMAP_LINEAR);
            tex->setFilter(osg::Texture2D::MAG_FILTER, osg::Texture2D::LINEAR);
            tex->setWrap(osg::Texture2D::WRAP_S, osg::Texture::CLAMP_TO_EDGE);
            tex->setWrap(osg::Texture2D::WRAP_T, osg::Texture::CLAMP_TO_EDGE);
            geoState->setTextureAttributeAndModes(0, tex, osg::StateAttribute::ON);
        }
    }
    if (material != NULL)
        setDefaultMaterial(geoState, transparent, material);
    geode->setStateSet(geoState);

    coSphere *sphere = new coSphere();
    sphere->setRenderMethod((coSphere::RenderMethod)iRenderMethod);
    sphere->setCoords(no_of_points, x_c, y_c, z_c, radii_c);
    if ((colorbinding == Bind::OverAll || colorbinding == Bind::None) && material != NULL)
    {
        sphere->updateColors(&material->diffuseColor[0], &material->diffuseColor[1], &material->diffuseColor[2]);
        sphere->setColorBinding(Bind::OverAll);
    }
    else
    {
        if (pc)
            sphere->updateColors(pc);
        else
            sphere->updateColors(rgbR_c, rgbG_c, rgbB_c);
        sphere->setColorBinding(colorbinding);
    }
    if (no_of_normals >= no_of_points && nx && ny && nz)
        sphere->updateNormals(nx, ny, nz);

    if (no_of_vertexAttributes > 0)
    {
        osg::Vec3Array *vertArray = new osg::Vec3Array;
        for (int i = 0; i < no_of_points; i++)
        {
            vertArray->push_back(osg::Vec3(vax[i], vay[i], vaz[i]));
        }
        sphere->setVertexAttribArray(6, vertArray);
    }
    geode->addDrawable(sphere);

    return geode;
}
