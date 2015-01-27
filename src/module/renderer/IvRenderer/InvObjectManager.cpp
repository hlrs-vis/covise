/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/* $Log:  $
 * Revision 1.1  1994/04/12  13:39:31  zrfu0125
 * Initial revision
 * */

//**************************************************************************
//
// * Description    : This is the object manager for the renderer
//
//
// * Class(es)      : InvObjectManager
//
//
// * inherited from : none
//
//
// * Author  : Dirk Rantzau, Uwe Woessner
//
//
// * History : 29.03.94 V 1.0
//
//
//
//**************************************************************************

//
// certain def's for our renderer
//
#include <covise/covise.h>
#include "InvObjectManager.h"
#include "InvObjectManagerExtern.h"
#include "InvError.h"
#include <Inventor/nodes/SoShapeHints.h>
#include <Inventor/nodes/SoLabel.h>
#include "InvCoviseViewer.h"

#ifndef WITHOUT_VIRVO
#include "SoVolume.h"
#include <virvo/vvdebugmsg.h>
#endif

//
// X11 stuff
//
#include <X11/Intrinsic.h>
#include <X11/StringDefs.h>
#include <X11/keysym.h>

//
// C stuff
//
#include <unistd.h>

//
// Covise List class AWe
//
#include <util/covise_list.h>

// List of timestep switches AW and timesteppers now class variables

InvSequencer *InvObjectManager::timeStepper;
List<SoSwitch> InvObjectManager::timestepSwitchList;

//
// the externally defined object and render manager
//
extern InvObjectManager *om;
extern InvRenderManager *rm;
void setMeshCoords(SoCoordinate3 *coord, SoQuadMesh *quadmesh, int VerticesPerRow, int VerticesPerColumn,
                   float *x_c, float *y_c, float *z_c);

//======================================================================
// InvObjectManager
//======================================================================

//======================================================================
// add Grids
//======================================================================

SoGroup *om_addUGridCB(const char *object, const char *rootName, int xsize, int ysize,
                       int zsize, float xmin, float xmax,
                       float ymin, float ymax, float zmin, float zmax,
                       int no_of_colors, int colorbinding, int colorpacking,
                       float *r, float *g, float *b, uint32_t *pc, unsigned char *byteData,
                       int no_of_normals, int normalbinding,
                       float *nx, float *ny, float *nz, float transparency)
{
    int j, k;
    float *x_c, *y_c, *z_c, dj, dk;
    float *x_ct, *y_ct, *z_ct;
    SoGroup *root;
    InvQuadmesh *mesh = NULL;
    if (zsize == 1)
    {
        mesh = new InvQuadmesh(colorpacking);
        root = (SoGroup *)mesh->getTopNode();
        x_ct = x_c = new float[xsize * ysize];
        y_ct = y_c = new float[xsize * ysize];
        z_ct = z_c = new float[xsize * ysize];
        dj = (xmax - xmin) / (float)(xsize - 1);
        dk = (ymax - ymin) / (float)(ysize - 1);
        for (j = 0; j < xsize; j++)
            for (k = 0; k < ysize; k++)
            {
                *(z_ct++) = zmin;
                *(x_ct++) = xmin + j * dj;
                *(y_ct++) = ymin + k * dk;
            }
        mesh->setCoords(ysize, xsize, x_c, y_c, z_c);
    }
    else if (ysize == 1)
    {
        mesh = new InvQuadmesh(colorpacking);
        root = (SoGroup *)mesh->getTopNode();
        x_ct = x_c = new float[xsize * zsize];
        y_ct = y_c = new float[xsize * zsize];
        z_ct = z_c = new float[xsize * zsize];
        dj = (xmax - xmin) / (float)(xsize - 1);
        dk = (zmax - zmin) / (float)(zsize - 1);
        for (j = 0; j < xsize; j++)
            for (k = 0; k < zsize; k++)
            {
                *(y_ct++) = ymin;
                *(x_ct++) = xmin + j * dj;
                *(z_ct++) = zmin + k * dk;
            }
        mesh->setCoords(zsize, xsize, x_c, y_c, z_c);
    }
    else if (xsize == 1)
    {
        mesh = new InvQuadmesh(colorpacking);
        root = (SoGroup *)mesh->getTopNode();
        x_ct = x_c = new float[ysize * zsize];
        y_ct = y_c = new float[ysize * zsize];
        z_ct = z_c = new float[ysize * zsize];
        dj = (ymax - ymin) / (float)(ysize - 1);
        dk = (zmax - zmin) / (float)(zsize - 1);
        for (j = 0; j < ysize; j++)
            for (k = 0; k < zsize; k++)
            {
                *(x_ct++) = xmin;
                *(y_ct++) = ymin + j * dj;
                *(z_ct++) = zmin + k * dk;
            }
        mesh->setCoords(zsize, ysize, x_c, y_c, z_c);
    }
    else
    {
#ifdef WITHOUT_VIRVO
        return NULL;
#else
        // hier Volume elemente auspacken

        SoSwitch *top_switch = new SoSwitch;
        SoSeparator *root = new SoSeparator;
        SoLabel *objName = new SoLabel;
        SoTransform *transform = new SoTransform;
        SoPackedColor *color = new SoPackedColor;
        SoMaterialBinding *matbind = new SoMaterialBinding;
        SoDrawStyle *drawstyle = new SoDrawStyle;

        SoVolume *myVolume = new SoVolume();
        myVolume->init(xsize, ysize, zsize,
                       xmin, xmax, ymin, ymax, zmin, zmax, colorpacking, r, g, b, (uchar *)pc, byteData);

        root->renderCaching = SoSeparator::OFF;
        //
        // create object tree
        //
        //root->ref(); // is this correct

        top_switch->addChild(root);
        root->addChild(objName);
        root->addChild(transform);
        root->addChild(drawstyle);
        root->addChild(color);
        root->addChild(matbind);
        root->addChild(myVolume);
        objName->label.setValue(object);
        char *g_name = new char[strlen(object) + 3];
        strcpy(g_name, "G_");
        strcat(g_name, object);
        myVolume->setName(SbName(g_name));
        strcpy(g_name, "S_");
        strcat(g_name, object);
        top_switch->setName(SbName(g_name));
        //
        // default settings
        //
        top_switch->whichChild.setValue(0);
        drawstyle->lineWidth.setValue(2);
        matbind->value = SoMaterialBinding::OVERALL;
        if (rootName == NULL)
            rm->addToSceneGraph((SoGroup *)top_switch, object, NULL);
        else
            rm->addToSceneGraph((SoGroup *)top_switch, object, (om->list->search(rootName))->getObject());
        om_addObjectCB(object, (SoGroup *)top_switch);
        return (SoGroup *)top_switch;
#endif
    }
    delete x_c;
    delete y_c;
    delete z_c;
    mesh->setVertexOrdering(2); // Turn on two-sided lighting
    mesh->setColorBinding(colorbinding);

    if (colorbinding != INV_NONE)
    {
        if (colorpacking == INV_RGBA)
            mesh->setColors(no_of_colors, pc);
        else
            mesh->setColors(no_of_colors, r, g, b);
    }

    mesh->setNormalBinding(normalbinding);

    if (normalbinding != INV_NONE)
        mesh->setNormals(no_of_normals, nx, ny, nz);

    mesh->setTransparency(transparency);
    mesh->setName(object);

    // create an inventor object and replace it with the current
    // scenegraph
    //

    if (rootName == NULL)
        rm->addToSceneGraph(root, object, NULL);
    else
        rm->addToSceneGraph(root, object, (om->list->search(rootName))->getObject());
    om_addObjectCB(object, root);

    print_comment(__LINE__, __FILE__, "UGrid object added successfully");
    return root;
}

//----------------------------------------------------------------------------------
//
//----------------------------------------------------------------------------------
SoGroup *om_replaceUGridCB(const char *object, int xsize, int ysize,
                           int zsize, float xmin, float xmax,
                           float ymin, float ymax, float zmin, float zmax,
                           int no_of_colors, int colorbinding, int colorpacking,
                           float *r, float *g, float *b, uint32_t *pc, int no_of_normals,
                           int normalbinding,
                           float *nx, float *ny, float *nz, float transparency)
{
    int numnodes, i, k = 0, j = 0;
    SoGroup *root = NULL;
    SoNode *node;
    SoGroup *sep;
    SoCoordinate3 *coord = NULL;
    SoMaterial *material = NULL;
    SoPackedColor *color = NULL;
    SoMaterialBinding *matbind = NULL;
    SoNormal *normal = NULL;
    SoNormalBinding *normbind = NULL;
    SoQuadMesh *quadmesh = NULL;
    //  float *coord_points=NULL;        // eliminate Warning AW
    float *colors, *n;
    float *x_c, *y_c, *z_c, dj, dk;
    float *x_ct, *y_ct, *z_ct;
    InvObject *obj = om->list->search(object);
    if (obj != NULL)
    {
        root = obj->getObject(); // the top switch
        sep = (SoGroup *)root->getChild(0); // the separator
        numnodes = sep->getNumChildren();
        for (i = 0; i < numnodes; i++)
        {
            node = sep->getChild(i);
            if (node->isOfType(SoCoordinate3::getClassTypeId()))
                coord = (SoCoordinate3 *)node;
            if (node->isOfType(SoMaterial::getClassTypeId()))
                material = (SoMaterial *)node;
            if (node->isOfType(SoPackedColor::getClassTypeId()))
                color = (SoPackedColor *)node;
            if (node->isOfType(SoMaterialBinding::getClassTypeId()))
                matbind = (SoMaterialBinding *)node;
            if (node->isOfType(SoNormal::getClassTypeId()))
                normal = (SoNormal *)node;
            if (node->isOfType(SoNormalBinding::getClassTypeId()))
                normbind = (SoNormalBinding *)node;
            if (node->isOfType(SoQuadMesh::getClassTypeId()))
                quadmesh = (SoQuadMesh *)node;
        }
    }
    //
    // check what we got
    //
    if (coord == NULL || quadmesh == NULL)
    {
        print_comment(__LINE__, __FILE__, "ERROR: Replaced set has different dimension");
        return NULL;
    }
    //
    // store new coordinates
    //
    if (zsize == 1)
    {
        x_ct = x_c = new float[xsize * ysize];
        y_ct = y_c = new float[xsize * ysize];
        z_ct = z_c = new float[xsize * ysize];
        dj = (xmax - xmin) / (float)(xsize - 1);
        dk = (ymax - ymin) / (float)(ysize - 1);
        for (j = 0; j < xsize; j++)
            for (k = 0; k < ysize; k++)
            {
                *(z_ct++) = zmin;
                *(x_ct++) = xmin + j * dj;
                *(y_ct++) = ymin + k * dk;
            }
        setMeshCoords(coord, quadmesh, ysize, xsize, x_c, y_c, z_c);
    }
    else if (ysize == 1)
    {
        x_ct = x_c = new float[xsize * zsize];
        y_ct = y_c = new float[xsize * zsize];
        z_ct = z_c = new float[xsize * zsize];
        dj = (xmax - xmin) / (float)(xsize - 1);
        dk = (zmax - zmin) / (float)(zsize - 1);
        for (j = 0; j < xsize; j++)
            for (k = 0; k < zsize; k++)
            {
                *(y_ct++) = ymin;
                *(x_ct++) = xmin + j * dj;
                *(z_ct++) = zmin + k * dk;
            }
        setMeshCoords(coord, quadmesh, zsize, xsize, x_c, y_c, z_c);
    }
    else
    {
        x_ct = x_c = new float[ysize * zsize];
        y_ct = y_c = new float[ysize * zsize];
        z_ct = z_c = new float[ysize * zsize];
        dj = (ymax - ymin) / (float)(ysize - 1);
        dk = (zmax - zmin) / (float)(zsize - 1);
        for (j = 0; j < ysize; j++)
            for (k = 0; k < zsize; k++)
            {
                *(x_ct++) = xmin;
                *(y_ct++) = ymin + j * dj;
                *(z_ct++) = zmin + k * dk;
            }
        setMeshCoords(coord, quadmesh, zsize, ysize, x_c, y_c, z_c);
    }
    delete x_c;
    delete y_c;
    delete z_c;
    //
    // store new colorbinding
    //
    if (colorbinding == INV_PER_VERTEX)
        matbind->value = SoMaterialBinding::PER_VERTEX_INDEXED;
    else if (colorbinding == INV_PER_FACE)
        matbind->value = SoMaterialBinding::PER_FACE;
    else if (colorbinding == INV_NONE)
        matbind->value = SoMaterialBinding::OVERALL;
    else if (colorbinding == INV_OVERALL)
        matbind->value = SoMaterialBinding::OVERALL;
    if (colorbinding != INV_NONE)
    {
        if (colorpacking == INV_RGBA)
        {
            color->orderedRGBA.setValues(0, no_of_colors, (const uint32_t *)pc);
        }
        else
        {
            //
            // store new colors
            //
            colors = new float[no_of_colors * 3];
            if (colors != NULL)
            {
                k = 0;
                for (int j = 0; j < no_of_colors; j++)
                {
                    *(colors + k) = *(r + j);
                    *(colors + k + 1) = *(g + j);
                    *(colors + k + 2) = *(b + j);
                    k = k + 3;
                }
                material->diffuseColor.deleteValues(0);
                material->diffuseColor.setValues(0, no_of_colors, (float(*)[3])colors);
                delete colors;
            }
            else
            {
                print_comment(__LINE__, __FILE__, "ERROR: memory allocation failed");
                return NULL;
            }
        }
    }
    //
    // store new normalbinding
    //
    if (normalbinding == INV_PER_VERTEX)
        normbind->value = SoNormalBinding::PER_VERTEX_INDEXED;
    else if (normalbinding == INV_PER_FACE)
        normbind->value = SoNormalBinding::PER_FACE;
    else if (normalbinding == INV_NONE)
        normbind->value = SoNormalBinding::DEFAULT;
    if (normalbinding != INV_NONE)
    {
        //
        // store new normals
        //
        n = new float[no_of_normals * 3];
        if (n != NULL)
        {
            k = 0;
            for (j = 0; j < no_of_normals; j++)
            {
                *(n + k) = *(nx + j);
                *(n + k + 1) = *(ny + j);
                *(n + k + 2) = *(nz + j);
                k = k + 3;
            }
            normal->vector.deleteValues(0);
            normal->vector.setValues(0, no_of_normals, (float(*)[3])n);
            delete n;
        }
        else
            print_comment(__LINE__, __FILE__, "ERROR: memory allocation failed");
    }

    if (colorpacking != INV_RGBA)
        material->transparency.setValue(transparency);

    print_comment(__LINE__, __FILE__, "SGrid object replaced successfully");
    return root;
}

//----------------------------------------------------------------------------------
//
//----------------------------------------------------------------------------------
SoGroup *om_addRGridCB(const char *object, const char *rootName, int xsize, int ysize,
                       int zsize, float *x_c, float *y_c,
                       float *z_c,
                       int no_of_colors, int colorbinding, int colorpacking,
                       float *r, float *g, float *b, uint32_t *pc, int no_of_normals,
                       int normalbinding,
                       float *nx, float *ny, float *nz, float transparency)
{
    int j, k;
    float *x_d, *y_d, *z_d;
    float *x_ct, *y_ct, *z_ct;
    SoGroup *root;

    InvQuadmesh *mesh = new InvQuadmesh(colorpacking);
    root = (SoGroup *)mesh->getTopNode();
    if (zsize == 1)
    {
        x_ct = x_d = new float[xsize * ysize];
        y_ct = y_d = new float[xsize * ysize];
        z_ct = z_d = new float[xsize * ysize];
        for (j = 0; j < xsize; j++)
            for (k = 0; k < ysize; k++)
            {
                *(z_ct++) = *z_c;
                *(x_ct++) = *(x_c + j);
                *(y_ct++) = *(y_c + k);
            }
        mesh->setCoords(ysize, xsize, x_d, y_d, z_d);
    }
    else if (ysize == 1)
    {
        x_ct = x_d = new float[xsize * zsize];
        y_ct = y_d = new float[xsize * zsize];
        z_ct = z_d = new float[xsize * zsize];
        for (j = 0; j < xsize; j++)
            for (k = 0; k < zsize; k++)
            {
                *(y_ct++) = *y_c;
                *(x_ct++) = *(x_c + j);
                *(z_ct++) = *(z_c + k);
            }
        mesh->setCoords(zsize, xsize, x_d, y_d, z_d);
    }
    else
    {
        x_ct = x_d = new float[ysize * zsize];
        y_ct = y_d = new float[ysize * zsize];
        z_ct = z_d = new float[ysize * zsize];
        for (j = 0; j < ysize; j++)
            for (k = 0; k < zsize; k++)
            {
                *(x_ct++) = *x_c;
                *(y_ct++) = *(y_c + j);
                *(z_ct++) = *(z_c + k);
            }
        mesh->setCoords(zsize, ysize, x_d, y_d, z_d);
    }
    delete x_d;
    delete y_d;
    delete z_d;
    mesh->setVertexOrdering(2); // Turn on two-sided lighting
    mesh->setColorBinding(colorbinding);

    if (colorbinding != INV_NONE)
    {
        if (colorpacking == INV_RGBA)
            mesh->setColors(no_of_colors, pc);
        else
            mesh->setColors(no_of_colors, r, g, b);
    }
    mesh->setNormalBinding(normalbinding);

    if (normalbinding != INV_NONE)
        mesh->setNormals(no_of_normals, nx, ny, nz);

    mesh->setTransparency(transparency);
    mesh->setName(object);

    // create an inventor object and replace it with the current
    // scenegraph
    //

    if (rootName == NULL)
        rm->addToSceneGraph(root, object, NULL);
    else
        rm->addToSceneGraph(root, object, (om->list->search(rootName))->getObject());
    om_addObjectCB(object, root);

    print_comment(__LINE__, __FILE__, "RGrid object added successfully");
    return root;
}

//----------------------------------------------------------------------------------
//
//----------------------------------------------------------------------------------
SoGroup *om_replaceRGridCB(const char *object, int xsize, int ysize,
                           int zsize, float *x_c, float *y_c,
                           float *z_c,
                           int no_of_colors, int colorbinding, int colorpacking,
                           float *r, float *g, float *b, uint32_t *pc, int no_of_normals,
                           int normalbinding,
                           float *nx, float *ny, float *nz, float transparency)
{
    int numnodes, i, k = 0, j = 0;
    SoGroup *root = NULL;
    SoNode *node;
    SoGroup *sep;
    SoCoordinate3 *coord = NULL;
    SoPackedColor *color = NULL;
    SoMaterial *material = NULL;
    SoMaterialBinding *matbind = NULL;
    SoNormal *normal = NULL;
    SoNormalBinding *normbind = NULL;
    SoQuadMesh *quadmesh = NULL;
    //float *coord_points=NULL;        // eliminate Warning AW
    float *colors, *n;
    float *x_d, *y_d, *z_d;
    float *x_ct, *y_ct, *z_ct;
    InvObject *obj = om->list->search(object);
    if (obj != NULL)
    {
        root = obj->getObject();
        sep = (SoGroup *)root->getChild(0);
        numnodes = sep->getNumChildren();
        for (i = 0; i < numnodes; i++)
        {
            node = sep->getChild(i);
            if (node->isOfType(SoCoordinate3::getClassTypeId()))
                coord = (SoCoordinate3 *)node;
            if (node->isOfType(SoMaterial::getClassTypeId()))
                material = (SoMaterial *)node;
            if (node->isOfType(SoPackedColor::getClassTypeId()))
                color = (SoPackedColor *)node;
            if (node->isOfType(SoMaterialBinding::getClassTypeId()))
                matbind = (SoMaterialBinding *)node;
            if (node->isOfType(SoNormal::getClassTypeId()))
                normal = (SoNormal *)node;
            if (node->isOfType(SoNormalBinding::getClassTypeId()))
                normbind = (SoNormalBinding *)node;
            if (node->isOfType(SoQuadMesh::getClassTypeId()))
                quadmesh = (SoQuadMesh *)node;
        }
    }
    //
    // check what we got
    //
    if (coord == NULL || quadmesh == NULL)
    {
        print_comment(__LINE__, __FILE__, "ERROR: Replaced set has different dimension");
        return NULL;
    }
    //
    // store new coordinates
    //
    if (zsize == 1)
    {
        x_ct = x_d = new float[xsize * ysize];
        y_ct = y_d = new float[xsize * ysize];
        z_ct = z_d = new float[xsize * ysize];
        for (j = 0; j < xsize; j++)
            for (k = 0; k < ysize; k++)
            {
                *(z_ct++) = *z_c;
                *(x_ct++) = *(x_c + j);
                *(y_ct++) = *(y_c + k);
            }
        setMeshCoords(coord, quadmesh, ysize, xsize, x_d, y_d, z_d);
    }
    else if (ysize == 1)
    {
        x_ct = x_d = new float[xsize * zsize];
        y_ct = y_d = new float[xsize * zsize];
        z_ct = z_d = new float[xsize * zsize];
        for (j = 0; j < xsize; j++)
            for (k = 0; k < zsize; k++)
            {
                *(y_ct++) = *y_c;
                *(x_ct++) = *(x_c + j);
                *(z_ct++) = *(z_c + k);
            }
        setMeshCoords(coord, quadmesh, zsize, xsize, x_d, y_d, z_d);
    }
    else
    {
        x_ct = x_d = new float[ysize * zsize];
        y_ct = y_d = new float[ysize * zsize];
        z_ct = z_d = new float[ysize * zsize];
        for (j = 0; j < ysize; j++)
            for (k = 0; k < zsize; k++)
            {
                *(x_ct++) = *x_c;
                *(y_ct++) = *(y_c + j);
                *(z_ct++) = *(z_c + k);
            }
        setMeshCoords(coord, quadmesh, zsize, ysize, x_d, y_d, z_d);
    }
    delete x_d;
    delete y_d;
    delete z_d;
    //
    // store new colorbinding
    //
    if (colorbinding == INV_PER_VERTEX)
        matbind->value = SoMaterialBinding::PER_VERTEX_INDEXED;
    else if (colorbinding == INV_PER_FACE)
        matbind->value = SoMaterialBinding::PER_FACE;
    else if (colorbinding == INV_NONE)
        matbind->value = SoMaterialBinding::OVERALL;
    else if (colorbinding == INV_OVERALL)
        matbind->value = SoMaterialBinding::OVERALL;
    if (colorbinding != INV_NONE)
    {
        if (colorpacking == INV_RGBA)
        {
            color->orderedRGBA.setValues(0, no_of_colors, (const uint32_t *)pc);
        }
        else
        {
            //
            // store new colors
            //
            colors = new float[no_of_colors * 3];
            if (colors != NULL)
            {
                k = 0;
                for (int j = 0; j < no_of_colors; j++)
                {
                    *(colors + k) = *(r + j);
                    *(colors + k + 1) = *(g + j);
                    *(colors + k + 2) = *(b + j);
                    k = k + 3;
                }
                material->diffuseColor.deleteValues(0);
                material->diffuseColor.setValues(0, no_of_colors, (float(*)[3])colors);
                delete colors;
            }
            else
            {
                print_comment(__LINE__, __FILE__, "ERROR: memory allocation failed");
                return NULL;
            }
        }
    }
    //
    // store new normalbinding
    //
    if (normalbinding == INV_PER_VERTEX)
        normbind->value = SoNormalBinding::PER_VERTEX_INDEXED;
    else if (normalbinding == INV_PER_FACE)
        normbind->value = SoNormalBinding::PER_FACE;
    else if (normalbinding == INV_NONE)
        normbind->value = SoNormalBinding::DEFAULT;
    if (normalbinding != INV_NONE)
    {
        //
        // store new normals
        //
        n = new float[no_of_normals * 3];
        if (n != NULL)
        {
            k = 0;
            for (j = 0; j < no_of_normals; j++)
            {
                *(n + k) = *(nx + j);
                *(n + k + 1) = *(ny + j);
                *(n + k + 2) = *(nz + j);
                k = k + 3;
            }
            normal->vector.deleteValues(0);
            normal->vector.setValues(0, no_of_normals, (float(*)[3])n);
            delete n;
        }
        else
            print_comment(__LINE__, __FILE__, "ERROR: memory allocation failed");
    }
    if (colorpacking != INV_RGBA)
        material->transparency.setValue(transparency);

    print_comment(__LINE__, __FILE__, "SGrid object replaced successfully");
    return root;
}

//----------------------------------------------------------------------------------
//
//----------------------------------------------------------------------------------
SoGroup *om_addSGridCB(const char *object, const char *rootName, int xsize, int ysize,
                       int zsize, float *x_c, float *y_c,
                       float *z_c,
                       int no_of_colors, int colorbinding, int colorpacking,
                       float *r, float *g, float *b, uint32_t *pc, int no_of_normals,
                       int normalbinding,
                       float *nx, float *ny, float *nz, float transparency)
{
    SoGroup *root;

    InvQuadmesh *mesh = new InvQuadmesh(colorpacking);
    root = (SoGroup *)mesh->getTopNode();
    if (zsize == 1)
    {
        mesh->setCoords(ysize, xsize, x_c, y_c, z_c);
    }
    else if (ysize == 1)
    {
        mesh->setCoords(zsize, xsize, x_c, y_c, z_c);
    }
    else
    {
        mesh->setCoords(zsize, ysize, x_c, y_c, z_c);
    }
    mesh->setVertexOrdering(2); // Turn on two-sided lighting
    mesh->setColorBinding(colorbinding);

    if (colorbinding != INV_NONE)
    {
        if (colorpacking == INV_RGBA)
            mesh->setColors(no_of_colors, pc);
        else
            mesh->setColors(no_of_colors, r, g, b);
    }
    mesh->setNormalBinding(normalbinding);

    if (normalbinding != INV_NONE)
        mesh->setNormals(no_of_normals, nx, ny, nz);

    mesh->setTransparency(transparency);
    mesh->setName(object);

    // create an inventor object and replace it with the current
    // scenegraph
    //

    if (rootName == NULL)
        rm->addToSceneGraph(root, object, NULL);
    else
        rm->addToSceneGraph(root, object, (om->list->search(rootName))->getObject());
    om_addObjectCB(object, root);

    print_comment(__LINE__, __FILE__, "SGrid object added successfully");
    return root;
}

//----------------------------------------------------------------------------------
//
//----------------------------------------------------------------------------------
SoGroup *om_replaceSGridCB(const char *object, int xsize, int ysize,
                           int zsize, float *x_c, float *y_c,
                           float *z_c,
                           int no_of_colors, int colorbinding, int colorpacking,
                           float *r, float *g, float *b, uint32_t *pc, int no_of_normals,
                           int normalbinding,
                           float *nx, float *ny, float *nz, float transparency)
{
    int numnodes, i, k = 0, j = 0;
    SoGroup *root = NULL;
    SoNode *node;
    SoGroup *sep;
    SoCoordinate3 *coord = NULL;
    SoPackedColor *color = NULL;
    SoMaterial *material = NULL;
    SoMaterialBinding *matbind = NULL;
    SoNormal *normal = NULL;
    SoNormalBinding *normbind = NULL;
    SoQuadMesh *quadmesh = NULL;
    //float *coord_points=NULL;        // eliminate Warning AW
    float *colors, *n;
    InvObject *obj = om->list->search(object);
    if (obj != NULL)
    {
        root = obj->getObject();
        sep = (SoGroup *)root->getChild(0);
        numnodes = sep->getNumChildren();
        for (i = 0; i < numnodes; i++)
        {
            node = sep->getChild(i);
            if (node->isOfType(SoCoordinate3::getClassTypeId()))
                coord = (SoCoordinate3 *)node;
            if (node->isOfType(SoMaterial::getClassTypeId()))
                material = (SoMaterial *)node;
            if (node->isOfType(SoPackedColor::getClassTypeId()))
                color = (SoPackedColor *)node;
            if (node->isOfType(SoPackedColor::getClassTypeId()))
                color = (SoPackedColor *)node;
            if (node->isOfType(SoMaterialBinding::getClassTypeId()))
                matbind = (SoMaterialBinding *)node;
            if (node->isOfType(SoNormal::getClassTypeId()))
                normal = (SoNormal *)node;
            if (node->isOfType(SoNormalBinding::getClassTypeId()))
                normbind = (SoNormalBinding *)node;
            if (node->isOfType(SoQuadMesh::getClassTypeId()))
                quadmesh = (SoQuadMesh *)node;
        }
    }
    //
    // check what we got
    //
    if (coord == NULL || quadmesh == NULL)
    {
        print_comment(__LINE__, __FILE__, "ERROR: Replaced set has different dimension");
        return NULL;
    }

    //
    // store new coordinates
    //
    if (zsize == 1)
    {
        setMeshCoords(coord, quadmesh, ysize, xsize, x_c, y_c, z_c);
    }
    else if (ysize == 1)
    {
        setMeshCoords(coord, quadmesh, zsize, xsize, x_c, y_c, z_c);
    }
    else
    {
        setMeshCoords(coord, quadmesh, zsize, ysize, x_c, y_c, z_c);
    }
    //
    // store new colorbinding
    //
    if (colorbinding == INV_PER_VERTEX)
        matbind->value = SoMaterialBinding::PER_VERTEX_INDEXED;
    else if (colorbinding == INV_PER_FACE)
        matbind->value = SoMaterialBinding::PER_FACE;
    else if (colorbinding == INV_NONE)
        matbind->value = SoMaterialBinding::OVERALL;
    else if (colorbinding == INV_OVERALL)
        matbind->value = SoMaterialBinding::OVERALL;
    if (colorbinding != INV_NONE)
    {
        if (colorpacking == INV_RGBA)
        {
            color->orderedRGBA.setValues(0, no_of_colors, (const uint32_t *)pc);
        }
        else
        {
            //
            // store new colors
            //
            colors = new float[no_of_colors * 3];
            if (colors != NULL)
            {
                k = 0;
                for (int j = 0; j < no_of_colors; j++)
                {
                    *(colors + k) = *(r + j);
                    *(colors + k + 1) = *(g + j);
                    *(colors + k + 2) = *(b + j);
                    k = k + 3;
                }
                material->diffuseColor.deleteValues(0);
                material->diffuseColor.setValues(0, no_of_colors, (float(*)[3])colors);
                delete colors;
            }
            else
            {
                print_comment(__LINE__, __FILE__, "ERROR: memory allocation failed");
                return NULL;
            }
        }
    }
    //
    // store new normalbinding
    //
    if (normalbinding == INV_PER_VERTEX)
        normbind->value = SoNormalBinding::PER_VERTEX_INDEXED;
    else if (normalbinding == INV_PER_FACE)
        normbind->value = SoNormalBinding::PER_FACE;
    else if (normalbinding == INV_NONE)
        normbind->value = SoNormalBinding::DEFAULT;
    if (normalbinding != INV_NONE)
    {
        //
        // store new normals
        //
        n = new float[no_of_normals * 3];
        if (n != NULL)
        {
            k = 0;
            for (j = 0; j < no_of_normals; j++)
            {
                *(n + k) = *(nx + j);
                *(n + k + 1) = *(ny + j);
                *(n + k + 2) = *(nz + j);
                k = k + 3;
            }
            normal->vector.deleteValues(0);
            normal->vector.setValues(0, no_of_normals, (float(*)[3])n);
            delete n;
        }
        else
            print_comment(__LINE__, __FILE__, "ERROR: memory allocation failed");
    }

    if (colorpacking != INV_RGBA)
        material->transparency.setValue(transparency);

    print_comment(__LINE__, __FILE__, "SGrid object replaced successfully");
    return root;
}

//======================================================================
// add polygon
//======================================================================
SoGroup *
om_addPolygonCB(const char *name, const char *rootName, int no_of_polygons, int no_of_vertices,
                int no_of_coords, float *x_c, float *y_c,
                float *z_c, int *v_l, int *l_l, int no_of_colors,
                int colorbinding, int colorpacking,
                float *r, float *g, float *b, uint32_t *pc, int no_of_normals,
                int normalbinding,
                float *nx, float *ny, float *nz, float transparency,
                int vertexOrder,
                int texWidth, int texHeight, int pixelSize, unsigned char *image,
                int no_of_texCoords, float *tx, float *ty, coMaterial *material,
                char *rName, char *label)
{
    SoGroup *root;

    InvPolygon *polygon = new InvPolygon(colorpacking);

    root = (SoGroup *)polygon->getTopNode();
    polygon->setCoords(no_of_polygons, no_of_vertices,
                       no_of_coords, x_c, y_c, z_c,
                       v_l, l_l);

    if (rName)
        polygon->setRealObjName(rName);
    if (label)
        polygon->setGrpLabel(label);

    polygon->setColorBinding(colorbinding);

    if (material)
    {
        polygon->setMaterial(material);
    }

    if (colorbinding != INV_NONE)
    {
        if (colorpacking == INV_RGBA)
        {
            if ((no_of_colors > 1) || (material == NULL))
                polygon->setColors(no_of_colors, pc);
        }
        else if (colorpacking == INV_TEXTURE)
        {
            polygon->setTexture(texWidth, texHeight, pixelSize, image);
            polygon->setTexCoords(no_of_texCoords, tx, ty);
            polygon->setTextureCoordinateBinding(INV_NONE);
            rm->addToTextureList(polygon->getTexture());
        }
        else
        {
            if ((no_of_colors > 1) || (material == NULL))
                polygon->setColors(no_of_colors, r, g, b);
        }
    }
    polygon->setNormalBinding(normalbinding);
    polygon->setVertexOrdering(vertexOrder);

    if (normalbinding != INV_NONE)
    {
        polygon->setNormals(no_of_normals, nx, ny, nz);
    }

    polygon->setTransparency(transparency);
    polygon->setName(name);

    // create an inventor object and replace it with the current
    // scenegraph
    //

    if (rootName == NULL)
        rm->addToSceneGraph(root, name, NULL);
    else
        rm->addToSceneGraph(root, name, (om->list->search(rootName))->getObject());
    om_addObjectCB(name, root);

    print_comment(__LINE__, __FILE__, "polygon object added successfully");
    return root;
}

//======================================================================
// replace polygon
//======================================================================
SoGroup *om_replacePolygonCB(const char *name, int no_of_polygons, int no_of_vertices,
                             int no_of_coords, float *x_c, float *y_c,
                             float *z_c, int *v_l, int *l_l, int no_of_colors,
                             int colorbinding, int colorpacking,
                             float *r, float *g, float *b, uint32_t *pc, int no_of_normals,
                             int normalbinding,
                             float *nx, float *ny, float *nz, float transparency,
                             int vertexOrder,
                             int texWidth, int texHeight, int pixelSize, unsigned char *image,
                             int no_of_texCoords, float *tx, float *ty, coMaterial *comaterial)
{
    int numnodes, i, k = 0, j = 0;
    long no_poly;
    SoGroup *root = NULL;
    SoNode *node;
    SoGroup *sep;
    SoCoordinate3 *coord = NULL;
    SoMaterial *material = NULL;
    SoPackedColor *color = NULL;
    SoMaterialBinding *matbind = NULL;
    SoNormal *normal = NULL;
    SoNormalBinding *normbind = NULL;
    SoTexture2 *texture = NULL;
    SoTextureCoordinate2 *texCoord = NULL;
    SoTextureCoordinateBinding *texbind = NULL;
    SoIndexedFaceSet *polygon = NULL;
    SoShapeHints *shapehints = NULL;
    float *coord_points = NULL;
    float *colors, *n;
    float *tc = NULL; // texture coordinates
    int32_t *vertices;
    InvObject *obj = om->list->search(name);
    (void)comaterial;
    if (obj != NULL)
    {
        root = obj->getObject();
        sep = (SoGroup *)root->getChild(0);
        numnodes = sep->getNumChildren();
        for (i = 0; i < numnodes; i++)
        {
            node = sep->getChild(i);
            if (node->isOfType(SoCoordinate3::getClassTypeId()))
                coord = (SoCoordinate3 *)node;
            if (node->isOfType(SoMaterial::getClassTypeId()))
                material = (SoMaterial *)node;
            if (node->isOfType(SoPackedColor::getClassTypeId()))
                color = (SoPackedColor *)node;
            if (node->isOfType(SoMaterialBinding::getClassTypeId()))
                matbind = (SoMaterialBinding *)node;
            if (node->isOfType(SoNormal::getClassTypeId()))
                normal = (SoNormal *)node;
            if (node->isOfType(SoNormalBinding::getClassTypeId()))
                normbind = (SoNormalBinding *)node;
            if (node->isOfType(SoTexture2::getClassTypeId()))
                texture = (SoTexture2 *)node;
            if (node->isOfType(SoTextureCoordinate2::getClassTypeId()))
                texCoord = (SoTextureCoordinate2 *)node;
            if (node->isOfType(SoTextureCoordinateBinding::getClassTypeId()))
                texbind = (SoTextureCoordinateBinding *)node;
            if (node->isOfType(SoIndexedFaceSet::getClassTypeId()))
                polygon = (SoIndexedFaceSet *)node;
            if (node->isOfType(SoShapeHints::getClassTypeId()))
                shapehints = (SoShapeHints *)node;
        }
    }

    //
    // check what we got
    //
    if (coord == NULL || polygon == NULL)
    {
        print_comment(__LINE__, __FILE__, "ERROR: Replaced set has different dimension");
        return NULL;
    }

    //
    // store new coordinates
    //
    coord_points = new float[no_of_coords * 3];
    vertices = new int[no_of_vertices + no_of_polygons];

    if (coord_points != NULL && vertices != NULL)
    {
        k = 0;
        for (j = 0; j < no_of_coords; j++)
        {
            *(coord_points + k) = *(x_c + j);
            *(coord_points + k + 1) = *(y_c + j);
            *(coord_points + k + 2) = *(z_c + j);
            k = k + 3;
        }
        coord->point.deleteValues(0);
        coord->point.setValues(0, no_of_coords, (float(*)[3])coord_points);

        //
        // store vertices list ( used by normal and coordinate field )
        //

        j = 0;
        for (no_poly = 0; no_poly < no_of_polygons - 1; no_poly++)
        {
            for (k = *(l_l + no_poly); k < *(l_l + no_poly + 1); k++)
            {
                *(vertices + j) = *(v_l + k);
                j++;
            }
            *(vertices + j) = SO_END_FACE_INDEX;
            j++;
        }
        // last polygon
        for (k = *(l_l + no_of_polygons - 1); k < no_of_vertices; k++)
        {
            *(vertices + j) = *(v_l + k);
            j++;
        }
        *(vertices + j) = SO_END_FACE_INDEX;
        polygon->coordIndex.deleteValues(0);
        polygon->coordIndex.setValues(0, no_of_vertices + no_of_polygons, (const int32_t *)vertices);

        delete coord_points;
        delete vertices;
    }
    else
        print_comment(__LINE__, __FILE__, "ERROR: memory allocation failed");
    //
    // store new colorbinding
    //
    if (colorbinding == INV_PER_VERTEX)
        matbind->value = SoMaterialBinding::PER_VERTEX_INDEXED;
    else if (colorbinding == INV_PER_FACE)
        matbind->value = SoMaterialBinding::PER_FACE;
    else if (colorbinding == INV_NONE)
        matbind->value = SoMaterialBinding::OVERALL;
    else if (colorbinding == INV_OVERALL)
        matbind->value = SoMaterialBinding::OVERALL;
    if (colorbinding != INV_NONE)
    {
        if (colorpacking == INV_RGBA)
        {
            color->orderedRGBA.setValues(0, no_of_colors, (const uint32_t *)pc);
        }
        else if (colorpacking == INV_TEXTURE)
        {
            // new texture image
            texture->image.setValue(SbVec2s(texWidth, texHeight), pixelSize, image);
            texture->wrapS.setValue(SoTexture2::CLAMP);
            texture->wrapT.setValue(SoTexture2::CLAMP);
            texbind->value = SoTextureCoordinateBinding::DEFAULT;

            //
            // store new texture coordinates
            //
            tc = new float[no_of_texCoords * 2];
            if (tc != NULL)
            {
                k = 0;
                for (j = 0; j < no_of_texCoords; j++)
                {
                    *(tc + k) = *(tx + j);
                    *(tc + k + 1) = *(ty + j);
                    k = k + 2;
                }
                texCoord->point.setValues(0, no_of_texCoords, (float(*)[2])tc);
            }
            else
            {
                print_comment(__LINE__, __FILE__, "ERROR: memory allocation failed");
                return NULL;
            }
            delete tc;
        }
        else
        {
            //
            // store new colors
            //
            colors = new float[no_of_colors * 3];
            if (colors != NULL)
            {
                k = 0;
                for (int j = 0; j < no_of_colors; j++)
                {
                    *(colors + k) = *(r + j);
                    *(colors + k + 1) = *(g + j);
                    *(colors + k + 2) = *(b + j);
                    k = k + 3;
                }
                material->diffuseColor.deleteValues(0);
                material->diffuseColor.setValues(0, no_of_colors, (float(*)[3])colors);
                delete colors;
            }
            else
            {
                print_comment(__LINE__, __FILE__, "ERROR: memory allocation failed");
                return NULL;
            }
        }
    }
    //
    // store new normalbinding
    //
    if (normalbinding == INV_PER_VERTEX)
        normbind->value = SoNormalBinding::PER_VERTEX_INDEXED;
    else if (normalbinding == INV_PER_FACE)
        normbind->value = SoNormalBinding::PER_FACE;
    else if (normalbinding == INV_NONE)
        normbind->value = SoNormalBinding::DEFAULT;
    if (normalbinding != INV_NONE)
    {
        //
        // store new normals
        //
        n = new float[no_of_normals * 3];
        if (n != NULL)
        {
            k = 0;
            for (j = 0; j < no_of_normals; j++)
            {
                *(n + k) = *(nx + j);
                *(n + k + 1) = *(ny + j);
                *(n + k + 2) = *(nz + j);
                k = k + 3;
            }
            normal->vector.deleteValues(0);
            normal->vector.setValues(0, no_of_normals, (float(*)[3])n);
            delete n;
        }
        else
            print_comment(__LINE__, __FILE__, "ERROR: memory allocation failed");
    }

    if (colorpacking != INV_RGBA)
        material->transparency.setValue(transparency);

    switch (vertexOrder)
    {
    case 0:
        shapehints->vertexOrdering = SoShapeHints::UNKNOWN_ORDERING;
        break;
    case 1:
        shapehints->vertexOrdering = SoShapeHints::CLOCKWISE;
        break;
    case 2:
        shapehints->vertexOrdering = SoShapeHints::COUNTERCLOCKWISE;
        break;
    }
    if (root)
        root->touch();
    print_comment(__LINE__, __FILE__, "polygon object replaced successfully");
    return root;
}

//======================================================================
// add TriangleStrip
//======================================================================
SoGroup *
om_addTriangleStripCB(const char *name, const char *rootName, int no_of_strips, int no_of_vertices,
                      int no_of_coords, float *x_c, float *y_c,
                      float *z_c, int *v_l, int *l_l, int no_of_colors,
                      int colorbinding, int colorpacking,
                      float *r, float *g, float *b, uint32_t *pc, int no_of_normals,
                      int normalbinding,
                      float *nx, float *ny, float *nz, float transparency,
                      int vertexOrder,
                      int texWidth, int texHeight, int pixelSize, unsigned char *image,
                      int no_of_texCoords, float *tx, float *ty, coMaterial *material, char *rName, char *label)
{

    //cerr << "om_addTriangleStripCB(..) name : " << name << " : " << rootName << "  " << label << endl;

    SoGroup *root;

    InvTriangleStrip *strip = new InvTriangleStrip(colorpacking);
    root = (SoGroup *)strip->getTopNode();
    strip->setCoords(no_of_strips, no_of_vertices,
                     no_of_coords, x_c, y_c, z_c,
                     v_l, l_l);

    if (rName)
        strip->setRealObjName(rName);
    if (label)
        strip->setGrpLabel(label);

    if (material)
    {
        strip->setMaterial(material);
    }
    strip->setColorBinding(colorbinding);

    strip->setNormalBinding(normalbinding);
    strip->setVertexOrdering(vertexOrder);
    if (normalbinding != INV_NONE)
        strip->setNormals(no_of_normals, nx, ny, nz);
    if (colorbinding != INV_NONE)
    {
        if (colorpacking == INV_RGBA)
        {
            if ((no_of_colors > 1) || (material == NULL))
                strip->setColors(no_of_colors, pc);
        }
        else if (colorpacking == INV_TEXTURE)
        {
            strip->setTexture(texWidth, texHeight, pixelSize, image);
            strip->setTexCoords(no_of_texCoords, tx, ty);
            strip->setTextureCoordinateBinding(INV_PER_VERTEX);
            rm->addToTextureList(strip->getTexture());
        }
        else
        {
            if ((no_of_colors > 1) || (material == NULL))
                strip->setColors(no_of_colors, r, g, b);
        }
    }

    strip->setTransparency(transparency);
    strip->setName(name);

    // create an inventor object and replace it with the current
    // scenegraph
    //

    if (rootName == NULL)
        rm->addToSceneGraph(root, name, NULL);
    else
        rm->addToSceneGraph(root, name, (om->list->search(rootName))->getObject());
    om_addObjectCB(name, root);

    print_comment(__LINE__, __FILE__, "strip object added successfully");
    return root;
}

//======================================================================
// replace polygon
//======================================================================
SoGroup *
om_replaceTriangleStripCB(const char *name, int no_of_strips, int no_of_vertices,
                          int no_of_coords, float *x_c, float *y_c,
                          float *z_c, int *v_l, int *l_l, int no_of_colors,
                          int colorbinding, int colorpacking,
                          float *r, float *g, float *b, uint32_t *pc, int no_of_normals,
                          int normalbinding,
                          float *nx, float *ny, float *nz, float transparency,
                          int vertexOrder,
                          int texWidth, int texHeight, int pixelSize, unsigned char *image,
                          int no_of_texCoords, float *tx, float *ty, coMaterial *comaterial)
{
    int numnodes, i, k = 0, j = 0;
    long no_strip;
    SoGroup *root = NULL;
    SoNode *node;
    SoGroup *sep;
    SoCoordinate3 *coord = NULL;
    SoMaterial *material = NULL;
    SoPackedColor *color = NULL;
    SoMaterialBinding *matbind = NULL;
    SoNormal *normal = NULL;
    SoNormalBinding *normbind = NULL;
    SoIndexedTriangleStripSet *strip = NULL;
    SoShapeHints *shapehints = NULL;
    SoTexture2 *texture = NULL;
    SoTextureCoordinate2 *texCoord = NULL;
    SoTextureCoordinateBinding *texbind = NULL;
    float *coord_points = NULL;
    float *colors, *n;
    long *vertices;
    float *tc = NULL; // texture coordinates
    InvObject *obj = om->list->search(name);
    (void)comaterial;
    if (obj != NULL)
    {
        root = obj->getObject();

        //	numnodes  = root->getNumChildren();
        //	cerr << endl;
        //	cerr << "-----------------------------------------" << endl;
        //	cerr << "I found the top node     :" << root->getName().getString() << endl;
        //	cerr << "This node is of type     :" << root->getClassTypeId().getName().getString() << endl;
        //	cerr << "The number of children is:" << numnodes << endl;
        //	cerr << "-----------------------------------------" << endl;
        //	cerr << endl;

        sep = (SoGroup *)root->getChild(0);
        numnodes = sep->getNumChildren();
        for (i = 0; i < numnodes; i++)
        {
            node = sep->getChild(i);
            if (node->isOfType(SoCoordinate3::getClassTypeId()))
            {
                coord = (SoCoordinate3 *)node;
                //cerr << "Child #" << i << " is a " << node->getClassTypeId().getName().getString() << endl;
            }
            else if (node->isOfType(SoMaterial::getClassTypeId()))
            {
                material = (SoMaterial *)node;
                //cerr << "Child #" << i << " is a " << node->getClassTypeId().getName().getString() << endl;
            }
            else if (node->isOfType(SoPackedColor::getClassTypeId()))
            {
                color = (SoPackedColor *)node;
            }
            else if (node->isOfType(SoMaterialBinding::getClassTypeId()))
            {
                matbind = (SoMaterialBinding *)node;
                //cerr << "Child #" << i << " is a " << node->getClassTypeId().getName().getString() << endl;
            }
            else if (node->isOfType(SoNormal::getClassTypeId()))
            {
                normal = (SoNormal *)node;
                //cerr << "Child #" << i << " is a " << node->getClassTypeId().getName().getString() << endl;
            }
            else if (node->isOfType(SoNormalBinding::getClassTypeId()))
            {
                normbind = (SoNormalBinding *)node;
                //cerr << "Child #" << i << " is a " << node->getClassTypeId().getName().getString() << endl;
            }
            else if (node->isOfType(SoTexture2::getClassTypeId()))
                texture = (SoTexture2 *)node;
            else if (node->isOfType(SoTextureCoordinate2::getClassTypeId()))
                texCoord = (SoTextureCoordinate2 *)node;
            else if (node->isOfType(SoTextureCoordinateBinding::getClassTypeId()))
                texbind = (SoTextureCoordinateBinding *)node;
            else if (node->isOfType(SoIndexedTriangleStripSet::getClassTypeId()))
            {
                strip = (SoIndexedTriangleStripSet *)node;
                //cerr << "Child #" << i << " is a " << node->getClassTypeId().getName().getString() << endl;
            }
            else if (node->isOfType(SoShapeHints::getClassTypeId()))
            {
                shapehints = (SoShapeHints *)node;
                //cerr << "Child #" << i << " is a " << node->getClassTypeId().getName().getString() << endl;
            }
            else
            {
                //cerr << "Child #" << i << " is a " << node->getClassTypeId().getName().getString() << endl;
            }
        }
        //cerr << "-----------------------------------------" << endl;
        //cerr << endl;
    }

    //
    // check what we got
    //
    if (coord == NULL || strip == NULL)
    {
        print_comment(__LINE__, __FILE__, "ERROR: Replaced set has different dimension");
        return NULL;
    }

    //
    // store new coordinates
    //
    coord_points = new float[no_of_coords * 3];
    vertices = new long[no_of_vertices + no_of_strips];

    // added () ()  AW
    if ((coord_points != NULL) && (vertices != NULL))
    {
        k = 0;
        for (j = 0; j < no_of_coords; j++)
        {
            *(coord_points + k) = *(x_c + j);
            *(coord_points + k + 1) = *(y_c + j);
            *(coord_points + k + 2) = *(z_c + j);
            k = k + 3;
        }
        coord->point.deleteValues(0);
        coord->point.setValues(0, no_of_coords, (float(*)[3])coord_points);

        //
        // store vertices list ( used by normal and coordinate field )
        //

        j = 0;
        for (no_strip = 0; no_strip < no_of_strips - 1; no_strip++)
        {
            for (k = *(l_l + no_strip); k < *(l_l + no_strip + 1); k++)
            {
                *(vertices + j) = *(v_l + k);
                j++;
            }
            *(vertices + j) = SO_END_STRIP_INDEX;
            j++;
        }
        // last strip
        for (k = *(l_l + no_of_strips - 1); k < no_of_vertices; k++)
        {
            *(vertices + j) = *(v_l + k);
            j++;
        }
        *(vertices + j) = SO_END_STRIP_INDEX;
        strip->coordIndex.deleteValues(0);
        strip->coordIndex.setValues(0, no_of_vertices + no_of_strips, (const int32_t *)vertices);

        delete coord_points;
        delete vertices;
    }
    else // added a { } around the next two lines, this was missing !! D.R.
    {
        print_comment(__LINE__, __FILE__, "ERROR: memory allocation failed");
        return NULL;
    }
    //
    // store new colorbinding
    //
    if (colorbinding == INV_PER_VERTEX)
        matbind->value = SoMaterialBinding::PER_VERTEX_INDEXED;
    else if (colorbinding == INV_PER_FACE)
        matbind->value = SoMaterialBinding::PER_FACE;
    else if (colorbinding == INV_NONE)
        matbind->value = SoMaterialBinding::OVERALL;
    else if (colorbinding == INV_OVERALL)
        matbind->value = SoMaterialBinding::OVERALL;
    if (colorbinding != INV_NONE)
    {
        if (colorpacking == INV_RGBA)
        {
            color->orderedRGBA.setValues(0, no_of_colors, (const uint32_t *)pc);
        }
        else if (colorpacking == INV_TEXTURE)
        {
            // new texture image
            texture->image.setValue(SbVec2s(texWidth, texHeight), pixelSize, image);
            texture->wrapS.setValue(SoTexture2::CLAMP);
            texture->wrapT.setValue(SoTexture2::CLAMP);
            texbind->value = SoTextureCoordinateBinding::DEFAULT;

            //
            // store new texture coordinates
            //
            tc = new float[no_of_texCoords * 2];
            if (tc != NULL)
            {
                k = 0;
                for (j = 0; j < no_of_texCoords; j++)
                {
                    *(tc + k) = *(tx + j);
                    *(tc + k + 1) = *(ty + j);
                    k = k + 2;
                }
                texCoord->point.setValues(0, no_of_texCoords, (float(*)[2])tc);
            }
            else
            {
                print_comment(__LINE__, __FILE__, "ERROR: memory allocation failed");
                return NULL;
            }
            delete tc;
        }
        else
        {
            //
            // store new colors
            //
            colors = new float[no_of_colors * 3];
            if (colors != NULL)
            {
                k = 0;
                for (int j = 0; j < no_of_colors; j++)
                {
                    *(colors + k) = *(r + j);
                    *(colors + k + 1) = *(g + j);
                    *(colors + k + 2) = *(b + j);
                    k = k + 3;
                }
                material->diffuseColor.deleteValues(0);
                material->diffuseColor.setValues(0, no_of_colors, (float(*)[3])colors);
                delete colors;
            }
            else
            {
                print_comment(__LINE__, __FILE__, "ERROR: memory allocation failed");
                return NULL;
            }
        }
    }
    //
    // store new normalbinding
    //
    if (normalbinding == INV_PER_VERTEX)
        normbind->value = SoNormalBinding::PER_VERTEX_INDEXED;
    else if (normalbinding == INV_PER_FACE)
        normbind->value = SoNormalBinding::PER_FACE;
    else if (normalbinding == INV_NONE)
        normbind->value = SoNormalBinding::DEFAULT;
    if (normalbinding != INV_NONE)
    {
        //
        // store new normals
        //
        n = new float[no_of_normals * 3];
        if (n != NULL)
        {
            k = 0;
            for (j = 0; j < no_of_normals; j++)
            {
                *(n + k) = *(nx + j);
                *(n + k + 1) = *(ny + j);
                *(n + k + 2) = *(nz + j);
                k = k + 3;
            }
            normal->vector.deleteValues(0);
            normal->vector.setValues(0, no_of_normals, (float(*)[3])n);
            delete n;
        }
        else
        {
            print_comment(__LINE__, __FILE__, "ERROR: memory allocation failed");
            return NULL;
        }
    }

    if (colorpacking != INV_RGBA)
        material->transparency.setValue(transparency);

    switch (vertexOrder)
    {
    case 0:
        shapehints->vertexOrdering = SoShapeHints::UNKNOWN_ORDERING;
        break;
    case 1:
        shapehints->vertexOrdering = SoShapeHints::CLOCKWISE;
        break;
    case 2:
        shapehints->vertexOrdering = SoShapeHints::COUNTERCLOCKWISE;
        break;
    }
    if (root)
        root->touch();
    print_comment(__LINE__, __FILE__, "TriangleStrip object replaced successfully");
    return root;
}

//======================================================================
// add line
//======================================================================
SoGroup *
om_addLineCB(const char *name, const char *rootName, int no_of_lines, int no_of_vertices,
             int no_of_coords, float *x_c, float *y_c,
             float *z_c, int *v_l, int *l_l, int no_of_colors,
             int colorbinding, int colorpacking,
             float *r, float *g, float *b, uint32_t *pc, int no_of_normals,
             int normalbinding,
             float *nx, float *ny, float *nz, coMaterial *material,
             char *rName, char *label)
{
    SoGroup *root;

    if (no_of_vertices <= no_of_lines)
    {
        root = new SoGroup;
    }
    else
    {
        InvLine *line = new InvLine(colorpacking);
        root = (SoGroup *)line->getTopNode();

        line->setCoords(no_of_lines, no_of_vertices,
                        no_of_coords, x_c, y_c, z_c,
                        v_l, l_l);

        if (rName)
            line->setRealObjName(rName);
        if (label)
            line->setGrpLabel(label);

        if (material)
        {
            line->setMaterial(material);
        }
        line->setColorBinding(colorbinding);

        if (colorbinding != INV_NONE)
        {
            if (colorpacking == INV_RGBA)
            {
                if ((no_of_colors > 1) || (material == NULL))
                {
                    line->setColors(no_of_colors, pc);
                }
            }
            else
            {
                if ((no_of_colors > 1) || (material == NULL))
                {
                    line->setColors(no_of_colors, r, g, b);
                }
            }
        }
        line->setNormalBinding(normalbinding);

        if (normalbinding != INV_NONE)
            line->setNormals(no_of_normals, nx, ny, nz);

        line->setName(name);
    }
    //
    // create an inventor object and replace it with the current
    // scenegraph
    //
    if (rootName == NULL)
        rm->addToSceneGraph(root, name, NULL);
    else
        rm->addToSceneGraph(root, name, (om->list->search(rootName))->getObject());
    om_addObjectCB(name, root);

    print_comment(__LINE__, __FILE__, "line object added successfully");
    return root;
}

//======================================================================
// replace line
//======================================================================
SoGroup *
om_replaceLineCB(const char *name, int no_of_lines, int no_of_vertices,
                 int no_of_coords, float *x_c, float *y_c,
                 float *z_c, int *v_l, int *l_l, int no_of_colors,
                 int colorbinding, int colorpacking,
                 float *r, float *g, float *b, uint32_t *pc, int no_of_normals,
                 int normalbinding,
                 float *nx, float *ny, float *nz)
{
    int numnodes, i, k = 0, j = 0;
    long no_lines;
    SoGroup *root = NULL;
    SoNode *node;
    SoGroup *sep;
    SoPackedColor *color = NULL;
    SoCoordinate3 *coord = NULL;
    SoMaterial *material = NULL;
    SoMaterialBinding *matbind = NULL;
    SoNormal *normal = NULL;
    SoNormalBinding *normbind = NULL;
    SoIndexedLineSet *lines = NULL;
    float *coord_points = NULL;
    float *colors, *n;
    long *vertices;
    InvObject *obj = om->list->search(name);
    if (obj != NULL)
    {
        root = obj->getObject();
        sep = (SoGroup *)root->getChild(0);
        numnodes = sep->getNumChildren();
        for (i = 0; i < numnodes; i++)
        {
            node = sep->getChild(i);
            if (node->isOfType(SoCoordinate3::getClassTypeId()))
                coord = (SoCoordinate3 *)node;
            if (node->isOfType(SoMaterial::getClassTypeId()))
                material = (SoMaterial *)node;
            if (node->isOfType(SoPackedColor::getClassTypeId()))
                color = (SoPackedColor *)node;
            if (node->isOfType(SoMaterialBinding::getClassTypeId()))
                matbind = (SoMaterialBinding *)node;
            if (node->isOfType(SoNormal::getClassTypeId()))
                normal = (SoNormal *)node;
            if (node->isOfType(SoNormalBinding::getClassTypeId()))
                normbind = (SoNormalBinding *)node;
            if (node->isOfType(SoIndexedLineSet::getClassTypeId()))
                lines = (SoIndexedLineSet *)node;
        }
    }

    //
    // check what we got
    //
    if (coord == NULL || lines == NULL)
    {
        print_comment(__LINE__, __FILE__, "ERROR: Replaced set has different dimension");
        return NULL;
    }

    //
    // store new coordinates
    //
    coord_points = new float[no_of_coords * 3];
    vertices = new long[no_of_vertices + no_of_lines];

    if (coord_points != NULL && vertices != NULL)
    {
        k = 0;
        for (j = 0; j < no_of_coords; j++)
        {
            *(coord_points + k) = *(x_c + j);
            *(coord_points + k + 1) = *(y_c + j);
            *(coord_points + k + 2) = *(z_c + j);
            k = k + 3;
        }
        coord->point.deleteValues(0);
        coord->point.setValues(0, no_of_coords, (float(*)[3])coord_points);

        //
        // store vertices list ( used by normal and coordinate field )
        //
        j = 0;
        for (no_lines = 0; no_lines < no_of_lines - 1; no_lines++)
        {
            for (k = *(l_l + no_lines); k < *(l_l + no_lines + 1); k++)
            {
                *(vertices + j) = *(v_l + k);
                j++;
            }
            *(vertices + j) = SO_END_LINE_INDEX;
            j++;
        }
        // last line
        for (k = *(l_l + no_of_lines - 1); k < no_of_vertices; k++)
        {
            *(vertices + j) = *(v_l + k);
            j++;
        }
        *(vertices + j) = SO_END_LINE_INDEX;

        lines->coordIndex.deleteValues(0);
        lines->coordIndex.setValues(0, no_of_vertices + no_of_lines, (const int32_t *)vertices);

        delete coord_points;
        delete vertices;
    }
    else
        print_comment(__LINE__, __FILE__, "ERROR: memory allocation failed");
    //
    // store new colorbinding
    //
    if (colorbinding == INV_PER_VERTEX)
        matbind->value = SoMaterialBinding::PER_VERTEX_INDEXED;
    else if (colorbinding == INV_PER_FACE)
        matbind->value = SoMaterialBinding::PER_FACE;
    else if (colorbinding == INV_NONE)
        matbind->value = SoMaterialBinding::OVERALL;
    else if (colorbinding == INV_OVERALL)
        matbind->value = SoMaterialBinding::OVERALL;
    if (colorbinding != INV_NONE)
    {
        if (colorpacking == INV_RGBA)
        {
            if (color)
                color->orderedRGBA.setValues(0, no_of_colors, (const uint32_t *)pc);
        }
        else
        {
            //
            // store new colors
            //
            colors = new float[no_of_colors * 3];
            if (colors != NULL)
            {
                k = 0;
                for (int j = 0; j < no_of_colors; j++)
                {
                    *(colors + k) = *(r + j);
                    *(colors + k + 1) = *(g + j);
                    *(colors + k + 2) = *(b + j);
                    k = k + 3;
                }
                material->diffuseColor.deleteValues(0);
                material->diffuseColor.setValues(0, no_of_colors, (float(*)[3])colors);
                delete colors;
            }
            else
            {
                print_comment(__LINE__, __FILE__, "ERROR: memory allocation failed");
                return NULL;
            }
        }
    }
    //
    // store new normalbinding
    //
    if (normalbinding == INV_PER_VERTEX)
        normbind->value = SoNormalBinding::PER_VERTEX_INDEXED;
    else if (normalbinding == INV_PER_FACE)
        normbind->value = SoNormalBinding::PER_FACE;
    else if (normalbinding == INV_NONE)
        normbind->value = SoNormalBinding::DEFAULT;
    if (normalbinding != INV_NONE)
    {
        //
        // store new normals
        //
        n = new float[no_of_normals * 3];
        if (n != NULL)
        {
            k = 0;
            for (j = 0; j < no_of_normals; j++)
            {
                *(n + k) = *(nx + j);
                *(n + k + 1) = *(ny + j);
                *(n + k + 2) = *(nz + j);
                k = k + 3;
            }
            normal->vector.deleteValues(0);
            normal->vector.setValues(0, no_of_normals, (float(*)[3])n);
            delete n;
        }
        else
            print_comment(__LINE__, __FILE__, "ERROR: memory allocation failed");
    }
    print_comment(__LINE__, __FILE__, "line object replaced successfully");
    return root;
}

//======================================================================
// add points
//======================================================================
SoGroup *om_addPointCB(const char *name, const char *rootName, int no_of_points,
                       float *x_c, float *y_c,
                       float *z_c, int numColors, int colorbinding, int colorpacking,
                       float *r, float *g, float *b, uint32_t *packedColor,
                       float pointsize)
{
    SoGroup *root;

    InvPoint *point = new InvPoint(colorpacking);
    root = (SoGroup *)point->getTopNode();
    point->setCoords(no_of_points, x_c, y_c, z_c);
    point->setColorBinding(colorbinding);

    if (colorbinding != INV_NONE)
    {
        if (colorpacking == INV_RGBA)
            point->setColors(numColors, packedColor);
        else
            point->setColors(numColors, r, g, b);
    }
    point->setName(name);
    if (pointsize > 0)
        point->setSize(pointsize);
    //
    // create an inventor object and replace it with the current
    // scenegraph
    //
    if (rootName == NULL)
        rm->addToSceneGraph(root, name, NULL);
    else
        rm->addToSceneGraph(root, name, (om->list->search(rootName))->getObject());
    om_addObjectCB(name, root);

    print_comment(__LINE__, __FILE__, "point object added successfully");
    return root;
}

//======================================================================
// replace points
//======================================================================
SoGroup *om_replacePointCB(const char *name, int no_of_points,
                           float *x_c, float *y_c,
                           float *z_c, int colorbinding, int colorpacking,
                           float *r, float *g, float *b, uint32_t *pc)
{
    int numnodes, i, k = 0;
    SoGroup *root = NULL;
    SoNode *node;
    SoGroup *sep;
    SoPackedColor *color = NULL;
    SoCoordinate3 *coord = NULL;
    SoMaterial *material = NULL;
    float *coord_points = NULL;
    float *colors;
    InvObject *obj = om->list->search(name);
    if (obj != NULL)
    {
        root = obj->getObject();
        sep = (SoGroup *)root->getChild(0);
        numnodes = sep->getNumChildren();
        for (i = 0; ((i < numnodes) && ((coord == NULL) || (material == NULL))); i++)
        {
            node = sep->getChild(i);
            if (node->isOfType(SoCoordinate3::getClassTypeId()))
                coord = (SoCoordinate3 *)node;
            if (node->isOfType(SoPackedColor::getClassTypeId()))
                color = (SoPackedColor *)node;
            if (node->isOfType(SoMaterial::getClassTypeId()))
                material = (SoMaterial *)node;
        }
    }

    //
    // check what we got
    //
    if (coord == NULL)
    {
        print_comment(__LINE__, __FILE__, "ERROR: Replaced set has different dimension");
        return NULL;
    }

    //
    // store new coordinates
    //
    coord_points = new float[no_of_points * 3];
    if (coord_points != NULL)
    {
        k = 0;
        for (int j = 0; j < no_of_points; j++)
        {
            *(coord_points + k) = *(x_c + j);
            *(coord_points + k + 1) = *(y_c + j);
            *(coord_points + k + 2) = *(z_c + j);
            k = k + 3;
        }
        coord->point.deleteValues(0);
        coord->point.setValues(0, no_of_points, (float(*)[3])coord_points);
        delete coord_points;
    }
    else
        print_comment(__LINE__, __FILE__, "ERROR: memory allocation failed");
    if (colorbinding != INV_NONE)
    {
        if (colorpacking == INV_RGBA)
        {
            if (color)
                color->orderedRGBA.setValues(0, no_of_points, (const uint32_t *)pc);
        }
        else
        {
            //
            // store new colors
            //
            colors = new float[no_of_points * 3];
            if (colors != NULL)
            {
                k = 0;
                for (int j = 0; j < no_of_points; j++)
                {
                    *(colors + k) = *(r + j);
                    *(colors + k + 1) = *(g + j);
                    *(colors + k + 2) = *(b + j);
                    k = k + 3;
                }
                material->diffuseColor.deleteValues(0);
                material->diffuseColor.setValues(0, no_of_points, (float(*)[3])colors);
                delete colors;
            }
            else
            {
                print_comment(__LINE__, __FILE__, "ERROR: memory allocation failed");
                return NULL;
            }
        }
    }
    print_comment(__LINE__, __FILE__, "point object replaced successfully");
    return root;
}

//======================================================================
// add Inventor Scene Graph description
//======================================================================
void om_addIvCB(const char *object, const char *rootName, char *IvDescription, int size)
{
    char Buffer[255];
    SoInput in;
    SoLabel *objName;

    in.setBuffer(IvDescription, size);
    // SoNode *ivData;
    SoSwitch *top_switch;
    SoSeparator *root;
    SoTransform *transform;

    // SoDB::read(&in,ivData);
    top_switch = new SoSwitch;
    root = new SoSeparator;
    objName = new SoLabel;
    transform = new SoTransform;
    top_switch->whichChild.setValue(0);
    top_switch->addChild(root);
    root->addChild(transform);
    // root->addChild(ivData);
    root->addChild(SoDB::readAll(&in));
    root->addChild(objName);
    objName->label.setValue(object);
    strcpy(Buffer, "S_");
    strcat(Buffer, object);
    top_switch->setName(SbName(Buffer));

    if (rootName == NULL)
        rm->addToSceneGraph(top_switch, object, NULL);
    else
        rm->addToSceneGraph(top_switch, object, (om->list->search(rootName))->getObject());
    om_addObjectCB(object, top_switch);
}

//======================================================================
// timestep changed callvback
//======================================================================
void om_timestepCB(void *userData, void *callbackData)
{
    char buf[255];
    int val, state, old_state;
    (void)callbackData;
    InvSequencer *seq = (InvSequencer *)userData;

    val = seq->getValue();
    state = seq->getSeqState();

    // send new values to the slave renderers
    // all switches are synchronized to the same timestep, so sending one is sufficient

    if (rm_isSynced() != SYNC_LOOSE)
    {
        old_state = seq->getOldState();
        if ((state != old_state)
            || ((state != PLAYBACKWARD)
                && (state != PLAYFORWARD)))
        {
            sprintf(buf, "%d %d %d %d %d %d %d", val, seq->getMinimum(), seq->getMaximum(), seq->getSliderMinimum(), seq->getSliderMaximum(), seq->getSeqAct(), state);
            rm_sendSequencer(buf);
        }
    }

    // changes to multiple sets: AW
    // timestepSwitch->whichChild.setValue(2+seq->getValue()-seq->getMinimum());

    InvObjectManager::timestepSwitchList.reset();
    SoSwitch *swit;
    while ((swit = InvObjectManager::timestepSwitchList.next()) != NULL)
    {
        swit->whichChild.setValue(2 + val - seq->getMinimum());
    }
}

//======================================================================
// sequencer receive info
//======================================================================
void
InvObjectManager::receiveSequencer(const char *message)
{
    int val, min, max, boundmin, boundmax, active, state;
    SoSwitch *swit;

    int ret = sscanf(message, "%d %d %d %d %d %d %d", &val, &min, &max, &boundmin, &boundmax,
                     &active, &state);
    if (ret != 7)
    {
        fprintf(stderr, "InvObjectManager::receiveSequencer: sscanf failed\n");
    }

    //cerr << " InvObjectManager::receiveSequencer(..) called " << endl;

    // reset list
    InvObjectManager::timestepSwitchList.reset();
    while ((swit = InvObjectManager::timestepSwitchList.next()) != NULL)
    {
        swit->whichChild.setValue(2 + val - min);
    }

    if (timeStepper != NULL)
    {
        timeStepper->stop(); //implicit STOP
        timeStepper->setSliderBounds(min, max, val, boundmin, boundmax);
        timeStepper->setSeqState(state);
        timeStepper->setSeqAct(active);
        if ((state == PLAYFORWARD) || (state == PLAYBACKWARD))
        {
            timeStepper->play(state);
        }
    }
}

//======================================================================
// add a Separator
//======================================================================
void om_addSeparatorCB(const char *object, const char *rootName, int is_timestep, int min, int max)
{
    //static InvSequencer *timeStepper = NULL; // moved to class AW
    SoLabel *objName;
    SoGroup *root;
    SoTransform *transform;

    if (is_timestep)
    {

        SoSwitch *swit;
        // create the switch node  -> list AW
        // root = timestepSwitch =  new SoSwitch;
        root = swit = new SoSwitch;
        SoSwitch *firstSwitch = InvObjectManager::timestepSwitchList.get_first();
        InvObjectManager::timestepSwitchList.add(swit);

        if (firstSwitch)
        {
            swit->whichChild.setValue(firstSwitch->whichChild.getValue());
        }
        else
        {
            swit->whichChild.setValue(2);
        }
        swit->setName("TIMESTEP_SWITCH");
    }
    else
        root = new SoSeparator;

    objName = new SoLabel;
    transform = new SoTransform;

    root->addChild(objName);
    root->addChild(transform);
    objName->label.setValue(object);

    if (is_timestep)
    {

        // if not yet existing create sequencer widget
        if (InvObjectManager::timeStepper == NULL)
        {
            if (coviseViewer)
            {
                InvObjectManager::timeStepper = new InvSequencer(coviseViewer->getSeqParent());
            }
            else
            {
                InvObjectManager::timeStepper = new InvSequencer();
            }
            InvObjectManager::timeStepper->setValueChangedCallback(om_timestepCB, (void *)InvObjectManager::timeStepper);
            InvObjectManager::timeStepper->setSliderBounds(min, max, min, min, max);
            InvObjectManager::timeStepper->show();
        }
        else
        {
            InvObjectManager::timeStepper->setSliderBounds(min, max, min, min, max);
        }

        if (rm_isMaster())
            InvObjectManager::timeStepper->setActive();
        else // slave
        {
            if (rm_isSynced() == SYNC_LOOSE)
            {
                InvObjectManager::timeStepper->setActive();
            }
            else
            {
                InvObjectManager::timeStepper->setInactive();
            }
        }
    }

    if (rootName == NULL)
        rm->addToSceneGraph(root, object, NULL);
    else
        rm->addToSceneGraph(root, object, (om->list->search(rootName))->getObject());
    om_addObjectCB(object, root);
}

//======================================================================
// replace Inventor Scene Graph description
//======================================================================
void om_replaceIvCB(const char *object, char *IvDescription, int size)
{
    om_deleteObjectCB(object);
    om_addIvCB(object, NULL, IvDescription, size);
}

//======================================================================
// add a Colormap
//======================================================================
void om_addColormapCB(const char *object, const char *colormap)
{
    rm->addColormap(object, colormap);
}

//======================================================================
// replace Colormap
//======================================================================
void om_replaceColormapCB(const char *object, const char *colormap)
{
    // om_deleteColormapCB(object);
    // om_addColormapCB(object,colormap);
    rm->replaceColormap(object, colormap);
}

//======================================================================
// delete a Colormap
//======================================================================
void om_deleteColormapCB(const char *object)
{
    if (NULL != object)
    {
        rm->deleteColormap(object);
    }
}

//======================================================================
// add a Part
//======================================================================
void om_addPartCB(const char *object, int part_id, SoSwitch *s)
{
    rm->addPart(object, part_id, s);
}

//======================================================================
// replace a Part
//======================================================================
void om_replacePartCB(const char *object, int part_id, SoSwitch *s)
{
    rm->replacePart(object, part_id, s);
}

//======================================================================
// delete a Part
//======================================================================
void om_deletePartCB(const char *object)
{
    if (NULL != object)
    {
        rm->deletePart(object);
    }
}

//======================================================================
// add a Time-Part
//======================================================================
void om_addTimePartCB(const char *object, int timeStep, int part_id, SoSwitch *s)
{
    rm->addTimePart(object, timeStep, part_id, s);
}

//======================================================================
// replace a Time-Part
//======================================================================
void om_replaceTimePartCB(const char *object, int timeStep, int part_id, SoSwitch *s)
{
    rm->replaceTimePart(object, timeStep, part_id, s);
}

//======================================================================
// delete a Time-Part
//======================================================================
void om_deleteTimePartCB(const char *object)
{
    if (NULL != object)
    {
        rm->deleteTimePart(object);
    }
}

//======================================================================
// add object to the object list
//======================================================================
void om_addObjectCB(const char *name, SoGroup *ptr)
{
    InvObject *obj = new InvObject(name, ptr);

    om->list->add(obj);

    if ((InvObjectManager::timeStepper)
            // no timesteps any more
        && (InvObjectManager::timestepSwitchList.get_last() != NULL))
    {
        InvObjectManager::timeStepper->activate();
        InvObjectManager::timeStepper->noSeq = 1;
        InvObjectManager::timeStepper->show();
        InvObjectManager::timeStepper->stop();
    }

    //DBG  cerr << "RENDERER: Current Object List :" << endl;
    //DBG  om->list->print();
    //DBG  cerr << "RENDERER: new object added" << endl;
}

//======================================================================
// delete object from object list
//======================================================================
void
om_deleteObjectCB(const char *name)
{
    if (NULL == name)
    {
        return;
    }
    SoGroup *deleteRoot;
    SoSearchAction saNode;
    SoTexture2 *tex;
    SoPath *path;
    // InvObject *listobject;
    // int i;

    // add TimeSwitch deleting

    InvObject *obj = om->list->search(name);

    saNode.setFind(SoSearchAction::TYPE);
    saNode.setType(SoTexture2::getClassTypeId());

    if (obj != NULL)
    {
        deleteRoot = obj->getObject();

        // if this is a switch, erase it from the list   AW
        if (deleteRoot)
        {
            //cerr << "om_deleteObjectCB(char *name) try to delete <" << name << "> " << endl;
            if (deleteRoot->isOfType(SoSwitch::getClassTypeId()))
            {
                SoSwitch *delSwitch = (SoSwitch *)deleteRoot;
                InvObjectManager::timestepSwitchList.reset();
                SoSwitch *swit = InvObjectManager::timestepSwitchList.next();
                while (swit && (swit != delSwitch))
                    swit = InvObjectManager::timestepSwitchList.next();
                if (swit)
                {
                    InvObjectManager::timestepSwitchList.remove(delSwitch);
                    print_comment(__LINE__, __FILE__, "deleted timstep reference");
                }
                else
                    print_comment(__LINE__, __FILE__, "timstep delete FAILED");

                if ((InvObjectManager::timeStepper)
                        // no timesteps any more
                    && (InvObjectManager::timestepSwitchList.get_last() == NULL))
                {
                    InvObjectManager::timeStepper->stop();
                    InvObjectManager::timeStepper->hide();
                    InvObjectManager::timeStepper->noSeq = 0;
                    //InvObjectManager::timeStepper->setInactive();
                    //delete InvObjectManager::timeStepper;
                    //InvObjectManager::timeStepper=NULL;
                }
            }
        }
        saNode.apply(deleteRoot);
        path = saNode.getPath();
        if (path)
        {
            tex = (SoTexture2 *)path->getTail();
            rm->removeFromTextureList(tex);
        }

        om->list->remove(obj);
        rm->removeFromSceneGraph(deleteRoot, name);
        print_comment(__LINE__, __FILE__, "object deleted successfully");
    }

#ifndef TOLERANT
    print_comment(__LINE__, __FILE__, "found no object to delete");
#endif
}

//======================================================================
// constructor
//======================================================================
InvObjectManager::InvObjectManager()
{
    list = new InvObjectList();
    timeStepper = NULL;
}

//======================================================================
// create the inventor structure from the current object list
//======================================================================
SoNode *InvObjectManager::createInventorGraph()
{
    if (root != NULL)
    {
        root = new SoSeparator;
        //DBG cerr << "RENDERER: new om created" << endl;
    }
    else
        root = new SoSeparator;

    return root;
}

//======================================================================
// destructor
//======================================================================
InvObjectManager::~InvObjectManager()
{
    delete list;
}

//======================================================================
//
//======================================================================
void setMeshCoords(SoCoordinate3 *coord, SoQuadMesh *quadmesh, int VerticesPerRow, int VerticesPerColumn,
                   float *x_c, float *y_c, float *z_c)
{
    int num_vec;
    long j, k;
    float *vertices;

    //
    // store coordinates
    //
    num_vec = VerticesPerRow * VerticesPerColumn;
    vertices = new float[num_vec * 3];

    if (vertices != NULL)
    {
        k = 0;
        for (j = 0; j < num_vec; j++)
        {
            *(vertices + k) = *(x_c + j);
            *(vertices + k + 1) = *(y_c + j);
            *(vertices + k + 2) = *(z_c + j);
            k += 3;
        }
        coord->point.deleteValues(0);
        coord->point.setValues(0, num_vec, (float(*)[3])vertices);
        quadmesh->verticesPerRow = VerticesPerRow;
        quadmesh->verticesPerColumn = VerticesPerColumn;

        delete vertices;
    }
    else
        print_comment(__LINE__, __FILE__, "ERROR: memory allocation failed");
}
