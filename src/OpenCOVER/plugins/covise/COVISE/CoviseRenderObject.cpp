/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include <boost/lexical_cast.hpp>
#include <boost/static_assert.hpp>

#include "CoviseRenderObject.h"
#include <cover/coVRMSController.h>
#include <PluginUtil/coSphere.h>
#include <net/message.h>
#include <net/tokenbuffer.h>
#include <do/coDistributedObject.h>
#include <do/coDoPoints.h>
#include <do/coDoSpheres.h>
#include <do/coDoLines.h>
#include <do/coDoPolygons.h>
#include <do/coDoTriangleStrips.h>
#include <do/coDoData.h>
#include <do/coDoIntArr.h>
#include <do/coDoPixelImage.h>
#include <do/coDoSet.h>
#include <do/coDoText.h>
#include <do/coDoTexture.h>
#include <do/coDoUnstructuredGrid.h>
#include <do/coDoStructuredGrid.h>
#include <do/coDoRectilinearGrid.h>
#include <do/coDoUniformGrid.h>
#include <do/coDoGeometry.h>

#include "coVRDistributionManager.h"
#include <cover/coVRPluginSupport.h>

using namespace opencover;
using namespace covise;
#define addInt(x) tb.addBinary((char *)&(x), sizeof(int));
#define addFloat(x) tb.addBinary((char *)&(x), sizeof(float));
#define copyInt(x) memcpy(&x, tb.getBinary(sizeof(int)), sizeof(int));
#define copyFloat(x) memcpy(&x, tb.getBinary(sizeof(float)), sizeof(float));
#define HASOBJ 1
#define HASNORMALS 2
#define HASCOLORS 4
#define HASTEXTURE 8
#define HASVERTEXATTRIBUTE 16
#define HASMULTICOLORS 32

CoviseRenderObject::CoviseRenderObject(const coDistributedObject *co, const std::vector<int> &assignedTo)
    : assignedTo(assignedTo)
{
    int i;
    type[0] = '\0';
    texture = NULL;
    textureCoords = NULL;
    numTC = 0;
    numAttributes = 0;
    attrNames = NULL;
    attributes = NULL;
    size = 0;
    sizeu = 0;
    sizev = 0;
    sizew = 0;
    for (int c = 0; c < Field::NumChannels; ++c)
    {
        barr[c] = NULL;
        iarr[c] = NULL;
        farr[c] = NULL;
        min_[c] = 0;
        max_[c] = 0;
    }
    COVdobj = NULL;
    COVnormals = NULL;
    for (int c = 0; c < Field::NumChannels; ++c)
        COVcolors[c] = NULL;
    COVtexture = NULL;
    COVvertexAttribute = NULL;
    name = NULL;
    objs = NULL;
    geometryObject = NULL;
    normalObject = NULL;
    colorObject = NULL;
    textureObject = NULL;
    vertexAttributeObject = NULL;
    geometryFlag = 0;
    pc = NULL;
    coviseObject = co;
    cluster = coVRMSController::instance()->isCluster();

    if (cluster && coVRDistributionManager::instance().isActive() && this->assignedTo.empty())
    {
        this->assignedTo = coVRDistributionManager::instance().assign(co);
    }

    if (coVRMSController::instance()->isMaster())
    {
        TokenBuffer tb;
        if (co)
        {

            memcpy(type, co->getType(), 7);
            name = new char[strlen(co->getName()) + 1];
            strcpy(name, co->getName());
            numAttributes = co->getNumAttributes();
            attrNames = new char *[numAttributes];
            attributes = new char *[numAttributes];
            const char **attrs = NULL, **anames = NULL;
            co->getAllAttributes(&anames, &attrs);
            for (int i = 0; i < numAttributes; i++)
            {
                attrNames[i] = new char[strlen(anames[i]) + 1];
                strcpy(attrNames[i], anames[i]);
                attributes[i] = new char[strlen(attrs[i]) + 1];
                strcpy(attributes[i], attrs[i]);
            }
            if (cluster)
            {
                tb.addBinary(type, 7);
                tb << name;
                addInt(numAttributes);
                for (int i = 0; i < numAttributes; i++)
                {
                    tb << attrNames[i];
                    tb << attributes[i];
                }
            }

            //cerr << "Sending Object " << name << " numAttribs: " << numAttributes << endl;
            //cerr << "type " << type << endl;
            if (strcmp(type, "GEOMET") == 0)
            {
                coDoGeometry *geometry = (coDoGeometry *)co;
                COVdobj = geometry->getGeometry();
                COVnormals = geometry->getNormals();
                for (int c = 0; c < Field::NumChannels; ++c)
                    COVcolors[c] = geometry->getColors(c);
                COVtexture = geometry->getTexture();
                COVvertexAttribute = geometry->getVertexAttribute();
                if (COVdobj)
                    geometryFlag |= HASOBJ;
                if (COVnormals)
                    geometryFlag |= HASNORMALS;
                for (int c = 0; c < Field::NumChannels; ++c)
                {
                    if (COVcolors[c])
                    {
                        geometryFlag |= HASCOLORS;
                        if (c > 0)
                        {
                            geometryFlag |= HASMULTICOLORS;
                        }
                    }
                }
                if (COVtexture)
                    geometryFlag |= HASTEXTURE;
                if (COVvertexAttribute)
                    geometryFlag |= HASVERTEXATTRIBUTE;
                if (cluster)
                {
                    addInt(geometryFlag);
                }
            }
            else if (strcmp(type, "SETELE") == 0)
            {
                coDoSet *set = (coDoSet *)co;
                size = set->getNumElements();
                if (cluster)
                {
                    addInt(size);
                }
            }
            else if (strcmp(type, "POLYGN") == 0)
            {
                coDoPolygons *poly = (coDoPolygons *)co;
                sizeu = poly->getNumPolygons();
                sizev = poly->getNumVertices();
                size = poly->getNumPoints();
                poly->getAddresses(&farr[0], &farr[1], &farr[2], &iarr[0], &iarr[1]);
                if (cluster)
                {
                    addInt(sizeu);
                    addInt(sizev);
                    addInt(size);
                    tb.addBinary((char *)farr[0], size * sizeof(float));
                    tb.addBinary((char *)farr[1], size * sizeof(float));
                    tb.addBinary((char *)farr[2], size * sizeof(float));
                    tb.addBinary((char *)iarr[0], sizev * sizeof(int));
                    tb.addBinary((char *)iarr[1], sizeu * sizeof(int));
                }
            }
            else if (strcmp(type, "TRIANG") == 0)
            {
                coDoTriangleStrips *strip = (coDoTriangleStrips *)co;
                sizeu = strip->getNumStrips();
                sizev = strip->getNumVertices();
                size = strip->getNumPoints();
                strip->getAddresses(&farr[0], &farr[1], &farr[2], &iarr[0], &iarr[1]);
                if (cluster)
                {
                    addInt(sizeu);
                    addInt(sizev);
                    addInt(size);
                    tb.addBinary((char *)farr[0], size * sizeof(float));
                    tb.addBinary((char *)farr[1], size * sizeof(float));
                    tb.addBinary((char *)farr[2], size * sizeof(float));
                    tb.addBinary((char *)iarr[0], sizev * sizeof(int));
                    tb.addBinary((char *)iarr[1], sizeu * sizeof(int));
                }
            }
            else if (strcmp(type, "UNIGRD") == 0)
            {
                coDoUniformGrid *ugrid = (coDoUniformGrid *)co;
                ugrid->getGridSize(&sizeu, &sizev, &sizew);
                ugrid->getMinMax(&min_[0], &max_[0], &min_[1], &max_[1], &min_[2], &max_[2]);
                if (cluster)
                {
                    addInt(sizeu);
                    addInt(sizev);
                    addInt(sizew);
                    addFloat(min_[0]);
                    addFloat(max_[0]);
                    addFloat(min_[1]);
                    addFloat(max_[1]);
                    addFloat(min_[2]);
                    addFloat(max_[2]);
                }
            }
            else if (strcmp(type, "UNSGRD") == 0)
            {
                coDoUnstructuredGrid *unsgrid = (coDoUnstructuredGrid *)co;
                unsgrid->getGridSize(&sizeu, &sizev, &sizew);
                unsgrid->getAddresses(&iarr[1], &iarr[0], &farr[0], &farr[1], &farr[2]);
                unsgrid->getTypeList(&iarr[2]);
                if (cluster)
                {
                    addInt(sizeu);
                    addInt(sizev);
                    addInt(sizew);
                    tb.addBinary((char *)farr[0], sizew * sizeof(float));
                    tb.addBinary((char *)farr[1], sizew * sizeof(float));
                    tb.addBinary((char *)farr[2], sizew * sizeof(float));
                    tb.addBinary((char *)iarr[0], sizev * sizeof(int));
                    tb.addBinary((char *)iarr[1], sizeu * sizeof(int));
                    tb.addBinary((char *)iarr[2], sizeu * sizeof(int));
                }
            }
            else if (strcmp(type, "RCTGRD") == 0)
            {
                coDoRectilinearGrid *rgrid = (coDoRectilinearGrid *)co;
                rgrid->getGridSize(&sizeu, &sizev, &sizew);
                rgrid->getAddresses(&farr[0], &farr[1], &farr[2]);
                if (cluster)
                {
                    addInt(sizeu);
                    addInt(sizev);
                    addInt(sizew);
                    tb.addBinary((char *)farr[0], sizeu * sizeof(float));
                    tb.addBinary((char *)farr[1], sizev * sizeof(float));
                    tb.addBinary((char *)farr[2], sizew * sizeof(float));
                }
            }
            else if (strcmp(type, "STRGRD") == 0)
            {
                coDoStructuredGrid *sgrid = (coDoStructuredGrid *)co;
                sgrid->getGridSize(&sizeu, &sizev, &sizew);
                sgrid->getAddresses(&farr[0], &farr[1], &farr[2]);
                if (cluster)
                {
                    addInt(sizeu);
                    addInt(sizev);
                    addInt(sizew);
                    tb.addBinary((char *)farr[0], sizeu * sizev * sizew * sizeof(float));
                    tb.addBinary((char *)farr[1], sizeu * sizev * sizew * sizeof(float));
                    tb.addBinary((char *)farr[2], sizeu * sizev * sizew * sizeof(float));
                }
            }
            else if (strcmp(type, "POINTS") == 0)
            {
                coDoPoints *points = (coDoPoints *)co;
                size = points->getNumPoints();
                points->getAddresses(&farr[0], &farr[1], &farr[2]);
                if (cluster)
                {
                    addInt(size);
                    tb.addBinary((char *)farr[0], size * sizeof(float));
                    tb.addBinary((char *)farr[1], size * sizeof(float));
                    tb.addBinary((char *)farr[2], size * sizeof(float));
                }
            }
            else if (strcmp(type, "SPHERE") == 0)
            {
                coDoSpheres *spheres = (coDoSpheres *)co;
                size = spheres->getNumSpheres();
                geometryFlag = coSphere::RENDER_METHOD_CPU_BILLBOARDS;
                const char *rm = spheres->getAttribute("RENDER_METHOD");
                if (rm)
                {
                    if (!strcmp(rm, "CG_SHADER"))
                        geometryFlag = coSphere::RENDER_METHOD_CG_SHADER;
                    else if (!strcmp(rm, "POINT_SPRITES"))
                        geometryFlag = coSphere::RENDER_METHOD_ARB_POINT_SPRITES;
                    else if (!strcmp(rm, "PARTICLE_CLOUD"))
                        geometryFlag = coSphere::RENDER_METHOD_PARTICLE_CLOUD;
                    else if (!strcmp(rm, "DISC"))
                        geometryFlag = coSphere::RENDER_METHOD_DISC;
                    else if (!strcmp(rm, "TEXTURE"))
                        geometryFlag = coSphere::RENDER_METHOD_TEXTURE;
                    else if (!strcmp(rm, "CG_SHADER_INVERTED"))
                        geometryFlag = coSphere::RENDER_METHOD_CG_SHADER_INVERTED;
                }
                spheres->getAddresses(&farr[0], &farr[1], &farr[2], &farr[3]);
                if (cluster)
                {
                    addInt(size);
                    addInt(geometryFlag);
                    tb.addBinary((char *)farr[0], size * sizeof(float));
                    tb.addBinary((char *)farr[1], size * sizeof(float));
                    tb.addBinary((char *)farr[2], size * sizeof(float));
                    tb.addBinary((char *)farr[3], size * sizeof(float));
                }
            }
            else if (strcmp(type, "LINES") == 0)
            {
                coDoLines *lines = (coDoLines *)co;
                sizeu = lines->getNumLines();
                sizev = lines->getNumVertices();
                size = lines->getNumPoints();
                lines->getAddresses(&farr[0], &farr[1], &farr[2], &iarr[0], &iarr[1]);
                if (cluster)
                {
                    addInt(sizeu);
                    addInt(sizev);
                    addInt(size);
                    tb.addBinary((char *)farr[0], size * sizeof(float));
                    tb.addBinary((char *)farr[1], size * sizeof(float));
                    tb.addBinary((char *)farr[2], size * sizeof(float));
                    tb.addBinary((char *)iarr[0], sizev * sizeof(int));
                    tb.addBinary((char *)iarr[1], sizeu * sizeof(int));
                }
            }
            else if (strcmp(type, "QUADS") == 0)
            {
                coDoQuads *lines = (coDoQuads *)co;
                sizev = lines->getNumVertices();
                size = lines->getNumPoints();
                lines->getAddresses(&farr[0], &farr[1], &farr[2], &iarr[0]);
                if (cluster)
                {
                    addInt(sizev);
                    addInt(size);
                    tb.addBinary((char *)farr[0], size * sizeof(float));
                    tb.addBinary((char *)farr[1], size * sizeof(float));
                    tb.addBinary((char *)farr[2], size * sizeof(float));
                    tb.addBinary((char *)iarr[0], sizev * sizeof(int));
                }
            }
            else if (strcmp(type, "TRITRI") == 0)
            {
                coDoTriangles *lines = (coDoTriangles *)co;
                sizev = lines->getNumVertices();
                size = lines->getNumPoints();
                lines->getAddresses(&farr[0], &farr[1], &farr[2], &iarr[0]);
                if (cluster)
                {
                    addInt(sizev);
                    addInt(size);
                    tb.addBinary((char *)farr[0], size * sizeof(float));
                    tb.addBinary((char *)farr[1], size * sizeof(float));
                    tb.addBinary((char *)farr[2], size * sizeof(float));
                    tb.addBinary((char *)iarr[0], sizev * sizeof(int));
                }
            }
            else if (strcmp(type, "USTSTD") == 0)
            {
                coDoVec2 *vector_data = (coDoVec2 *)co;
                size = vector_data->getNumPoints();
                vector_data->getAddresses(&farr[0], &farr[1]);
                if (cluster)
                {
                    addInt(size);
                    tb.addBinary((char *)farr[0], size * sizeof(float));
                    tb.addBinary((char *)farr[1], size * sizeof(float));
                }
            }
            else if (strcmp(type, "USTVDT") == 0)
            {
                coDoVec3 *normal_data = (coDoVec3 *)co;
                size = normal_data->getNumPoints();
                normal_data->getAddresses(&farr[0], &farr[1], &farr[2]);
                if (cluster)
                {
                    addInt(size);
                    tb.addBinary((char *)farr[0], size * sizeof(float));
                    tb.addBinary((char *)farr[1], size * sizeof(float));
                    tb.addBinary((char *)farr[2], size * sizeof(float));
                }
            }
            else if (strcmp(type, "TEXTUR") == 0)
            {
                coDoTexture *tex = (coDoTexture *)co;
                coDoPixelImage *img = tex->getBuffer();
                texture = (unsigned char *)(img->getPixels());
                sizeu = img->getWidth();
                sizev = img->getHeight();
                sizew = img->getPixelsize();
                numTC = tex->getNumCoordinates();
                textureCoords = tex->getCoordinates();
                if (cluster)
                {
                    addInt(sizeu);
                    addInt(sizev);
                    addInt(sizew);
                    addInt(numTC);
                    tb.addBinary((char *)texture, sizeu * sizev * sizew);
                    tb.addBinary((char *)textureCoords[0], numTC * sizeof(float));
                    tb.addBinary((char *)textureCoords[1], numTC * sizeof(float));
                }
            }
            else if (strcmp(type, "RGBADT") == 0 || strcmp(type, "colors") == 0)
            {
                coDoRGBA *colors = (coDoRGBA *)co;
                colors->getAddress(&pc);
                size = colors->getNumPoints();
                if (cluster)
                {
                    addInt(size);
                    tb.addBinary((char *)pc, size * sizeof(int));
                }
            }
            else if (strcmp(type, "BYTEDT") == 0)
            {
                coDoByte *bytes = (coDoByte *)co;
                bytes->getAddress(&barr[0]);
                size = bytes->getNumPoints();
                if (cluster)
                {
                    addInt(size);
                    tb.addBinary((const char *)barr[0], size);
                }
            }
            else if (strcmp(type, "USTSDT") == 0)
            {
                coDoFloat *volume_sdata = (coDoFloat *)co;
                size = volume_sdata->getNumPoints();
                volume_sdata->getAddress(&farr[0]);
                if (cluster)
                {
                    addInt(size);
                    tb.addBinary((char *)farr[0], size * sizeof(float));
                }
            }
            else
            {
                tb.addBinary("UNKNOW", 6);
                cerr << "unknown dataobject" << endl;
            }
        }
        else
        {
            tb.addBinary("EMPTY ", 6);
            tb << "NoName";
            int nix = 0;
            addInt(nix);
            //send over an empty dataobject
        }

        //std::cerr << "CoviseRenderObject::<init> info: object " << (co ? co->getName() : "*NULL*") << " assigned to slaves";
        //for (std::vector<int>::iterator s = this->assignedTo.begin(); s != this->assignedTo.end(); ++s)
        //   std::cerr << " " << *s;
        //std::cerr << std::endl;

        Message msg(tb);
        coVRMSController::instance()->sendSlaves(&msg);

        //std::cerr << "CoviseRenderObject::<init> info: object " << (co ? co->getName() : "*NULL*") << " sent to slaves" << std::endl;

    } /* endif coVRMSController->isMaster() */
    else // Slave
    {
        // receive a dataobject

        //std::cerr << "CoviseRenderObject::<init> info: read from master" << std::endl;

        Message msg;
        coVRMSController::instance()->readMaster(&msg);
        TokenBuffer tb(&msg);

        strncpy(type, tb.getBinary(7), 7);
        char *n;
        tb >> n;
        name = new char[strlen(n) + 1];
        strcpy(name, n);

        //std::cerr << "CoviseRenderObject::<init> info: object " << (name ? name : "*NULL*") << " assigned to slaves";
        //for (std::vector<int>::iterator s = this->assignedTo.begin(); s != this->assignedTo.end(); ++s)
        //   std::cerr << " " << *s;
        //std::cerr << std::endl;

        copyInt(numAttributes);
        //cerr <<"Receiving Object " << name << " numAttribs: " << numAttributes << endl;
        //cerr <<"type " << type << endl;
        attrNames = new char *[numAttributes];
        attributes = new char *[numAttributes];
        for (i = 0; i < numAttributes; i++)
        {
            tb >> n;
            attrNames[i] = new char[strlen(n) + 1];
            strcpy(attrNames[i], n);
            tb >> n;
            attributes[i] = new char[strlen(n) + 1];
            strcpy(attributes[i], n);
        }

        if (strcmp(type, "GEOMET") == 0)
        {
            copyInt(geometryFlag);
        }

        if (strcmp(type, "SETELE") == 0)
        {
            copyInt(size);
        }
        else if (strcmp(type, "POLYGN") == 0)
        {
            copyInt(sizeu);
            copyInt(sizev);
            copyInt(size);
            farr[0] = new float[size];
            farr[1] = new float[size];
            farr[2] = new float[size];
            iarr[0] = new int[sizev];
            iarr[1] = new int[sizeu];
            memcpy(farr[0], tb.getBinary(size * sizeof(float)), size * sizeof(float));
            memcpy(farr[1], tb.getBinary(size * sizeof(float)), size * sizeof(float));
            memcpy(farr[2], tb.getBinary(size * sizeof(float)), size * sizeof(float));
            memcpy(iarr[0], tb.getBinary(sizev * sizeof(int)), sizev * sizeof(int));
            memcpy(iarr[1], tb.getBinary(sizeu * sizeof(int)), sizeu * sizeof(int));
        }
        else if (strcmp(type, "TRIANG") == 0)
        {
            copyInt(sizeu);
            copyInt(sizev);
            copyInt(size);
            farr[0] = new float[size];
            farr[1] = new float[size];
            farr[2] = new float[size];
            iarr[0] = new int[sizev];
            iarr[1] = new int[sizeu];
            memcpy(farr[0], tb.getBinary(size * sizeof(float)), size * sizeof(float));
            memcpy(farr[1], tb.getBinary(size * sizeof(float)), size * sizeof(float));
            memcpy(farr[2], tb.getBinary(size * sizeof(float)), size * sizeof(float));
            memcpy(iarr[0], tb.getBinary(sizev * sizeof(int)), sizev * sizeof(int));
            memcpy(iarr[1], tb.getBinary(sizeu * sizeof(int)), sizeu * sizeof(int));
        }
        else if (strcmp(type, "UNIGRD") == 0)
        {
            copyInt(sizeu);
            copyInt(sizev);
            copyInt(sizew);
            copyFloat(min_[0]);
            copyFloat(max_[0]);
            copyFloat(min_[1]);
            copyFloat(max_[1]);
            copyFloat(min_[2]);
            copyFloat(max_[2]);
        }
        else if (strcmp(type, "UNSGRD") == 0)
        {
            copyInt(sizeu);
            copyInt(sizev);
            copyInt(sizew);
            farr[0] = new float[sizew];
            farr[1] = new float[sizew];
            farr[2] = new float[sizew];
            iarr[0] = new int[sizev];
            iarr[1] = new int[sizeu];
            iarr[2] = new int[sizeu];
            memcpy(farr[0], tb.getBinary(sizew * sizeof(float)), sizew * sizeof(float));
            memcpy(farr[1], tb.getBinary(sizew * sizeof(float)), sizew * sizeof(float));
            memcpy(farr[2], tb.getBinary(sizew * sizeof(float)), sizew * sizeof(float));
            memcpy(iarr[0], tb.getBinary(sizev * sizeof(int)), sizev * sizeof(int));
            memcpy(iarr[1], tb.getBinary(sizeu * sizeof(int)), sizeu * sizeof(int));
            memcpy(iarr[2], tb.getBinary(sizeu * sizeof(int)), sizeu * sizeof(int));
        }
        else if (strcmp(type, "RCTGRD") == 0)
        {
            copyInt(sizeu);
            copyInt(sizev);
            copyInt(sizew);
            farr[0] = new float[sizeu];
            farr[1] = new float[sizev];
            farr[2] = new float[sizew];
            memcpy(farr[0], tb.getBinary(sizeu * sizeof(float)), sizeu * sizeof(float));
            memcpy(farr[1], tb.getBinary(sizev * sizeof(float)), sizev * sizeof(float));
            memcpy(farr[2], tb.getBinary(sizew * sizeof(float)), sizew * sizeof(float));
        }
        else if (strcmp(type, "STRGRD") == 0)
        {
            copyInt(sizeu);
            copyInt(sizev);
            copyInt(sizew);
            farr[0] = new float[sizeu * sizev * sizew];
            farr[1] = new float[sizeu * sizev * sizew];
            farr[2] = new float[sizeu * sizev * sizew];
            memcpy(farr[0], tb.getBinary(sizeu * sizev * sizew * sizeof(float)), sizeu * sizev * sizew * sizeof(float));
            memcpy(farr[1], tb.getBinary(sizeu * sizev * sizew * sizeof(float)), sizeu * sizev * sizew * sizeof(float));
            memcpy(farr[2], tb.getBinary(sizeu * sizev * sizew * sizeof(float)), sizeu * sizev * sizew * sizeof(float));
        }
        else if (strcmp(type, "POINTS") == 0)
        {
            copyInt(size);
            farr[0] = new float[size];
            farr[1] = new float[size];
            farr[2] = new float[size];
            memcpy(farr[0], tb.getBinary(size * sizeof(float)), size * sizeof(float));
            memcpy(farr[1], tb.getBinary(size * sizeof(float)), size * sizeof(float));
            memcpy(farr[2], tb.getBinary(size * sizeof(float)), size * sizeof(float));
        }
        else if (strcmp(type, "SPHERE") == 0)
        {
            copyInt(size);
            copyInt(geometryFlag);
            farr[0] = new float[size];
            farr[1] = new float[size];
            farr[2] = new float[size];
            farr[3] = new float[size];
            memcpy(farr[0], tb.getBinary(size * sizeof(float)), size * sizeof(float));
            memcpy(farr[1], tb.getBinary(size * sizeof(float)), size * sizeof(float));
            memcpy(farr[2], tb.getBinary(size * sizeof(float)), size * sizeof(float));
            memcpy(farr[3], tb.getBinary(size * sizeof(float)), size * sizeof(float));
        }
        else if (strcmp(type, "LINES") == 0)
        {
            copyInt(sizeu);
            copyInt(sizev);
            copyInt(size);
            farr[0] = new float[size];
            farr[1] = new float[size];
            farr[2] = new float[size];
            iarr[0] = new int[sizev];
            iarr[1] = new int[sizeu];
            memcpy(farr[0], tb.getBinary(size * sizeof(float)), size * sizeof(float));
            memcpy(farr[1], tb.getBinary(size * sizeof(float)), size * sizeof(float));
            memcpy(farr[2], tb.getBinary(size * sizeof(float)), size * sizeof(float));
            memcpy(iarr[0], tb.getBinary(sizev * sizeof(int)), sizev * sizeof(int));
            memcpy(iarr[1], tb.getBinary(sizeu * sizeof(int)), sizeu * sizeof(int));
        }
        else if (strcmp(type, "QUADS") == 0)
        {
            copyInt(sizev);
            copyInt(size);
            farr[0] = new float[size];
            farr[1] = new float[size];
            farr[2] = new float[size];
            iarr[0] = new int[sizev];
            memcpy(farr[0], tb.getBinary(size * sizeof(float)), size * sizeof(float));
            memcpy(farr[1], tb.getBinary(size * sizeof(float)), size * sizeof(float));
            memcpy(farr[2], tb.getBinary(size * sizeof(float)), size * sizeof(float));
            memcpy(iarr[0], tb.getBinary(sizev * sizeof(int)), sizev * sizeof(int));
        }
        else if (strcmp(type, "TRITRI") == 0)
        {
            copyInt(sizev);
            copyInt(size);
            farr[0] = new float[size];
            farr[1] = new float[size];
            farr[2] = new float[size];
            iarr[0] = new int[sizev];
            memcpy(farr[0], tb.getBinary(size * sizeof(float)), size * sizeof(float));
            memcpy(farr[1], tb.getBinary(size * sizeof(float)), size * sizeof(float));
            memcpy(farr[2], tb.getBinary(size * sizeof(float)), size * sizeof(float));
            memcpy(iarr[0], tb.getBinary(sizev * sizeof(int)), sizev * sizeof(int));
        }
        else if (strcmp(type, "USTSTD") == 0)
        {
            copyInt(size);
            farr[0] = new float[size];
            farr[1] = new float[size];
            memcpy(farr[0], tb.getBinary(size * sizeof(float)), size * sizeof(float));
            memcpy(farr[1], tb.getBinary(size * sizeof(float)), size * sizeof(float));
        }
        else if (strcmp(type, "USTVDT") == 0)
        {
            copyInt(size);
            farr[0] = new float[size];
            farr[1] = new float[size];
            farr[2] = new float[size];
            memcpy(farr[0], tb.getBinary(size * sizeof(float)), size * sizeof(float));
            memcpy(farr[1], tb.getBinary(size * sizeof(float)), size * sizeof(float));
            memcpy(farr[2], tb.getBinary(size * sizeof(float)), size * sizeof(float));
        }
        else if (strcmp(type, "STRVDT") == 0)
        {
            copyInt(size);
            farr[0] = new float[size];
            farr[1] = new float[size];
            farr[2] = new float[size];
            memcpy(farr[0], tb.getBinary(size * sizeof(float)), size * sizeof(float));
            memcpy(farr[1], tb.getBinary(size * sizeof(float)), size * sizeof(float));
            memcpy(farr[2], tb.getBinary(size * sizeof(float)), size * sizeof(float));
        }
        else if (strcmp(type, "TEXTUR") == 0)
        {
            copyInt(sizeu);
            copyInt(sizev);
            copyInt(sizew);
            copyInt(numTC);
            texture = new unsigned char[sizeu * sizev * sizew];
            textureCoords = new float *[2];
            textureCoords[0] = new float[numTC];
            textureCoords[1] = new float[numTC];
            memcpy(texture, tb.getBinary(sizeu * sizev * sizew), sizeu * sizev * sizew);
            memcpy(textureCoords[0], tb.getBinary(numTC * sizeof(float)), numTC * sizeof(float));
            memcpy(textureCoords[1], tb.getBinary(numTC * sizeof(float)), numTC * sizeof(float));
        }
        else if (strcmp(type, "RGBADT") == 0 || strcmp(type, "colors") == 0)
        {
            copyInt(size);
            pc = new int[size];
            memcpy(pc, tb.getBinary(size * sizeof(int)), size * sizeof(int));
        }
        else if (strcmp(type, "BYTEDT") == 0)
        {
            copyInt(size);
            barr[0] = new unsigned char[size];
            memcpy(barr[0], tb.getBinary(size), size);
        }
        else if (strcmp(type, "STRSDT") == 0)
        {
            copyInt(size);
            farr[0] = new float[size];
            memcpy(farr[0], tb.getBinary(size * sizeof(float)), size * sizeof(float));
        }
        else if (strcmp(type, "USTSDT") == 0)
        {
            copyInt(size);
            farr[0] = new float[size];
            memcpy(farr[0], tb.getBinary(size * sizeof(float)), size * sizeof(float));
        }
    }
}

CoviseRenderObject::CoviseRenderObject(const coDistributedObject *const *cos, const std::vector<int> &assignedTo)
    : assignedTo(assignedTo)
    , texture(NULL)
    , textureCoords(NULL)
    , numTC(0)
    , numAttributes(0)
    , attrNames(NULL)
    , attributes(NULL)
    , size(0)
    , sizeu(0)
    , sizev(0)
    , sizew(0)
    , COVdobj(NULL)
    , COVnormals(NULL)
    , COVtexture(NULL)
    , COVvertexAttribute(NULL)
    , name(NULL)
    , objs(NULL)
    , geometryObject(NULL)
    , normalObject(NULL)
    , colorObject(NULL)
    , textureObject(NULL)
    , vertexAttributeObject(NULL)
    , geometryFlag(0)
    , pc(NULL)
    , coviseObject(NULL)    // TODO
    , cluster(coVRMSController::instance()->isCluster())
{
    BOOST_STATIC_ASSERT(Field::NumChannels >= 4);
    BOOST_STATIC_ASSERT((int)Field::NumChannels >= (int)coDoGeometry::NumChannels);

    type[0] = '\0';//TODO

    for (int c = 0; c < Field::NumChannels; ++c)
    {
        barr[c] = NULL;
        iarr[c] = NULL;
        farr[c] = NULL;
        min_[c] = 0;
        max_[c] = 0;
        COVcolors[c] = NULL;
    }

    for (int c = 0; c < Field::NumChannels; ++c)
    {
        const coDistributedObject *co = cos[c];
        char type[7] = "";
        char *name = NULL;
        int numAttributes = 0;
        char **attrNames = NULL, **attributes = NULL;
        size_t size = 0;

        if (cluster && coVRDistributionManager::instance().isActive() && this->assignedTo.empty())
        {
            this->assignedTo = coVRDistributionManager::instance().assign(co);
        }

        if (coVRMSController::instance()->isMaster())
        {
            TokenBuffer tb;
            if (co)
            {
                memcpy(type, co->getType(), 7);
                name = new char[strlen(co->getName()) + 1];
                strcpy(name, co->getName());
                numAttributes = co->getNumAttributes();
                attrNames = new char *[numAttributes];
                attributes = new char *[numAttributes];
                const char **attrs = NULL, **anames = NULL;
                co->getAllAttributes(&anames, &attrs);
                for (int i = 0; i < numAttributes; i++)
                {
                    attrNames[i] = new char[strlen(anames[i]) + 1];
                    strcpy(attrNames[i], anames[i]);
                    attributes[i] = new char[strlen(attrs[i]) + 1];
                    strcpy(attributes[i], attrs[i]);
                }
                if (cluster)
                {
                    tb.addBinary(type, 7);
                    tb << name;
                    addInt(numAttributes);
                    for (int i = 0; i < numAttributes; i++)
                    {
                        tb << attrNames[i];
                        tb << attributes[i];
                    }
                }

                if (strcmp(type, "BYTEDT") == 0)
                {
                    coDoByte *bytes = (coDoByte *)co;
                    bytes->getAddress(&barr[c]);
                    size = bytes->getNumPoints();
                    if (cluster)
                    {
                        addInt(size);
                        tb.addBinary((const char *)barr[c], size);
                }
                }
                else if (strcmp(type, "USTSDT") == 0)
                {
                    coDoFloat *volume_sdata = (coDoFloat *)co;
                    size = volume_sdata->getNumPoints();
                    volume_sdata->getAddress(&farr[c]);
                    if (cluster)
                    {
                        addInt(size);
                        tb.addBinary((char *)farr[c], size * sizeof(float));
                    }
                }
            }
            else
            {
                tb.addBinary("EMPTY ", 6);
                tb << "NoName";
                int nix = 0;
                addInt(nix);
                //send over an empty dataobject
            }

            if (co)
            {
                const char *min_str = co->getAttribute("MIN");
                const char *max_str = co->getAttribute("MAX");

                min_[c] = min_str ? boost::lexical_cast<float>(min_str) : min_[c];
                max_[c] = max_str ? boost::lexical_cast<float>(max_str) : max_[c];
            }
            addFloat(min_[c]);
            addFloat(max_[c]);

            Message msg(tb);
            coVRMSController::instance()->sendSlaves(&msg);
        } /* endif coVRMSController->isMaster() */
        else // Slave
        {
            // receive a dataobject

            //std::cerr << "CoviseRenderObject::<init> info: read from master" << std::endl;

            Message msg;
            coVRMSController::instance()->readMaster(&msg);
            TokenBuffer tb(&msg);

            strncpy(type, tb.getBinary(7), 7);
            char *n;
            tb >> n;
            name = new char[strlen(n) + 1];
            strcpy(name, n);

            //std::cerr << "CoviseRenderObject::<init> info: object " << (name ? name : "*NULL*") << " assigned to slaves";
            //for (std::vector<int>::iterator s = this->assignedTo.begin(); s != this->assignedTo.end(); ++s)
            //   std::cerr << " " << *s;
            //std::cerr << std::endl;

            copyInt(numAttributes);
            //cerr <<"Receiving Object " << name << " numAttribs: " << numAttributes << endl;
            //cerr <<"type " << type << endl;
            attrNames = new char *[numAttributes];
            attributes = new char *[numAttributes];
            for (int i = 0; i < numAttributes; i++)
            {
                tb >> n;
                attrNames[i] = new char[strlen(n) + 1];
                strcpy(attrNames[i], n);
                tb >> n;
                attributes[i] = new char[strlen(n) + 1];
                strcpy(attributes[i], n);
            }

            if (strcmp(type, "BYTEDT") == 0)
            {
                copyInt(size);
                barr[c] = new unsigned char[size];
                memcpy(barr[c], tb.getBinary(size), size);
            }
            else if (strcmp(type, "USTSDT") == 0)
            {
                copyInt(size);
                farr[c] = new float[size];
                memcpy(farr[c], tb.getBinary(size * sizeof(float)), size * sizeof(float));
            }

            copyFloat(min_[c]);
            copyFloat(max_[c]);
        }

        if (c == 0)
        {
            memcpy(this->type, type, sizeof(this->type));
            this->name = name;
            this->numAttributes = numAttributes;
            this->attrNames = attrNames;
            this->attributes = attributes;
            this->size = size;
        }
        else
        {
            if (size>0 && size!=this->size)
            {
                cerr << "size mismatch: channel 0: " << this->size << ", channel " << c << ": " << size << std::endl;
            }
            delete[] name;
            delete[] attrNames;
            delete[] attributes;
        }
    }
}

CoviseRenderObject::~CoviseRenderObject()
{
    delete[] objs;
    delete geometryObject;
    delete normalObject;
    delete colorObject;
    delete textureObject;
    delete vertexAttributeObject;
    if (!coVRMSController::instance()->isMaster())
    {
        int i;
        delete[] texture;
        if (textureCoords)
        {
            delete[] textureCoords[0];
            delete[] textureCoords[1];
        }
        delete[] textureCoords;
        for (i = 0; i < numAttributes; i++)
        {
            delete[] attrNames[i];
            delete[] attributes[i];
        }
        delete[] attrNames;
        delete[] attributes;
        for (int c = 0; c < Field::NumChannels; ++c)
        {
            delete[] barr[c];
            delete[] iarr[c];
            delete[] farr[c];
        }
        delete coviseObject;
        delete COVdobj;
        delete COVnormals;
        for (int c = 0; c < Field::NumChannels; ++c)
            delete COVcolors[c];
        delete COVtexture;
        delete COVvertexAttribute;
        delete[] name;
    }
}

CoviseRenderObject *CoviseRenderObject::getGeometry() const
{
    if (geometryObject)
        return geometryObject;
    if (geometryFlag & HASOBJ)
    {
        geometryObject = new CoviseRenderObject(COVdobj, this->assignedTo);
        return geometryObject;
    }
    else
    {
        return NULL;
    }
}

CoviseRenderObject *CoviseRenderObject::getNormals() const
{
    if (normalObject)
        return normalObject;
    if (geometryFlag & HASNORMALS)
    {
        normalObject = new CoviseRenderObject(COVnormals, this->assignedTo);
        return normalObject;
    }
    else
    {
        return NULL;
    }
}

CoviseRenderObject *CoviseRenderObject::getColors() const
{
    if (colorObject)
        return colorObject;
    if (geometryFlag & HASCOLORS)
    {
        if (geometryFlag & HASMULTICOLORS)
            colorObject = new CoviseRenderObject(COVcolors, this->assignedTo);
        else
            colorObject = new CoviseRenderObject(COVcolors[0], this->assignedTo);
        return colorObject;
    }
    else
    {
        return NULL;
    }
}

CoviseRenderObject *CoviseRenderObject::getTexture() const
{
    if (textureObject)
        return textureObject;
    if (geometryFlag & HASTEXTURE)
    {
        textureObject = new CoviseRenderObject(COVtexture, this->assignedTo);
        return textureObject;
    }
    else
    {
        return NULL;
    }
}

CoviseRenderObject *CoviseRenderObject::getVertexAttribute() const
{
    if (vertexAttributeObject)
        return vertexAttributeObject;
    if (geometryFlag & HASVERTEXATTRIBUTE)
    {
        vertexAttributeObject = new CoviseRenderObject(COVvertexAttribute, this->assignedTo);
        return vertexAttributeObject;
    }
    else
    {
        return NULL;
    }
}

const char *CoviseRenderObject::getAttribute(const char *attributeName) const
{
    for (int i = numAttributes - 1; i >= 0; i--)
    {
        if (strcmp(attributeName, attrNames[i]) == 0)
            return attributes[i];
    }
    return NULL;
}

const char *CoviseRenderObject::getAttributeName(size_t idx) const
{
    if (ssize_t(idx) >= numAttributes)
        return NULL;

    return attrNames[idx];
}

const char *CoviseRenderObject::getAttributeValue(size_t idx) const
{
    if (ssize_t(idx) >= numAttributes)
        return NULL;

    return attributes[idx];
}

size_t CoviseRenderObject::getNumAttributes() const
{
    return numAttributes;
}

int CoviseRenderObject::getAllAttributes(char **&name, char **&value) const
{
    name = attrNames;
    value = attributes;
    return numAttributes;
}

CoviseRenderObject *CoviseRenderObject::getElement(size_t idx) const
{
    if (ssize_t(idx) >= size)
        return NULL;

    int num;
    std::vector<std::vector<int> > assignments;
    RenderObject **objs = getAllElements(num, assignments);

    return dynamic_cast<CoviseRenderObject *>(objs[idx]);
}

RenderObject **CoviseRenderObject::getAllElements(int &numElements, std::vector<std::vector<int> > &assignments) const
{
    numElements = size;
    if (size == 0)
        return NULL;
    int i;

    bool unassigned = assignments.empty();

    if (objs == NULL)
    {
        objs = new RenderObject *[numElements];
        if (coVRMSController::instance()->isMaster())
        {
            const coDoSet *set = (const coDoSet *)coviseObject;
            const coDistributedObject *const *dobjs;
            dobjs = set->getAllElements(&numElements);
            assert(size == numElements);
            for (i = 0; i < numElements; i++)
            {
                //std::cerr << "CoviseRenderObject::getAllElements info: sending " << numElements << " elements";
                if (unassigned)
                {
                    CoviseRenderObject *cobj = new CoviseRenderObject(dobjs[i]);
                    assignments.push_back(cobj->getAssignment());
                    objs[i] = cobj;
                }
                else
                {
                    objs[i] = new CoviseRenderObject(dobjs[i], assignments[i]);
                }
            }
        }
        else
        {
            for (i = 0; i < numElements; i++)
            {
                //std::cerr << "CoviseRenderObject::getAllElements info: receiving " << numElements << " elements";
                if (unassigned)
                {
                    CoviseRenderObject *cobj = new CoviseRenderObject((const coDistributedObject *)0);
                    assignments.push_back(cobj->getAssignment());
                    objs[i] = cobj;
                }
                else
                {
                    objs[i] = new CoviseRenderObject((const coDistributedObject *)0, assignments[i]);
                }
            }
        }
    }
    return objs;
}

const char *CoviseRenderObject::getType() const
{
    return type;
}

bool
CoviseRenderObject::IsTypeField(const vector<string> &types,
                                bool strict_case)
{
    if (isType("SETELE"))
    {
        int no_elems;
        std::vector<std::vector<int> > a;
        getAllElements(no_elems, a);
        int i;
        bool ret = true;
        for (i = 0; i < no_elems; ++i)
        {
            bool thisType = static_cast<CoviseRenderObject *>(objs[i])->IsTypeField(types, strict_case);
            if (!thisType && strict_case)
            {
                ret = false;
                break;
            }
            else if (!strict_case && thisType)
            {
                break;
            }
        }
        if (!ret) // see !thisType && strict_case
        {
            return false;
        }
        else if (i != no_elems || strict_case) // see !strict_case && thisType
        {
            //   or strict_case and all true
            return true;
        }
        else
        {
            return false;
        }
    }
    else
    {
        unsigned int type;
        for (type = 0; type < types.size(); ++type)
        {
            if (types[type] == getType())
            {
                break;
            }
        }
        if (type == types.size())
        {
            return false;
        }
    }
    return true;
}

int CoviseRenderObject::getFloatRGBA(int pos, float *r, float *g, float *b, float *a) const
{
    if (pos < 0 || pos >= getNumColors())
        return 0;

    unsigned char *chptr = (unsigned char *)&pc[pos];
#ifdef BYTESWAP
    *a = ((float)(*chptr)) / (float)255.0;
    chptr++;
    *b = ((float)(*chptr)) / (float)255.0;
    chptr++;
    *g = ((float)(*chptr)) / (float)255.0;
    chptr++;
    *r = ((float)(*chptr)) / (float)255.0;
#else
    *r = ((float)(*chptr)) / (float)255.0;
    chptr++;
    *g = ((float)(*chptr)) / (float)255.0;
    chptr++;
    *b = ((float)(*chptr)) / (float)255.0;
    chptr++;
    *a = ((float)(*chptr)) / (float)255.0;
#endif
    return 1;
}

const unsigned char *CoviseRenderObject::getByte(Field::Id idx) const
{
    if (idx >= Field::Channel0 && idx < Field::NumChannels)
    {
        return barr[idx];
    }

    switch (idx)
    {
    case Field::Byte:
        return barr[0];
    case Field::Texture:
        return texture;
    default:
        break;
    }

    return NULL;
}

const int *CoviseRenderObject::getInt(Field::Id idx) const
{
    switch (idx)
    {
    case Field::RGBA:
        return pc;
    case Field::Connections:
        return iarr[0];
    case Field::Elements:
        return iarr[1];
    case Field::Types:
        return iarr[2];
    default:
        break;
    }

    return NULL;
}

const float *CoviseRenderObject::getFloat(Field::Id idx) const
{
    if (idx >= Field::Channel0 && idx < Field::NumChannels)
    {
        return farr[idx];
    }

    switch (idx)
    {
    case Field::X:
    case Field::Red:
        return farr[0];
    case Field::Y:
    case Field::Green:
        return farr[1];
    case Field::Z:
    case Field::Blue:
        return farr[2];
    default:
        break;
    }

    return NULL;
}

bool CoviseRenderObject::isAssignedToMe() const
{

    if (!coVRDistributionManager::instance().isActive() || !coVRMSController::instance()->isCluster())
    {
        return true;
    }

    // Assigned to all
    if (this->assignedTo.size() > 0 && this->assignedTo[0] == -1)
        return true;

    int myID = coVRMSController::instance()->getID();

    bool assignedToMe = find(this->assignedTo.begin(), this->assignedTo.end(), myID) != this->assignedTo.end();

    //   if (assignedToMe)
    //      std::cerr << "CoviseRenderObject::isAssignedToMe info: object " << (getName() ? getName() : "*NULL*") << " is assigned to me" << std::endl;

    return assignedToMe;
}
