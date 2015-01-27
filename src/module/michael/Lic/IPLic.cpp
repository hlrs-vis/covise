/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

// xx.yy.2002 / 1 / file IsoParaLic.cpp

/******************************************************************************\ 
 **                                                              (C)2001 RUS **
 **                                                                          **
 ** Description:  COVISE LineIntegralConvolution application module          **
 **                                                                          **
 **                                                                          **
 **                                                                          **
 **                                                                          **
 **                                                                          **
 ** Author: M. Muench                                                        **
 **                                                                          **
 ** History:                                                                 **
 ** xx. ???? 01 v1                                                            **
 ** XXXXXXXXX xx new covise api                                              **
 **                                                                          **
\******************************************************************************/

#include "IPLic.h"

/********************\ 
 *                  *
 * Covise main loop *
 *                  *
\********************/

int main(int argc, char *argv[])
{
    Lic *application = new Lic();
    application->start(argc, argv);
    return 0;
}

/******************************\ 
 *                            *
 * Ingredients of Application *
 *                            *
\******************************/

Lic::Lic()
{
    // this info appears in the module setup window
    set_module_description("LIC testing device");

    //parameters
    //resolution = addInt32Param("Resolution", "resolution");
    pixImgWidth = addInt32Param("Width", "width of pixel image");
    pixImgHeight = addInt32Param("Height", "height of pixel image");
    pixelSize = addInt32Param("Pixel Size", "bytes per pixel");
    //domainSize = addFloatParam("Domain Size", "size");
    scaleQuad = addFloatParam("Scale Quad", "quad size");

    //const int defaultDim = 2;
    //resolution->setValue(defaultDim);

    const int defaultWidth = 8;
    pixImgWidth->setValue(defaultWidth);

    const int defaultHeight = 8;
    pixImgHeight->setValue(defaultHeight);

    const int defaultPixelSize = 4;
    pixelSize->setValue(defaultPixelSize);

    //const float defaultSize = 1.0;
    //domainSize->setValue(defaultSize);

    const float defaultSQ = 1.1;
    scaleQuad->setValue(defaultSQ);

    // the input ports
    polygonInPort = addInputPort("polygonIn", "coDoPolygons", "Polygons");
    //vectorInPort = addInputPort("vectorIn","coDoVec3",\ 
   "Vector Data");

    // the output ports
    polygonOutPort = addOutputPort("polygonOut", "coDoPolygons", "Polygons");
    packageOutPort = addOutputPort("packageOut", "coDoPolygons", "2D Triangle Patch");
    textureOutPort = addOutputPort("textureOut", "coDoTexture", "Lic Texture");
}

Lic::~Lic()
{
}

void Lic::quit()
{
}

int Lic::compute()
{
    //int dimension = resolution->getValue();
    int width = pixImgWidth->getValue();
    int height = pixImgHeight->getValue();
    int pix = pixelSize->getValue();
    //float size = domainSize->getValue();
    float sq = scaleQuad->getValue();

    int num_triangles = 0;
    trivec triangles = trivec();

    Patch *triPatch = new Patch();

    coDoPolygons *polygons = 0;
    coDistributedObject *pInObj = 0;
    pInObj = polygonInPort->getCurrentObject();
    if (pInObj != 0)
    {
        if (pInObj->isType("POLYGN"))
        {
            if (!(polygons = dynamic_cast<coDoPolygons *>(pInObj)))
            {
                // this should never occur
                cerr << "FATAL error:  GetSubset::compute( ) dynamic cast failed in line "
                     << __LINE__ << " of file " << __FILE__ << endl;
            }
            else
                ;
        }
        else
        {
            sendError("Did not receive a POLYGON object at port %s",
                      polygonInPort->getName());
            sendInfo("Did not receive a POLYGON object at port %s.",
                     polygonInPort->getName());

            return STOP_PIPELINE;
        }
    }
    doPolygons(&triPatch, triangles, &polygons, &num_triangles, sq);

    //coDoPixelImage* pixels = NULL;
    //doPixelImage(width, height, pix, &pixels);

    //coDoVec3* vectorData = NULL;
    //vectorData = vectorInPort->getCurrentObject();
    //doConvolution(&triPatch, triangles, &polygons, vectorData, &pixels);

    //coDoTexture* texture = NULL;
    //doTexture(dimension, pix, &polygons, &pixels, &texture);

    delete triPatch;

    return SUCCESS;
}

void Lic::doPolygons(Patch **triPatch, trivec &triangles, coDoPolygons **polygon,
                     int *num_triangles, float sq)
{
    ///////////////////////////////////////////////////////////////////////
    /* 
      int num_points = (dimension+1)*(dimension+1);
      f2ten coordinates = f2ten(3);
      {
         for(int i = 0; i < 3; i++)
         {
            (coordinates[i]).resize(num_points);
         }
      }

      int num_polygons = 2*(dimension*dimension);  //triangles
   ivec polys = ivec(num_polygons);

   int num_corners = 3*num_polygons;  //triangles
   ivec corners = ivec(num_corners);

   float fact = static_cast<float> (dimension);
   fact = size/fact;

   //set coordinates
   {
   for(int i = 0; i < (dimension+1); i++)
   {
   for(int j = 0; j < (dimension+1); j++)
   {
   coordinates[0][i*(dimension+1)+j] = (fact*j);
   coordinates[1][i*(dimension+1)+j] = (fact*i);
   coordinates[2][i*(dimension+1)+j] = 0;
   }
   }
   }

   //set polys
   {
   for(int i = 0; i < num_polygons; i++)
   {
   polys[i] = 3*i;
   }
   }

   //set corners
   {
   int i = 0;
   int col = 0;
   while( i < num_corners )
   {
   if( ((col+1) %  (dimension+1)) == 0 )
   ++col;
   else
   ;

   corners[i] = col;
   corners[i+1] = col+1;
   corners[i+2] = col+(dimension+1);

   corners[i+3] = corners[i+1] = col+1;
   corners[i+4] = (col+1)+(dimension+1);
   corners[i+5] = col+(dimension+1);

   i += 6;
   ++col;
   }
   }

   (*polygon) = new coDoPolygons(polygonOutPort->getObjName(), num_points,\ 
   &coordinates[0][0], &coordinates[1][0],\ 
   &coordinates[2][0], num_corners, &corners[0],\ 
   num_polygons, &polys[0]);
   */
    /////////////////////////////////////////////////////////////////////

    //polygonOutPort->setCurrentObject(*polygon);
    //(*polygon)->addAttribute("vertexOrder","2");

    //do the triangle list
    int numTri = 0;

    {
        int npp = (*polygon)->getNumPoints();
        int npv = (*polygon)->getNumVertices();
        int npoly = (*polygon)->getNumPolygons();

        numTri = npoly;
        *num_triangles = numTri;

        triangles.resize(numTri);

        float *cx;
        float *cy;
        float *cz;
        int *cl;
        int *pl;

        ivec pNodes = ivec();
        i2ten pEdges = i2ten();
        f2ten pCoord = f2ten(3);

        {
            for (int i = 0; i < 3; i++)
            {
                (pCoord[i]).resize(npp);
            }
        }

        (*polygon)->getAddresses(&cx, &cy, &cz, &cl, &pl);

        //////////////////////////////////////////////////////////////////

        int kk = 0;
        int begin = 0;
        int end = 0;

        for (int j = 0; j < numTri; j++)
        {

            (triangles[j]).setId(*(pl + j));

            if (j == (numTri - 1))
            {
                end = (npv);
                begin = (*(pl + j));
            }
            else
            {
                end = ((*(pl + (j + 1))));
                begin = (*(pl + j));
            }

            for (kk = begin; kk < end; kk++)
            {
                if ((kk == begin))
                {
                    pNodes.resize((end - begin));
                    pNodes[kk - begin] = *(cl + kk);
                }
                else if ((kk != begin) && (pNodes.size() != (end - begin)))
                {
                    pNodes.resize((end - begin));
                    pNodes[kk - begin] = *(cl + kk);
                }
                else
                {
                    pNodes[kk - begin] = *(cl + kk);
                }
            }
            (triangles[j]).setNodes(pNodes);

            //edges
            int sz = end - begin;
            //cout << "\nend-begin = " << flush << sz << '\n' << flush;

            if (pEdges.size() < sz)
            {
                pEdges.resize(sz);
            }
            else
            {
            }

            {
                for (int i = 0; i < sz; i++)
                {
                    (pEdges[i]).resize(2);

                    if (i == (sz - 1))
                    {
                        pEdges[i][0] = *(cl + begin + i);
                        pEdges[i][1] = *(cl + begin);
                    }
                    else
                    {
                        pEdges[i][0] = *(cl + begin + i);
                        pEdges[i][1] = *(cl + begin + i + 1);
                    }
                }
            }
            (triangles[j]).setEdges(pEdges);

            //coordinates
            ivec tmp = ivec();
            ivecCopy(tmp, (triangles[j]).getNodes());
            {
                int ni = 0;
                int size = tmp.size();
                for (int i = 0; i < 3; i++)
                {
                    (pCoord[i]).resize(size);
                }

                for (int kl = 0; kl < size; kl++)
                {
                    ni = tmp[kl];
                    pCoord[0][kl] = *(cx + ni);
                    pCoord[1][kl] = *(cy + ni);
                    pCoord[2][kl] = *(cz + ni);
                }
            }
            (triangles[j]).setCoord(pCoord);
        }
        /* 
            //create triangle neighbour list
            {
               for(int i = 0; i < numTri; i++)
               {
                  for(int j = i+1; j < numTri; j++)
                  {
                     //test for common edge
                     if( commonEdge( &(triangles[i]), &(triangles[j]) ) == true)
                     {
                        (triangles[i]).setNeighbour(j, COMMON_EDGE);
      (triangles[j]).setNeighbour(i, COMMON_EDGE);
      }
      else
      {
      //test for common node
      if( commonNode( &(triangles[i]), &(triangles[j]) ) == true )
      {
      (triangles[i]).setNeighbour(j, COMMON_NODE);
      (triangles[j]).setNeighbour(i, COMMON_NODE);
      }
      else
      {
      }
      }
      }
      }
      }
      */

        //set 2D euclidean coordinates for triangles
        {
            f2ten tmp = f2ten(3);
            (tmp[0]).resize(3, 0);
            (tmp[1]).resize(3, 0);
            (tmp[2]).resize(3, 0);
            for (int i = 0; i < numTri; i++)
            {
                tmp = (triangles[i]).getCoord();
                (triangles[i]).setC2d(tmp);
                (triangles[i]).normaliseC2d();
                //if( i < IMIN(numTri, 10) )
                //{
                //(triangles[i]).prCoord();
                //(triangles[i]).prC2d();
                //}
                //else;
            }
        }

        //////////////////////////////////////////////////////////////////

        //create a patch of triangles packed to a quad
        (*triPatch)->setQ(sq);
        (*triPatch)->pack(triangles);

        coDoPolygons *package = NULL;
        triPack2polygons(&packageOutPort, &package, triangles);
    }
}

void Lic::doPixelImage(int width, int height, int pix, coDoPixelImage **pixels)
{
    long seed = -37;
    int PIXEL_SIZE = pix;
    int PIXEL_FORMAT = pix;

    int num_pixels = (width) * (height);
    int img_size = PIXEL_SIZE * num_pixels;

    char *image = new char[img_size];

    {
        for (int i = 0; i < num_pixels; i++)
        {
            if (PIXEL_SIZE == 1)
            {
                image[i] = static_cast<char>(255.0 * random2(&seed) + 0.5);
                if (image[i] < 128)
                {
                    image[i] = 0;
                }
                else
                {
                    image[i] = 255;
                }
            }
            else if (PIXEL_SIZE == 4)
            {
                image[4 * i] = static_cast<unsigned char>(255.0 * random2(&seed) + 0.5);
                image[4 * i + 1] = static_cast<unsigned char>(255.0 * random2(&seed) + 0.5);
                image[4 * i + 2] = static_cast<unsigned char>(255.0 * random2(&seed) + 0.5);
                image[4 * i + 3] = 255;
            }
            else
            {
                Covise::sendError("\"Unimplemented\" Pixel Size");
                return;
            }
        }
    }

    char *name = new char[strlen(textureOutPort->getObjName()) + 5];
    strcpy(name, textureOutPort->getObjName());
    strcat(name, "_Img");

    *pixels = new coDoPixelImage(name, width, height, PIXEL_FORMAT, PIXEL_SIZE,
                                 image);

    return;
}

void Lic::doTexture(int dimension, int pix, coDoPolygons **polygons,
                    coDoPixelImage **pixels, coDoTexture **texture)
{
    int PIXEL_SIZE = pix;

    int nPoints = (*polygons)->getNumPoints();

    float *x_coord = NULL;
    float *y_coord = NULL;
    float *z_coord = NULL;

    int *corner_list = NULL;
    int *polygon_list = NULL;

    (*polygons)->getAddresses(&x_coord, &y_coord, &z_coord, &corner_list,
                              &polygon_list);

    int *vertList = NULL;
    float **texCoord = NULL;

    //create texture coordinates
    int nVert = nPoints;
    {
        vertList = new int[nVert];
        for (int i = 0; i < nVert; i++)
        {
            vertList[i] = i;
        }
    }

    {
        int i;

        texCoord = new float *[2];
        texCoord[0] = new float[nPoints];
        texCoord[1] = new float[nPoints];

        if (((dimension + 1) * (dimension + 1)) != nPoints)
        {
            Covise::sendError("\nnumber of points inconsistent\n");
            return;
        }

        for (i = 0; i < nPoints; i++)
        {
            texCoord[0][i] = static_cast<float>(i % (dimension + 1));
            //0...dimension, 0...dimension, ...
            //0.0...1.0
            texCoord[0][i] /= static_cast<float>(dimension);
        }

        for (i = 0; i < nPoints; i++)
        {
            texCoord[1][i] = static_cast<float>(i / (dimension + 1));
            //0...0, 1...1, ..., dimension...dimension
            //0.0...1.0
            texCoord[1][i] /= static_cast<float>(dimension);
        }
    }

    //

    *texture = new coDoTexture(textureOutPort->getObjName(), *pixels, 0,
                               PIXEL_SIZE, TEXTURE_LEVEL, nVert, vertList,
                               nPoints, texCoord);
    textureOutPort->setCurrentObject(*texture);

    delete[] vertList;
    delete[] texCoord[0];
    delete[] texCoord[1];
    delete[] texCoord;

    return;
}

void Lic::doConvolution(Patch **triPatch, trivec &triangles,
                        coDoPolygons **polygon, coDoVec3 *vdata,
                        coDoPixelImage **pixels)
{
}
