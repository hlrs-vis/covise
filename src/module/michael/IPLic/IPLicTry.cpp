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
 ** xx. ???? 01 v1                                                           **
 ** XXXXXXXXX xx new covise api                                              **
 **                                                                          **
\******************************************************************************/

#include "IPLicTry.h"

/********************\ 
 *                  *
 * Covise main loop *
 *                  *
\********************/

int main(int argc, char *argv[])
{
    IPLic *application = new IPLic();
    application->start(argc, argv);
    return 0;
}

/******************************\ 
 *                            *
 * Ingredients of Application *
 *                            *
\******************************/

IPLic::IPLic()
{
    // this info appears in the module setup window
    set_module_description("LIC testing device");

    //parameters
    resolution = addInt32Param("Resolution", "resolution");
    //filterLength = addInt32Param("Filter Length", "filterLength");
    pixelSize = addInt32Param("Pixel Size", "bytes per pixel");

    const int defaultDim = 2;
    resolution->setValue(defaultDim);

    //const int defaultFLen = 0;
    //filterLength->setValue(defaultFLen);

    const int defaultPixelSize = 4;
    pixelSize->setValue(defaultPixelSize);

    // the input ports
    polygonInPort = addInputPort("polygonIn", "coDoPolygons", "Polygons");
    //polygonInPort->setRequired(0);
    vectorInPort = addInputPort("vectorIn", "coDoVec3", "Vector Data");
    vectorInPort->setRequired(0);

    // the output ports
    packageOutPort = addOutputPort("packageOut", "coDoPolygons", "2D Triangle Patch");
    textureOutPort = addOutputPort("textureOut", "coDoTexture", "IPLic Texture");
}

IPLic::~IPLic()
{
}

void IPLic::quit()
{
}

int IPLic::compute()
{
    int width = 1;
    int height = 1;

    int res = resolution->getValue();
    //int fLen = filterLength->getValue();
    int pix = pixelSize->getValue();

    int num_triangles = 0;
    int num_points = 0;
    i2ten triNeighbours = i2ten();
    trivec triangles = trivec();

    /////////////////////////////////////////////////////////////////////

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
                cerr << "FATAL error:  IPLic::compute( ) dynamic cast failed in line "
                     << __LINE__ << " of file " << __FILE__ << endl;
            }
            else
                ;
        }
        else
        {
            sendError("Did not receive a POLYGON object at port %s",
                      polygonInPort->getName());

            return STOP_PIPELINE;
        }
    }

    /////////////////////////////////////////////////////////////////////

    coDoVec3 *vectors = 0;
    coDistributedObject *vecInObj = 0;
    vecInObj = vectorInPort->getCurrentObject();
    if (vecInObj != 0)
    {
        if (vecInObj->isType("USTVDT"))
        {
            if (!(vectors = dynamic_cast<coDoVec3 *>(vecInObj)))
            {
                // this should never occur
                cerr << "FATAL error:  IPLic::compute( ) dynamic cast failed in line "
                     << __LINE__ << " of file " << __FILE__ << endl;
            }
            else
                ;
        }
        else
        {
            sendInfo("Did not receive a VECTOR object at port %s.",
                     vectorInPort->getName());

            return STOP_PIPELINE;
        }
    }
    else
    {
        cerr << "\nvecInObj = 0\n" << flush;
    }

    /////////////////////////////////////////////////////////////////////

    num_points = doPolygons(triangles, triNeighbours, &polygons,
                            &num_triangles);
    doVdata(triangles, &vectors, num_points);

    dimension(&width, &height, num_triangles);

    coDoPixelImage *pixels = NULL;
    //doPixelImage(width, height, res, pix, &pixels);

    coDoPolygons *texPoly = 0;
    doTexPoly(&texPoly, triangles, width, height, res);

    if (vectors != 0)
    {
        doTexVec(triangles); //could be done in doVdata(...) ?
        //printTexVec(triangles);
        //printVdata(triangles);
        doConvolution(triangles, &pixels, triNeighbours,
                      width, height, res, pix);
    }
    else
        ;

    coDoTexture *texture = NULL;
    doTexture(triangles, pix, &texPoly, &pixels, &texture);

    return SUCCESS;
}

int IPLic::doPolygons(trivec &triangles, i2ten &triNeighbours,
                      coDoPolygons **polygon, int *num_triangles)
{
    /////////////////////////////////////////////////////////////////////

    //do the triangle list
    int numTri = 0;

    int npp = (*polygon)->getNumPoints();
    int npv = (*polygon)->getNumVertices();
    int npoly = (*polygon)->getNumPolygons();

    numTri = npoly;
    *num_triangles = numTri;

    triangles.resize(numTri);
    triNeighbours.resize(3 * numTri);

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

    /////////////////////////////////////////////////////////////////////

    int kk = 0;
    int begin = 0;
    int end = 0;

    for (int j = 0; j < numTri; j++)
    {
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

    //create triangle neighbour list
    {
        int A = 0;
        int B = 0;
        int C = 0;
        ivec nodes = ivec(3, 0);

        for (int i = 0; i < numTri; i++)
        {
            nodes = (triangles[i]).getNodes();

            A = nodes[0];
            B = nodes[1];
            C = nodes[2];

            //which triangles belong to which node ??
            (triNeighbours[A]).push_back(i);
            (triNeighbours[B]).push_back(i);
            (triNeighbours[C]).push_back(i);
        }
    }

    /////////////////////////////////////////////////////////////////////

    return npp;
}

void IPLic::doVdata(trivec &triangles, coDoVec3 **vdata,
                    int num_values)
{
    /////////////////////////////////////////////////////////////////////

    int numTri = triangles.size();

    float *vx;
    float *vy;
    float *vz;

    f2ten v3d = f2ten(3);
    {
        int i;
        for (i = 0; i < 3; i++)
        {
            (v3d[i]).resize(num_values);
        }
    }

    if (*vdata != NULL)
    {
        (*vdata)->getAddresses(&vx, &vy, &vz);

        //////////////////////////////////////////////////////////////////

        {
            int j;
            for (j = 0; j < num_values; j++)
            {
                v3d[0][j] = *(vx + j);
                v3d[1][j] = *(vy + j);
                v3d[2][j] = *(vz + j);
            }
        }

        //////////////////////////////////////////////////////////////////

        {
            ivec nodes = ivec(3, 0);

            int i;
            for (i = 0; i < numTri; i++)
            {
                nodes = (triangles[i]).getNodes();
                (triangles[i]).setVdata(v3d, nodes);
            }
        }
    }

    else
    {
        cerr << "\n(*vdata) = 0\n" << flush;
    }
}

void IPLic::doTexPoly(coDoPolygons **texPoly, trivec &triangles,
                      int width, int height, int res)
{
    const int nt = triangles.size();
    const int first = 0;
    const int second = 1;
    const int third = 2;

    /////////////////////////////////////////////////////////////////////

    //cout << "\ndimension(...): width, height, num_triangles\n" << flush;
    //cout << width << flush << "  " << flush;
    //cout << height << flush << "  " << flush;
    //cout << nt << flush << "\n\n" << flush;

    /////////////////////////////////////////////////////////////////////

    float step = 0.0;
    if (width > 0)
    //if(height > 0)
    {
        step = 1.0 / static_cast<float>(width);
        //step = 1.0/static_cast<float>(height);
    }
    else
        ;

    float fract = 0.0;
    if (res > 0)
    {
        fract = step / res; //step = width of a square,
        //fract = width of a single pixel.
    }
    else
        ;

    /////////////////////////////////////////////////////////////////////

    fvec p1 = fvec(2, 0.0);
    fvec p2 = fvec(2, 0.0);
    fvec p3 = fvec(2, 0.0);
    //fvec p4 = fvec(2, 0.0);
    //fvec p5 = fvec(2, 0.0);
    //fvec p6 = fvec(2, 0.0);

    /////////////////////////////////////////////////////////////////////

    f2ten tpCoord(3);
    (tpCoord[0]).resize(3 * nt, 0.0);
    (tpCoord[1]).resize(3 * nt, 0.0);
    (tpCoord[2]).resize(3 * nt, 0.0);

    f2ten c3d_temp = f2ten(3);
    (c3d_temp[0]).resize(4, 0.0);
    (c3d_temp[1]).resize(4, 0.0);
    (c3d_temp[2]).resize(4, 0.0);

    /////////////////////////////////////////////////////////////////////

    ivec tpVert = ivec(3 * nt, 0);

    ivec tpPoly = ivec(nt, 0);

    /////////////////////////////////////////////////////////////////////

    int i;
    int jq;
    for (i = 0; i < nt; i++)
    {
        //jq = i/2;
        jq = i;

        //////////////////////////////////////////////////////////////////

        //first = (triangles[i]).getCoordIndex(1);
        //second = (triangles[i]).getCoordIndex(2);
        //third = (triangles[i]).getCoordIndex(3);

        //////////////////////////////////////////////////////////////////

        c3d_temp = (triangles[i]).getCoord();

        tpCoord[0][3 * i] = c3d_temp[0][first];
        tpCoord[1][3 * i] = c3d_temp[1][first];
        tpCoord[2][3 * i] = c3d_temp[2][first];

        tpCoord[0][3 * i + 1] = c3d_temp[0][second];
        tpCoord[1][3 * i + 1] = c3d_temp[1][second];
        tpCoord[2][3 * i + 1] = c3d_temp[2][second];

        tpCoord[0][3 * i + 2] = c3d_temp[0][third];
        tpCoord[1][3 * i + 2] = c3d_temp[1][third];
        tpCoord[2][3 * i + 2] = c3d_temp[2][third];

        //////////////////////////////////////////////////////////////////

        tpVert[3 * i] = 3 * i;
        tpVert[3 * i + 1] = 3 * i + 1;
        tpVert[3 * i + 2] = 3 * i + 2;

        tpPoly[i] = 3 * i;

        //////////////////////////////////////////////////////////////////

        int rmzero = IMAX(res, 0);
        //int rmone = IMAX(res-1, 0);
        //int rmtwo = IMAX(res-2, 0);

        if ((i % 2) == 0)
        {
            p1[0] = (jq % width) * step;
            p1[1] = (jq / width) * step;
            p2[0] = ((jq % width) * step) + (rmzero * fract);
            p2[1] = (jq / width) * step;
            p3[0] = (jq % width) * step;
            p3[1] = ((jq / width) * step) + (rmzero * fract);
        }
        else
        {
            p1[0] = (jq % width) * step;
            p1[1] = (jq / width) * step;
            p2[0] = ((jq % width) * step) + (rmzero * fract);
            p2[1] = (jq / width) * step;
            p3[0] = ((jq % width) * step) + (rmzero * fract);
            p3[1] = ((jq / width) * step) + (rmzero * fract);
        }

        //p4[0] = ((jq%width)*step) + (rmzero*fract);
        //p4[1] = (jq/width)*step + fract;
        //p5[0] = ((jq%width)*step) + (rmzero*fract);
        //p5[1] = ((jq/width)*step) + (rmzero*fract);
        //p6[0] = (jq%width)*step + fract;
        //p6[1] = ((jq/width)*step) + (rmzero*fract);

        //////////////////////////////////////////////////////////////////

        //p1[0] += halfFract;
        //p2[0] += halfFract;
        //p3[0] += halfFract;
        //p4[0] += halfFract;
        //p5[0] += halfFract;
        //p6[0] += halfFract;

        //p1[1] += halfFract;
        //p2[1] += halfFract;
        //p3[1] += halfFract;
        //p4[1] += halfFract;
        //p5[1] += halfFract;
        //p6[1] += halfFract;
        /*	
            cout << "\n\ntexture coordinates of triangles\n" << flush;
            cout << "\np1 = " << flush << p1[0] << flush << "  " << flush;
            cout << p1[1] << flush;
            cout << "\np2 = " << flush << p2[0] << flush << "  " << flush;
            cout << p2[1] << flush;
            cout << "\np3 = " << flush << p3[0] << flush << "  " << flush;
            cout << p3[1] << flush;
            cout << "\np4 = " << flush << p4[0] << flush << "  " << flush;
            cout << p4[1] << flush;
            cout << "\np5 = " << flush << p5[0] << flush << "  " << flush;
      cout << p5[1] << flush;
      cout << "\np6 = " << flush << p6[0] << flush << "  " << flush;
      cout << p6[1] << flush;
      */
        //////////////////////////////////////////////////////////////////

        (triangles[i]).initC2dTex();
        //if(i%2 == 0)
        //{
        (triangles[i]).setC2dTex(p1[0], 0, first);
        (triangles[i]).setC2dTex(p1[1], 1, first);

        (triangles[i]).setC2dTex(p2[0], 0, second);
        (triangles[i]).setC2dTex(p2[1], 1, second);

        (triangles[i]).setC2dTex(p3[0], 0, third);
        (triangles[i]).setC2dTex(p3[1], 1, third);
        //}
        /*
            else
            {
               (triangles[i]).setC2dTex(p6[0], 0, first);
               (triangles[i]).setC2dTex(p6[1], 1, first);

               (triangles[i]).setC2dTex(p4[0], 0, second);
               (triangles[i]).setC2dTex(p4[1], 1, second);

               (triangles[i]).setC2dTex(p5[0], 0, third);
               (triangles[i]).setC2dTex(p5[1], 1,third);
      }
      */
    }

    /////////////////////////////////////////////////////////////////////

    (*texPoly) = new coDoPolygons(packageOutPort->getObjName(), 3 * nt,
                                  &tpCoord[0][0], &tpCoord[1][0], &tpCoord[2][0],
                                  3 * nt, &tpVert[0], nt, &tpPoly[0]);

    packageOutPort->setCurrentObject(*texPoly);
    (*texPoly)->addAttribute("vertexOrder", "2");

    /////////////////////////////////////////////////////////////////////

    //cout << "\ndoTexPoly(...): width, height, num_triangles\n" << flush;
    //cout << width << flush << "  " << flush;
    //cout << height << flush << "  " << flush;
    //cout << nt << flush << "\n\n" << flush;

    /////////////////////////////////////////////////////////////////////
}

//task: projection of 3d-vectors onto triangle
//  &  transformation into isoparametric space
void IPLic::doTexVec(trivec &triangles)
{
    /////////////////////////////////////////////////////////////////////

    int i;
    int tsize = triangles.size();

    /////////////////////////////////////////////////////////////////////

    for (i = 0; i < tsize; i++)
    {
        (triangles[i]).setVTex();
    }
}

void IPLic::printTexVec(trivec &triangles)
{
    /////////////////////////////////////////////////////////////////////

    int i;
    int tsize = triangles.size();

    f2ten v = f2ten(2);
    (v[0]).resize(3, 0.0);
    (v[1]).resize(3, 0.0);

    //f2ten vo = f2ten(2);
    //(vo[0]).resize(3, 0.0);
    //(vo[1]).resize(3, 0.0);

    f2ten c = f2ten(2);
    (c[0]).resize(3, 0.0);
    (c[1]).resize(3, 0.0);

    f2ten co = f2ten(3);
    (co[0]).resize(3, 0.0);
    (co[1]).resize(3, 0.0);
    (co[2]).resize(3, 0.0);

    /////////////////////////////////////////////////////////////////////

    for (i = 0; i < tsize; i++)
    {
        v = (triangles[i]).getVTex();
        c = (triangles[i]).getC2dTex();
        co = (triangles[i]).getCoord();

        cout << "\n\n--------------------------------------------" << flush;

        cout << "\n\nA(" << flush << co[0][0] << flush;
        cout << '|' << flush << co[1][0] << flush << '|' << flush;
        cout << co[2][0] << flush << ')' << flush;

        cout << "   -->>   " << flush << '(' << flush;
        cout << c[0][0] << flush << '|' << flush;
        cout << c[1][0] << flush << ')' << flush;

        cout << "\nB(" << flush << co[0][1] << flush;
        cout << '|' << flush << co[1][1] << flush << '|' << flush;
        cout << co[2][1] << flush << ')' << flush;

        cout << "   -->>   " << flush << '(' << flush;
        cout << c[0][1] << flush << '|' << flush;
        cout << c[1][1] << flush << ')' << flush;

        cout << "\nC(" << flush << co[0][2] << flush;
        cout << '|' << flush << co[1][2] << flush << '|' << flush;
        cout << co[2][2] << flush << ')' << flush;

        cout << "   -->>   " << flush << '(' << flush;
        cout << c[0][2] << flush << '|' << flush;
        cout << c[1][2] << flush << ')' << flush;

        cout << "\n\nvA(" << flush << v[0][0] << flush;
        cout << '|' << flush << v[1][0] << flush << ')' << flush;

        cout << "\nvB(" << flush << v[0][1] << flush;
        cout << '|' << flush << v[1][1] << flush << ')' << flush;

        cout << "\nvC(" << flush << v[0][2] << flush;
        cout << '|' << flush << v[1][2] << flush << ')' << flush;

        cout << "\n\n--------------------------------------------" << flush;
    }
}

void IPLic::printVdata(trivec &triangles)
{
    /////////////////////////////////////////////////////////////////////

    int i;
    int tsize = triangles.size();

    f2ten v = f2ten(2);
    (v[0]).resize(3, 0.0);
    (v[1]).resize(3, 0.0);

    f2ten co = f2ten(3);
    (co[0]).resize(3, 0.0);
    (co[1]).resize(3, 0.0);
    (co[2]).resize(3, 0.0);

    /////////////////////////////////////////////////////////////////////

    for (i = 0; i < tsize; i++)
    {
        v = (triangles[i]).getVdata();
        co = (triangles[i]).getCoord();

        cout << "\n\n--------------------------------------------" << flush;

        cout << "\n\nA(" << flush << co[0][0] << flush;
        cout << '|' << flush << co[1][0] << flush << '|' << flush;
        cout << co[2][0] << flush << ')' << flush;

        cout << "\nB(" << flush << co[0][1] << flush;
        cout << '|' << flush << co[1][1] << flush << '|' << flush;
        cout << co[2][1] << flush << ')' << flush;

        cout << "\nC(" << flush << co[0][2] << flush;
        cout << '|' << flush << co[1][2] << flush << '|' << flush;
        cout << co[2][2] << flush << ')' << flush;

        cout << "\n\nvA(" << flush << v[0][0] << flush;
        cout << '|' << flush << v[1][0] << flush << ')' << flush;

        cout << "\nvB(" << flush << v[0][1] << flush;
        cout << '|' << flush << v[1][1] << flush << ')' << flush;

        cout << "\nvC(" << flush << v[0][2] << flush;
        cout << '|' << flush << v[1][2] << flush << ')' << flush;

        cout << "\n\n--------------------------------------------" << flush;
    }
}

void IPLic::doPixelImage(int width, int height, int res, int pix,
                         coDoPixelImage **pixels)
{
    long seed = -37;
    int PIXEL_SIZE = pix;
    int PIXEL_FORMAT = pix;

    int num_pixels = (width) * (height);
    num_pixels *= res;
    num_pixels *= res;
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

    *pixels = new coDoPixelImage(name, width * res, height * res, PIXEL_FORMAT,
                                 PIXEL_SIZE, image);

    return;
}

void IPLic::doTexture(trivec &triangles, int pix, coDoPolygons **texPoly,
                      coDoPixelImage **pixels, coDoTexture **texture)
{
    int PIXEL_SIZE = pix;

    int nPoints = (*texPoly)->getNumPoints();

    float *x_coord = NULL;
    float *y_coord = NULL;
    float *z_coord = NULL;

    int *corner_list = NULL;
    int *polygon_list = NULL;

    (*texPoly)->getAddresses(&x_coord, &y_coord, &z_coord, &corner_list,
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

        f2ten c2d = f2ten(2);
        (c2d[0]).resize(3, 0.0);
        (c2d[1]).resize(3, 0.0);

        int num_triangles = triangles.size();

        if ((3 * num_triangles) != nPoints)
        {
            Covise::sendError("\"3*number_of_triangles != number_of_points");
            return;
        }
        else
            ;

        for (i = 0; i < num_triangles; i++)
        {
            c2d = (triangles[i]).getC2dTex();

            texCoord[0][3 * i] = c2d[0][0];
            texCoord[1][3 * i] = c2d[1][0];

            texCoord[0][3 * i + 1] = c2d[0][1];
            texCoord[1][3 * i + 1] = c2d[1][1];

            texCoord[0][3 * i + 2] = c2d[0][2];
            texCoord[1][3 * i + 2] = c2d[1][2];
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

void IPLic::doConvolution(trivec &triangles, coDoPixelImage **pixels,
                          const i2ten &triNeighbours,
                          int width, int height, int res, int pix)
{
    long seed = -37;
    const int numTri = triangles.size();
    const int ww = width * res;
    const int hh = height * res;

    /////////////////////////////////////////////////////////////////

    i2ten hitcount = i2ten(hh);
    f3ten accum = f3ten();

    /////////////////////////////////////////////////////////////////////

    {
        hitcount.resize(hh);

        int i;
        for (i = 0; i < hh; i++)
        {
            (hitcount[i]).resize(ww, 0);
        }
    }

    /////////////////////////////////////////////////////////////////////

    const float delta = 1.0 / static_cast<float>(hh);

    /////////////////////////////////////////////////////////////////////

    //initialize accumulation buffer & hitcount

    if (pix == 1)
    {
        int i;

        accum.resize(1);
        (accum[0]).resize(hh);

        for (i = 0; i < hh; i++)
        {
            (accum[0][i]).resize(ww, 0.0);
        }
    }
    else if (pix == 4)
    {
        int i;
        int k;

        accum.resize(4);

        for (k = 0; k < 4; k++)
        {
            (accum[k]).resize(hh);

            for (i = 0; i < hh; i++)
            {
                (accum[k][i]).resize(ww, 0.0);
            }
        }
    }
    else
    {
        Covise::sendError("\"Unimplemented\" Pixel Size");
        return;
    }

    /////////////////////////////////////////////////////////////////////

    {
        int ii;
        int jj; //pixel indices

        //loop: start pix[0][0] -> pix[0][ww-1];pix[1][0] ...
        // -> pix[hh-1][ww-1] !
        for (ii = 0; ii < hh; ii++)
        {
            for (jj = 0; jj < ww; jj++)
            {
                if (hitcount[ii][jj] > 0)
                {
                    continue;
                }
                else
                {
                    doFieldLine(triangles, triNeighbours, accum, hitcount,
                                res, delta, ii, jj, hh, ww, &seed);
                }
            }
        }

        int img_size = (ww * hh) * pix;
        char *image = new char[img_size];

        accum2pixels(accum, hitcount, &image, img_size);

        char *name = new char[strlen(textureOutPort->getObjName()) + 5];
        strcpy(name, textureOutPort->getObjName());
        strcat(name, "_Img");

        *pixels = new coDoPixelImage(name, ww, hh, pix, pix, image);

        delete[] image;
    }

    /////////////////////////////////////////////////////////////////////
}
