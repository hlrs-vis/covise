/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/**************************************************************************
 *	        Libary to create files in COVISE format                   *
 *                                			                  *
 *                           (C) 2001 			                  *
 *                   VirCinity IT-Consulting GmbH                          *
 *                         Nobelstrasse 15				  *
 *                       D-70569 Stuttgart				  *
 *                            Germany	                                  *
 *                                       				  *
 * Author: Sven Kufer( sk@viricinity.com)                                  *
 * Date: 28. Juli 2001							  *
 **************************************************************************/

#ifndef _COVISE_WRITEFILELIB
#define _COVISE_WRITEFILELIB

#include "coFileExport.h"

/* COVISE unstructured grid types
 * See ProgrammingGuide Chapter 3 COVISE DataObjects, Unstructered Grid Types
 */

#if 0
#define CELL_TYPES_ONLY
#include <do/coDoUnstructuredGrid.h>
#undef CELL_TYPES_ONLY
#endif

/*************************************************************************
 *                                                                       *
 * 1. Attributes                                                         *
 *                                                                       *
 *************************************************************************/

/* Every data object in COVISE has attributes. An attribute has a string to store its name
 * and a string to store the value.
 *
 * Most used attributes are : TIMESTEP to indicate a series of timesteps, PART to
 * put an ID to an object, vertexOrder to control the lightning in the Renderer. See
 * libtest.cpp for examples.
 *
 * Attributes are set by following fields:
 *
 *             attrNam: list of pointers to attribute names
 *                    NOTE: either you give the number of attributes or
 *                    this list must be terminated by NULL( set numAttr=COUNT_ATTR in this case)
 *
 *             attrVal: list of pointers to attribute values
 *                    NOTE: see attrName conditions
 */

const int COUNT_ATTR = -1;

#ifdef __cplusplus
extern "C" {
#endif

/*************************************************************************
    *                                                                       *
    * 2. Return Values                                                      *
    *                                                                       *
    *************************************************************************/

/* ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    * all following functions return 0 in case of an error
    * +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    */

/*************************************************************************
    *                                                                       *
    * 3. Open/Close output file                                             *
    *                                                                       *
    *************************************************************************/

/* create a .covise output file: returns descriptor for the file */
int FILEEXPORT covOpenOutFile(const char *filename);

/* close output file */
int FILEEXPORT covCloseOutFile(int fd);

/*************************************************************************
    *                                                                       *
    * 4. COVISE objects                                                     *
    *                                                                       *
    *************************************************************************/

/********************************************************************
    *                                                                   *
    *   Following functions allow you to write out all kind of          *
    *   COVISE data types.                                              *
    *                                                                   *
    *   See ProgrammingGuide for detailed description.                  *
    *                                                                   *
    *   Common parameters:                                              *
    *        int fd  -  file descriptor                                 *
    *                   which was returned by covOpenOutFile            *
    *                                                                   *
    *        char **attrNam, char **attrVal, int numAttr  -             *
    *                   see Chapter Attributes                          *
    *                                                                   *
    ********************************************************************/

/* Unstructured Grid */
int FILEEXPORT covWriteUNSGRD(int fd, int numElem, int numConn, int numVert,
                              int *elemList, int *connList, int *typeList,
                              float *xCoord, float *yCoord, float *zCoord,
                              char **attrNam, char **attrVal, int numAttr);

/* Unstructured Float Scalar Data */
int FILEEXPORT covWriteUSTSDT(int fd, int numElem, float *values,
                              char **attrNam, char **attrVal, int numAttr);

/* Unstructured Integer Scalar Data */
int FILEEXPORT covWriteINTDT(int fd, int numElem, int *values,
                             char **attrNam, char **attrVal, int numAttr);

/* Unstructured Byte Scalar Data */
int FILEEXPORT covWriteBYTEDT(int fd, int numElem, unsigned char *values,
                              char **attrNam, char **attrVal, int numAttr);

/* Unstructured Vector Data*/
int FILEEXPORT covWriteUSTVDT(int fd, int numElem,
                              float *comp1, float *comp2, float *comp3,
                              char **attrNam, char **attrVal, int numAttr);

/* Unstructured Tensor Data */
int FILEEXPORT covWriteUSTTDT(int fd, int numElem, int type, float *values,
                              char **attrNam, char **attrVal, int numAttr);

/* Points */
int FILEEXPORT covWritePOINTS(int fd, int numPoints,
                              float *xCoord, float *yCoord, float *zCoord,
                              char **attrNam, char **attrVal, int numAttr);

/* Spheres */
int FILEEXPORT covWriteSPHERES(int fd, int numSpheres,
                               float *xCoord, float *yCoord, float *zCoord, float *radius,
                               char **attrNam, char **attrVal, int numAttr);

/* String object */
int FILEEXPORT covWriteDOTEXT(int fd, int numElem, char *data,
                              char **attrNam, char **attrVal, int numAttr);

/* Polygon */
int FILEEXPORT covWritePOLYGN(int fd, int numPolygons, int *polyList,
                              int numCorners, int *cornerList, int numPoints,
                              float *xCoord, float *yCoord, float *zCoord,
                              char **attrNam, char **attrVal, int numAttr);

/* Lines */
int FILEEXPORT covWriteLINES(int fd, int numLines, int *lineList,
                             int numCorners, int *cornerList, int numPoints,
                             float *xCoord, float *yCoord, float *zCoord,
                             char **attrNam, char **attrVal, int numAttr);

/* Triangles */
int FILEEXPORT covWriteTRI(int fd,
                           int numCorners, int *cornerList, int numPoints,
                           float *xCoord, float *yCoord, float *zCoord,
                           char **attrNam, char **attrVal, int numAttr);

/* Quads */
int FILEEXPORT covWriteQUADS(int fd,
                             int numCorners, int *cornerList, int numPoints,
                             float *xCoord, float *yCoord, float *zCoord,
                             char **attrNam, char **attrVal, int numAttr);

/* Triangle Strips */
int FILEEXPORT covWriteTRIANG(int fd, int numStrips, int *stripList,
                              int numCorners, int *cornerList, int numPoints,
                              float *xCoord, float *yCoord, float *zCoord,
                              char **attrNam, char **attrVal, int numAttr);

/* Uniform Grid */
int FILEEXPORT covWriteUNIGRD(int fd, int xsize, int ysize, int zsize,
                              float xmin, float xmax,
                              float ymin, float ymax,
                              float zmin, float zmax,
                              char **attrNam, char **attrVal, int numAttr);

/* Rectilinear Grid */
int FILEEXPORT covWriteRCTGRD(int fd, int xsize, int ysize, int zsize,
                              float *xCoord, float *yCoord, float *zCoord,
                              char **attrNam, char **attrVal, int numAttr);

/* Structured Grid */
int FILEEXPORT covWriteSTRGRD(int fd, int xsize, int ysize, int zsize,
                              float *xCoord, float *yCoord, float *zCoord,
                              char **attrNam, char **attrVal, int numAttr);

/* Structured Scalar Data */
int FILEEXPORT covWriteSTRSDT(int fd, int numElem, float *data,
                              int xsize, int ysize, int zsize,
                              char **attrNam, char **attrVal, int numAttr);

/* Structured Vector Data */
int FILEEXPORT covWriteSTRVDT(int fd, int numElem,
                              float *data_x, float *data_y, float *data_z,
                              int xsize, int ysize, int zsize,
                              char **attrNam, char **attrVal, int numAttr);

/* RGBA Data */
int FILEEXPORT covWriteRGBADT(int fd, int numElem, int *colors,
                              char **attrNam, char **attrVal, int numAttr);

/* Multi-dimensional Integer Array*/
int FILEEXPORT covWriteINTARR(int fd, int numDim, int numElem, int *dim_array, int *data,
                              char **attrNam, char **attrVal, int numAttr);

/* Texture */
int FILEEXPORT covWriteTEXTUR(int fd, int PixelImageWidth, int PixelImageHeight, int PixelImageSize,
                              int PixelImageFormatId, char *PixelImageBuffer,
                              char **ImageattrNam, char **ImageattrVal, int numImageAttr,
                              int NumberOfBorderPixels, int NumberOfComponents, int Level,
                              int NumberOfCoordinates, int NumberOfVertices, int *VertexIndices,
                              float **Coords, char **TextattrNam, char **TextattrVal, int numAttr);

/* Pixel Image */
int FILEEXPORT covWriteIMAGE(int fd, int PixelImageWidth, int PixelImageHeight, int PixelImageSize,
                             int PixelImageFormatId, char *PixelImageBuffer,
                             char **ImageattrNam, char **ImageattrVal, int numAttr);

/* OCT Tree */
int FILEEXPORT covWriteOCTREE(int fd,
                              int numCellLists,
                              int numMacroCellLists,
                              int numCellBBoxes,
                              int numGridBBoxes,
                              int *cellLists,
                              int *macroCellLists,
                              float *cellBBoxes,
                              float *gridBBoxes,
                              int fX,
                              int fY,
                              int fZ,
                              int max_no_levels);

/*************************************************************************
    *                                                                       *
    * 5. Grouping objects                                                   *
    *                                                                       *
    *************************************************************************/

/********************************************************************
    *                                                                  *
    * 5. 1.  Sets                                                      *
    *                                                                  *
    ********************************************************************/

/* Grouping objects is done by a set. For example, you define an object which contains several parts
    * by set of these parts. Or you define a series of time steps by a set of all time steps.
    * See libtest.cpp for an example.
    *
    * You have to use the following order to write a set of elements into a COVISE file:
    *
    *     covWriteSetBegin
    *         <write all elements of the set>
    *     covWriteSetEnd
    *
    * NOTE: every element can be an COVISE data object or a set of elements.
    *
    */

/* Start a set of "numElem" elements */
int FILEEXPORT covWriteSetBegin(int fd, int numElem);

/* End a set by its attributes */
int FILEEXPORT covWriteSetEnd(int fd, char **attrNam, char **attrVal, int numAttr);

/********************************************************************
    *                                                                  *
    * 5. 2.  Geometry Container                                        *
    *                                                                  *
    ********************************************************************/

/* The Geometry container is a special set to give the Renderer the object information
    *  in a structured way. In general this set is filled by the module Collect.
    *
    *  A geometry object is required for every Geometry container. The geometry
    *  object at the end of the pipeline normally consists of either polygons or lines.
    *  Optional you can add:  color information( given by the Colors module), normals and textures.
    *
    *  Content sum:
    *			geometry: 	         required
    *                   colors(RGBADT):          optional
    *                   normals(USTVDT):         optional
    *                   textures(TEXTUR):        optional
    * If you want to use an additional part set has_xxx to 1. Else set has_xxx to 0.
    *
    * NOTE: The Geometry container is a top level set. It musn't be included in another set.
    *
    */
int FILEEXPORT covWriteGeometryBegin(int fd, int has_colors, int has_normals, int has_texture);

/* end with attributes of Geometry container */
int FILEEXPORT covWriteGeometryEnd(int fd, char **attrNam, char **attrVal, int numAttr);

/* write reference to object */
extern int FILEEXPORT covWriteOBJREF(int fd, int num);
#ifdef __cplusplus
}
#endif
#endif
