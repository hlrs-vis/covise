/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/**************************************************************************
 *	        Libary to read files in COVISE format                     *
 *                                			                  *
 *                           (C); 2001 			                  *
 *                   VirCinity IT-Consulting GmbH                          *
 *                         Nobelstrasse 15				  *
 *                       D-70569 Stuttgart				  *
 *                            Germany					  *
 * Author: S. Kufer							  *
 * Date: 28. Juli 2001							  *
 **************************************************************************/

#ifndef _COVISE_READFILELIB
#define _COVISE_READFILELIB

#include <util/coTypes.h>

/** COVISE unstructured types **/

#ifdef __cplusplus
extern "C" {
#endif

extern int FILEEXPORT covReadSizeUNSGRD(int fd, int *numElem, int *numConn, int *numVert);
extern int FILEEXPORT covReadUNSGRD(int fd, int numElem, int numConn, int numVert,
                                    int *el, int *cl, int *tl,
                                    float *x, float *y, float *z);
extern int FILEEXPORT covCloseInFile(int fd);
extern int FILEEXPORT covOpenInFile(char *filename);
extern int FILEEXPORT covReadNumAttributes(int fd, int *num, int *size);
extern int FILEEXPORT covReadAttributes(int fd, char **atNam, char **atVal, int num, int size);
extern int FILEEXPORT covReadDescription(int fd, char *name);

extern int FILEEXPORT covReadGeometryBegin(int fd, int *has_geometry, int *has_colors, int *has_normals, int *has_texture);
extern int FILEEXPORT covReadOldGeometryBegin(int fd, int *has_geometry, int *has_colors, int *has_normals);
extern int FILEEXPORT covReadSetBegin(int fd, int *numElem);
extern int FILEEXPORT covReadSizePOINTS(int fd, int *numElem);
extern int FILEEXPORT covReadPOINTS(int fd, int numElem, float *x, float *y, float *z);
extern int FILEEXPORT covReadSizeDOTEXT(int fd, int *numElem);
extern int FILEEXPORT covReadDOTEXT(int fd, int numElem, char *data);
extern int FILEEXPORT covReadSizePOLYGN(int fd, int *numPolygons, int *numCorners, int *numPoints);

extern int FILEEXPORT covReadPOLYGN(int fd, int numPolygons, int *polyList, int numCorners, int *cornerList, int numPoints, float *x, float *y, float *z);
extern int FILEEXPORT covReadSizeLINES(int fd, int *numLines, int *numCorners, int *numPoints);
extern int FILEEXPORT covReadLINES(int fd, int numLines, int *lineList, int numCorners, int *cornerList,
                                   int numPoints, float *x, float *y, float *z);
extern int FILEEXPORT covReadSizeTRIANG(int fd, int *numStrips, int *numCorners, int *numPoints);
extern int FILEEXPORT covReadTRIANG(int fd, int numStrips, int *stripList, int numCorners, int *cornerList,
                                    int numPoints, float *x, float *y, float *z);
extern int FILEEXPORT covReadUNIGRD(int fd, int *xsize, int *ysize, int *zsize, float *xmin, float *xmax, float *ymin, float *ymax, float *zmin, float *zmax);
extern int FILEEXPORT covReadSizeRCTGRD(int fd, int *xsize, int *ysize, int *zsize);
extern int FILEEXPORT covReadRCTGRD(int fd, int xsize, int ysize, int zsize, float *x, float *y, float *z);
extern int FILEEXPORT covReadSizeSTRGRD(int fd, int *xsize, int *ysize, int *zsize);

extern int FILEEXPORT covReadSTRGRD(int fd, int xsize, int ysize, int zsize, float *x, float *y, float *z);

extern int FILEEXPORT covReadSizeUSTTDT(int fd, int *numElem, int *type);
extern int FILEEXPORT covReadUSTTDT(int fd, int numElem, int type, float *x);
extern int FILEEXPORT covReadSizeUSTSDT(int fd, int *numElem);
extern int FILEEXPORT covReadUSTSDT(int fd, int numElem, float *x);
extern int FILEEXPORT covReadSizeUSTVDT(int fd, int *numElem);
extern int FILEEXPORT covReadUSTVDT(int fd, int numElem, float *x, float *y, float *z);
extern int FILEEXPORT covReadSizeSTRSDT(int fd, int *numElem, int *xsize, int *ysize, int *zsize);
extern int FILEEXPORT covReadSTRSDT(int fd, int numElem, float *data, int xsize, int ysize, int zsize);
extern int FILEEXPORT covReadSizeSTRVDT(int fd, int *numElem, int *xsize, int *ysize, int *zsize);
extern int FILEEXPORT covReadSTRVDT(int fd, int numElem, float *data_x, float *data_y, float *data_z, int xsize, int ysize, int zsize);
extern int FILEEXPORT covReadSizeRGBADT(int fd, int *numElem);
extern int FILEEXPORT covReadRGBADT(int fd, int numElem, int *colors);

extern int FILEEXPORT covReadSizeIMAGE(int fd, int *PixelImageWidth, int *PixelImageHeight, int *PixelImageSize,
                                       int *PixelImageFormatId, int *PixImageBufferLength);
extern int FILEEXPORT covReadIMAGE(int fd, int PixelImageWidth, int PixelImageHeight, int PixelImageSize,
                                   int PixelImageFormatId, int PixImageBufferLength, char *PixelImageBuffer);
extern int FILEEXPORT covReadSizeTEXTUR(int fd, int *NumberOfBorderPixels, int *NumberOfComponents, int *Level,
                                        int *NumberOfCoordinates, int *NumberOfVertices);

extern int FILEEXPORT covReadTEXTUR(int fd, int NumberOfBorderPixels, int NumberOfComponents, int Level,
                                    int NumberOfCoordinates, int NumberOfVertices, int *VertexIndices,
                                    float **Coords);
extern int FILEEXPORT covReadDimINTARR(int fd, int *numDim);
extern int FILEEXPORT covReadSizeINTARR(int fd, int numDim, int *sizes, int *numElem);
extern int FILEEXPORT covReadINTARR(int fd, int numDim, int numElem, int *dim_array, int *data);

#ifdef __cplusplus
}
#endif
#endif
