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

#include "coFileExport.h"

/** COVISE unstructured types **/

#ifdef __cplusplus
extern "C" {
#endif

extern int FILEEXPORT covCloseInFile(int fd);
extern int FILEEXPORT covOpenInFile(const char *filename);

extern int FILEEXPORT covReadDescription(int fd, char *name);
extern int FILEEXPORT covReadNumAttributes(int fd, int *num, int *size);
extern int FILEEXPORT covReadAttributes(int fd, char **atNam, char **atVal, int num, int size);

extern int FILEEXPORT covReadSizeUNSGRD(int fd, int *numElem, int *numConn, int *numVert);
extern int FILEEXPORT covReadUNSGRD(int fd, int numElem, int numConn, int numVert,
                                    int *el, int *cl, int *tl,
                                    float *x, float *y, float *z);
extern int FILEEXPORT covSkipUNSGRD(int fd, int numElem, int numConn, int numVert);

extern int FILEEXPORT covReadGeometryBegin(int fd, int *has_geometry, int *has_colors, int *has_normals, int *has_texture);
extern int FILEEXPORT covReadOldGeometryBegin(int fd, int *has_geometry, int *has_colors, int *has_normals);

extern int FILEEXPORT covReadSetBegin(int fd, int *numElem);

extern int FILEEXPORT covReadSizePOINTS(int fd, int *numElem);
extern int FILEEXPORT covReadPOINTS(int fd, int numElem, float *x, float *y, float *z);
extern int FILEEXPORT covSkipPOINTS(int fd, int numElem);

extern int FILEEXPORT covReadSizeSPHERES(int fd, int *numElem);
extern int FILEEXPORT covReadSPHERES(int fd, int numElem, float *x, float *y, float *z, float *radius);
extern int FILEEXPORT covSkipSPHERES(int fd, int numElem);

extern int FILEEXPORT covReadSizeDOTEXT(int fd, int *numElem);
extern int FILEEXPORT covReadDOTEXT(int fd, int numElem, char *data);
extern int FILEEXPORT covSkipDOTEXT(int fd, int numElem);

extern int FILEEXPORT covReadSizePOLYGN(int fd, int *numPolygons, int *numCorners, int *numPoints);
extern int FILEEXPORT covReadPOLYGN(int fd, int numPolygons, int *polyList, int numCorners, int *cornerList, int numPoints, float *x, float *y, float *z);
extern int FILEEXPORT covSkipPOLYGN(int fd, int numPolygons, int numCorners, int numPoints);

extern int FILEEXPORT covReadSizeLINES(int fd, int *numLines, int *numCorners, int *numPoints);
extern int FILEEXPORT covReadLINES(int fd, int numLines, int *lineList, int numCorners, int *cornerList,
                                   int numPoints, float *x, float *y, float *z);
extern int FILEEXPORT covSkipLINES(int fd, int numLines, int numCorners, int numPoints);

extern int FILEEXPORT covReadTRI(int fd, int numCorners, int *cornerList,
                                 int numPoints, float *x, float *y, float *z);

extern int FILEEXPORT covReadQUADS(int fd, int numCorners, int *cornerList,
                                   int numPoints, float *x, float *y, float *z);
extern int FILEEXPORT covSkipQUADS(int fd, int numCorners, int numPoints);

extern int FILEEXPORT covReadSizeTRIANG(int fd, int *numStrips, int *numCorners, int *numPoints);
extern int FILEEXPORT covReadTRIANG(int fd, int numStrips, int *stripList, int numCorners, int *cornerList,
                                    int numPoints, float *x, float *y, float *z);
extern int FILEEXPORT covSkipTRIANG(int fd, int numStrips, int numCorners, int numPoints);

extern int FILEEXPORT covReadUNIGRD(int fd, int *xsize, int *ysize, int *zsize, float *xmin, float *xmax, float *ymin, float *ymax, float *zmin, float *zmax);

extern int FILEEXPORT covReadSizeRCTGRD(int fd, int *xsize, int *ysize, int *zsize);
extern int FILEEXPORT covReadRCTGRD(int fd, int xsize, int ysize, int zsize, float *x, float *y, float *z);
extern int FILEEXPORT covSkipRCTGRD(int fd, int xsize, int ysize, int zsize);

extern int FILEEXPORT covReadSizeSTRGRD(int fd, int *xsize, int *ysize, int *zsize);
extern int FILEEXPORT covReadSTRGRD(int fd, int xsize, int ysize, int zsize, float *x, float *y, float *z);
extern int FILEEXPORT covSkipSTRGRD(int fd, int xsize, int ysize, int zsize);

extern int FILEEXPORT covReadSizeUSTTDT(int fd, int *numElem, int *type);
extern int FILEEXPORT covReadUSTTDT(int fd, int numElem, int type, float *x);
extern int FILEEXPORT covSkipUSTTDT(int fd, int numElem, int type);

extern int FILEEXPORT covReadSizeUSTSDT(int fd, int *numElem);
extern int FILEEXPORT covReadUSTSDT(int fd, int numElem, float *x);
extern int FILEEXPORT covSkipUSTSDT(int fd, int numElem);

extern int FILEEXPORT covReadSizeINTDT(int fd, int *numElem);
extern int FILEEXPORT covReadINTDT(int fd, int numElem, int *x);
extern int FILEEXPORT covSkipINTDT(int fd, int numElem);

extern int FILEEXPORT covReadSizeBYTEDT(int fd, int *numElem);
extern int FILEEXPORT covReadBYTEDT(int fd, int numElem, unsigned char *x);
extern int FILEEXPORT covSkipBYTEDT(int fd, int numElem);

extern int FILEEXPORT covReadSizeUSTVDT(int fd, int *numElem);
extern int FILEEXPORT covReadUSTVDT(int fd, int numElem, float *x, float *y, float *z);
extern int FILEEXPORT covSkipUSTVDT(int fd, int numElem);

extern int FILEEXPORT covReadSizeSTRSDT(int fd, int *numElem, int *xsize, int *ysize, int *zsize);
extern int FILEEXPORT covReadSTRSDT(int fd, int numElem, float *data, int xsize, int ysize, int zsize);
extern int FILEEXPORT covSkipSTRSDT(int fd, int numElem, int xsize, int ysize, int zsize);

extern int FILEEXPORT covReadSizeSTRVDT(int fd, int *numElem, int *xsize, int *ysize, int *zsize);
extern int FILEEXPORT covReadSTRVDT(int fd, int numElem, float *data_x, float *data_y, float *data_z, int xsize, int ysize, int zsize);
extern int FILEEXPORT covSkipSTRVDT(int fd, int numElem, int xsize, int ysize, int zsize);

extern int FILEEXPORT covReadSizeRGBADT(int fd, int *numElem);
extern int FILEEXPORT covReadRGBADT(int fd, int numElem, int *colors);
extern int FILEEXPORT covSkipRGBADT(int fd, int numElem);

extern int FILEEXPORT covReadSizeIMAGE(int fd, int *PixelImageWidth, int *PixelImageHeight, int *PixelImageSize,
                                       int *PixelImageFormatId, int *PixImageBufferLength);
extern int FILEEXPORT covReadIMAGE(int fd, int PixelImageWidth, int PixelImageHeight, int PixelImageSize,
                                   int PixelImageFormatId, int PixImageBufferLength, char *PixelImageBuffer);
extern int FILEEXPORT covSkipIMAGE(int fd, int PixelImageWidth, int PixelImageHeight, int PixelImageSize,
                                   int PixelImageFormatId, int PixImageBufferLength);

extern int FILEEXPORT covReadSizeTEXTUR(int fd, int *NumberOfBorderPixels, int *NumberOfComponents, int *Level,
                                        int *NumberOfCoordinates, int *NumberOfVertices);
extern int FILEEXPORT covReadTEXTUR(int fd, int NumberOfBorderPixels, int NumberOfComponents, int Level,
                                    int NumberOfCoordinates, int NumberOfVertices, int *VertexIndices,
                                    float **Coords);
extern int FILEEXPORT covSkipTEXTUR(int fd, int NumberOfBorderPixels, int NumberOfComponents, int Level,
                                    int NumberOfCoordinates, int NumberOfVertices);

extern int FILEEXPORT covReadDimINTARR(int fd, int *numDim);
extern int FILEEXPORT covReadSizeINTARR(int fd, int numDim, int *sizes, int *numElem);
extern int FILEEXPORT covReadINTARR(int fd, int numDim, int numElem, int *dim_array, int *data);
extern int FILEEXPORT covSkipINTARR(int fd, int numDim, int numElem, int *dim_array, int *data);

extern int FILEEXPORT covReadSizeOCTREE(int fd, int *numCellLists, int *numMacroCellLists, int *numCellBBoxes, int *numGridBBoxes);
extern int FILEEXPORT covReadOCTREE(int fd, int *numCellLists, int *numMacroCellLists, int *numCellBBoxes, int *numGridBBoxes, int *cellLists, int *macroCellLists, float *cellBBoxes, float *gridBBoxes, int *fX, int *fY, int *fZ, int *max_no_levels);
extern int FILEEXPORT covSkipOCTREE(int fd, int *numCellLists, int *numMacroCellLists, int *numCellBBoxes, int *numGridBBoxes, int *cellLists, int *macroCellLists, float *cellBBoxes, float *gridBBoxes, int *fX, int *fY, int *fZ, int *max_no_levels);

extern int FILEEXPORT covReadOBJREF(int fd, int *num);

#ifdef __cplusplus
}
#endif
#endif
