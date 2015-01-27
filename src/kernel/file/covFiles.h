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
 *                            Germany					  *
 * Author: S. Kufer							  *
 * Date: 28. Juli 2001							  *
 **************************************************************************/

#ifndef COV_FILES_H
#define COV_FILES_H

#include "coFileExport.h"

#ifdef __cplusplus
extern FILEEXPORT "C"
{
#endif

    struct CovFile
    {
        int fd;
        int mode;
        int byteswap;
    };

    extern FILEEXPORT int covIoAttrib(int fd, int mode, int *num, int *size, char **atNam,
                                      char **atVal);
    extern FILEEXPORT int covWriteAttrib(int fd, int num, char **atNam, char **atVal);
    extern FILEEXPORT int covIoGeometryBegin(int fd, int mode, int *has_geometry, int *has_colors, int *has_normals, int *has_texture);
    extern FILEEXPORT int covIoGeometryEnd(int fd, int mode, char **atNam, char **atVal, int numAttr);

    extern FILEEXPORT int covIoSetBegin(int fd, int mode, int *numElem);
    extern FILEEXPORT int covIoSetEnd(int fd, int mode, char **atNam, char **atVal, int numAttr);

    extern FILEEXPORT int covIoUNSGRD(int fd, int mode, int *numElem, int *numConn, int *numVert,
                                      int *el, int *cl, int *tl,
                                      float *x, float *y, float *z,
                                      char **atNam, char **atVal, int *numAttr);
    extern FILEEXPORT int covIoPOINTS(int fd, int mode, int *numElem, float *x, float *y, float *z, char **atNam, char **atVal, int *numAttr);
    extern FILEEXPORT int covIoDOTEXT(int fd, int mode, int *numElem, char *data, char **atNam, char **atVal, int *numAttr);
    extern FILEEXPORT int covIoMetaLines(int fd, int mode, char *type, int *numObjects, int *objectList, int *numCorners, int *cornerList,
                                         int *numPoints, float *x, float *y, float *z, char **atNam, char **atVal, int *numAttr);
    extern FILEEXPORT int covIoUNIGRD(int fd, int mode, int *xsize, int *ysize, int *zsize, float *xmin, float *xmax, float *ymin,
                                      float *ymax, float *zmin, float *zmax, char **atNam, char **atVal, int *numAttr);
    extern FILEEXPORT int covIoRCTGRD(int fd, int mode, int *xsize, int *ysize, int *zsize, float *x, float *y, float *z,
                                      char **atNam, char **atVal, int *numAttr);
    extern FILEEXPORT int covIoSTRGRD(int fd, int mode, int *xsize, int *ysize, int *zsize, float *x, float *y, float *z,
                                      char **atNam, char **atVal, int *numAttr);

    extern FILEEXPORT int covIoUSTTDT(int fd, int mode, int *numElem, int *type, float *x, char **atNam, char **atVal, int *numAttr);
    extern FILEEXPORT int covIoUSTSDT(int fd, int mode, int *numElem, float *x, char **atNam, char **atVal, int *numAttr);
    extern FILEEXPORT int covIoUSTVDT(int fd, int mode, int *numElem,
                                      float *x, float *y, float *z,
                                      char **atNam, char **atVal, int *numAttr);

    extern FILEEXPORT int covIoSTRSDT(int fd, int mode, int *numElem, float *data, int *xsize, int *ysize, int *zsize,
                                      char **atNam, char **atVal, int *numAttr);
    extern FILEEXPORT int covIoSTRVDT(int fd, int mode, int *numElem, float *data_x, float *data_y, float *data_z, int *xsize, int *ysize, int *zsize,
                                      char **atNam, char **atVal, int *numAttr);
    extern FILEEXPORT int covIoRGBADT(int fd, int mode, int *numElem, int *colors,
                                      char **atNam, char **atVal, int *numAttr);
    extern FILEEXPORT int covIoIMAGE(int fd, int mode, int *PixelImageWidth, int *PixelImageHeight, int *PixelImageSize,
                                     int *PixelImageFormatId, int *PixelImageBufferLength, char *PixelImageBuffer,
                                     char **ImageatNam, char **ImageatVal, int *numAttr);
    extern FILEEXPORT int covIoTEXTUR(int fd, int mode, int *PixelImageWidth, int *PixelImageHeight, int *PixelImageSize,
                                      int *PixelImageFormatId, int *PixImageBufferLength, char *PixelImageBuffer,
                                      char **ImageatNam, char **ImageatVal, int *numImageAttr,
                                      int *NumberOfBorderPixels, int *NumberOfComponents, int *Level,
                                      int *NumberOfCoordinates, int *NumberOfVertices, int *VertexIndices,
                                      float **Coords, char **TextatNam, char **TextatVal, int *numTextAttr);

    extern FILEEXPORT int covIoINTARR(int fd, int mode, int *numDim, int *numElem, int *dim_array, int *data,
                                      char **atNam, char **atVal, int *numAttr);
    extern FILEEXPORT int covIoINTDT(int fd, int mode, int *numElem, int *x, char **atNam, char **atVal, int *numAttr);

    extern FILEEXPORT int covIoOBJREF(int fd, int mode, int *objNum);

#ifdef __cplusplus
}
#endif
#endif
