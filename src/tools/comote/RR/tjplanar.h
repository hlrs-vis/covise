/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef TJPLANAR_H
#define TJPLANAR_H

#ifdef __cplusplus
extern "C" {
#endif

#define jpegsub(s) (s >= 4 ? TJ_420 : s == 2 ? TJ_422 : s == 1 ? TJ_444 : s == 0 ? TJ_GRAYSCALE : TJ_444)

DLLEXPORT int DLLCALL tjDecompressPlanar(tjhandle h,
                                         unsigned char *srcbuf, unsigned long size,
                                         unsigned char **dstbuf, int width, int pitch, int height, int ncomp, int subsamp,
                                         int flags);

DLLEXPORT int DLLCALL tjCompressPlanar(tjhandle h,
                                       unsigned char **srcplane,
                                       int width, int pitch, int height, int ncomp,
                                       unsigned char *dstbuf,
                                       unsigned long *size,
                                       int subsamp,
                                       int qual,
                                       int flags);
#ifdef __cplusplus
}
#endif
#endif
