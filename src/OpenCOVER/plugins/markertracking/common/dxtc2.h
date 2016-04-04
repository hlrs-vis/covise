/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _DXTC2_H
#define _DXTC2_H

extern void cudaCompression(unsigned int width, unsigned int height, unsigned char *uncompressedData, unsigned char *compressedData, int videoMode);
extern void cudaDecompression(unsigned int width, unsigned int height, unsigned char *compressedData, unsigned char *uncompressedData, int videoMode);
extern void init_cuda();
extern void close_cuda();

#endif
