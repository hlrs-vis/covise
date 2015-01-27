/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _COVISE_FILES_H_
#define _COVISE_FILES_H_

#include <fstream.h>

class covOutFile
{
private:
    // our output stream
    ofstream d_str;

    // write to the file
    void write(const void *data, int bytes)
    {
        d_str.write((char *)data, bytes);
    }

public:
    // create a .covise output file
    covOutFile(const char *filename);

    // call before writing set elements, call writeattrib once after the elements
    void writeSetHeader(int numElem);

    // add attributes: user only for Sets and Geometries
    void writeattrib(const char **atNam, const char **atVal);

    // Write USG
    void writeUSG(int numElem, int numConn, int numVert,
                  const int *el, const int *cl, const int *tl,
                  const float *x, const float *y, const float *z,
                  const char **atNam, const char **atVal);

    // S3D Data
    void writeS3D(int numElem, const float *x,
                  const char **atNam, const char **atVal);
    // V3D Data
    void writeV3D(int numElem,
                  const float *x, const float *y, const float *z,
                  const char **atNam, const char **atVal);
};
#endif
