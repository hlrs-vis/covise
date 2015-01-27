/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/**************************************************************************\ 
 **                                                          (C)2005 RRZK  **
 **                                                                        **
 ** Description: Read STP3 volume files.                                   **
 **                                                                        **
 ** Author:      Martin Aumueller <aumueller@uni-koeln.de>                 **
 **                                                                        **
 ** Cration Date: 05.01.2005                                               **
\**************************************************************************/

#ifndef _READ_STP3_H
#define _READ_STP3_H

#include <api/coModule.h>
using namespace covise;

/** Reader module for STP3 volume files.
    The volume data will be in the range from 0.0 to 1.0.
    The volume files can be written by WriteVolume.
    @see coWriteVolume
*/

#define NO_VOIS 3

class coReadSTP3 : public coModule
{
private:
    int voi_total_no;
    int32_t resolution;
    float pixel_size;
    int32_t num_slices;
    float *slice_z;

    // Ports:
    coOutputPort *poGrid;
    coOutputPort *poVolume;
    coOutputPort *poVoi[NO_VOIS];

    // Parameters:
    coFileBrowserParam *pbrVolumeFile;
    coBooleanParam *pboUseVoi;
    coIntScalarParam *pisVoiNo;
    coFloatParam *pfsIgnoreValue;

    coIntScalarParam *pisVolumeFromVoi[NO_VOIS];

    // Methods:
    virtual int compute(const char *port);

    int readVoiSlice(FILE *fp, int voi_num, vector<float> *x, vector<float> *y);
    int getTransformation(const char *filename, double *mat, double *inv);
    void computeIntersections(int y, const vector<float> &p_x, const vector<float> &p_y, vector<float> *i_x);

public:
    coReadSTP3(int argc, char *argv[]);
};
#endif
