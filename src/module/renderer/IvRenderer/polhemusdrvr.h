/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#define TRUE 1
#define FALSE 0

typedef struct
{
    float x, y, z, w, q1, q2, q3;

} PolhemusRecord;

int fastrackOpen(char *portname);
void fastrackReset(int desc);
void fastrackSetHemisphere(int desc, float p1, float p2, float p3);
void fastrackSetPositionFilter(int desc, float f, float flow, float fhigh,
                               float factor);
void fastrackSetAttitudeFilter(int desc, float f, float flow, float fhigh,
                               float factor);
void fastrackSetAsciiFormat(int desc);
void fastrackDisableContinuousOutput(int desc);
void fastrackSetUnitToInches(int desc);
void fastrackSetUnitToCentimeters(int desc);
void fastrackSetReferenceFrame(int desc, float Ox, float Oy, float Oz,
                               float Xx, float Xy, float Xz, float Yx, float Yy, float Yz);
void fastrackSetOutputToQuaternions(int desc);
void fastrackGetSingleRecord(int desc, PolhemusRecord *record);
