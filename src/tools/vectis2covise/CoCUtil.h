/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef COCUTIL_H
#define COCUTIL_H

#include <stdio.h>
#include <stdlib.h>

void initUNSGRD(CoC_UNSGRD *grd);
void cleanUNSGRD(CoC_UNSGRD *grd);
void writeUNSGRD(FILE *fd, CoC_UNSGRD *grd);
void writeUNSGRDasci(FILE *fd, CoC_UNSGRD *grd);
void writeCoFileHeader(FILE *fd);
void writeCoTimeStepsAttr(FILE *fd, int numSteps);
void initIdxPhd(CoC_Polyed_UNSGRD *pIdx);
void initIdxPgn(CoC_Polyed_POLYGN *pIdx);
void make_covise_directory(char *basename);
FILE *open_covise_file(char *fname, char *mode);
char *get_covise_pathname(char *fname);
void covise_message(int msgtype, char *s);
int checkPolyeder(int *poly, int num);
#endif
