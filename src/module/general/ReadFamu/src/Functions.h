/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include <util/coviseCompat.h>
#include "Plane.h"

void exportHmo(Plane myPlane, const char *planeFile);
void exportHmascii(Plane myPlane);
void filesBinding(const char *firstFile, const char *secondFile, const char *thirdFile, const char *targetFile);
bool FileExists(const char *filename);
void rotatePoint(Node *origNode, Node centerNode, float degree, int axis);
void transformBlock(const char *In_BlockFile, const char *tempFile, float *moveVec3, float *scaleVec3);
