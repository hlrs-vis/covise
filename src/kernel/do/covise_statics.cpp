/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include <util/covise_list.h>

#include "coDoSet.h"
#include "coDoUniformGrid.h"
#include "coDoRectilinearGrid.h"
#include "coDoStructuredGrid.h"
#include "coDoUnstructuredGrid.h"
#include "coDoPoints.h"
#include "coDoSpheres.h"
#include "coDoLines.h"
#include "coDoPolygons.h"
#include "coDoTriangleStrips.h"
#include "coDoGeometry.h"
#include "coDoOctTree.h"
#include "coDoOctTreeP.h"
#include "coDoData.h"
#include "coDoIntArr.h"
#include "coDoText.h"
#include "coDoTexture.h"
#include "coDoPixelImage.h"
#include "coDoColormap.h"
#include "coDoDoubleArr.h"

/*
 $Log:  $
Revision 1.5  1994/03/23  18:07:07  zrf30125
Modifications for multiple Shared Memory segments have been finished
(not yet for Cray)

Revision 1.4  93/12/10  13:45:38  zrfg0125
statics order corrected and function moved

Revision 1.3  93/11/15  18:10:22  zrfg0125
strange static bug in IDE 4.1.1 compilers found

Revision 1.2  93/10/08  19:09:12  zrhk0125
modification for Cray initialization of statics

Revision 1.1  93/09/30  17:06:24  zrhk0125
Initial revision

*/

namespace covise
{

class coDoInitializer
{
public:
    coDoInitializer();
};
}

using namespace covise;

static coDoInitializer coDoInitialized;

coDoInitializer::coDoInitializer()
{
    coDistributedObject::set_vconstr("UNIGRD", coDoUniformGrid::virtualCtor);
    coDistributedObject::set_vconstr("RCTGRD", coDoRectilinearGrid::virtualCtor);
    coDistributedObject::set_vconstr("STRGRD", coDoStructuredGrid::virtualCtor);
    coDistributedObject::set_vconstr("UNSGRD", coDoUnstructuredGrid::virtualCtor);
    coDistributedObject::set_vconstr("OCTREE", coDoOctTree::virtualCtor);
    coDistributedObject::set_vconstr("OCTREP", coDoOctTreeP::virtualCtor);
    coDistributedObject::set_vconstr("POINTS", coDoPoints::virtualCtor);
    coDistributedObject::set_vconstr("SPHERES", coDoSpheres::virtualCtor);
    coDistributedObject::set_vconstr("LINES", coDoLines::virtualCtor);
    coDistributedObject::set_vconstr("TRITRI", coDoTriangles::virtualCtor);
    coDistributedObject::set_vconstr("QUADS", coDoQuads::virtualCtor);
    coDistributedObject::set_vconstr("POLYGN", coDoPolygons::virtualCtor);
    coDistributedObject::set_vconstr("TRIANG", coDoTriangleStrips::virtualCtor);
    coDistributedObject::set_vconstr("USTVDT", coDoVec3::virtualCtor);
    coDistributedObject::set_vconstr("USTREF", coDoMat3::virtualCtor);
    coDistributedObject::set_vconstr("USTTDT", coDoTensor::virtualCtor);
    coDistributedObject::set_vconstr("USTSTD", coDoVec2::virtualCtor);
    coDistributedObject::set_vconstr("GEOMET", coDoGeometry::virtualCtor);
    coDistributedObject::set_vconstr("SETELE", coDoSet::virtualCtor);
    coDistributedObject::set_vconstr("DOTEXT", coDoText::virtualCtor);
    coDistributedObject::set_vconstr("IMAGE", coDoPixelImage::virtualCtor);
    coDistributedObject::set_vconstr("TEXTUR", coDoTexture::virtualCtor);
    coDistributedObject::set_vconstr("INTARR", coDoIntArr::virtualCtor);
    coDistributedObject::set_vconstr("DBLARR", coDoDoubleArr::virtualCtor);
    coDistributedObject::set_vconstr("COLMAP", coDoColormap::virtualCtor);

    coDistributedObject::set_vconstr(RGBADT, coDoRGBA::virtualCtor);

    coDistributedObject::set_vconstr(USTSDT, coDoFloat::virtualCtor);
    coDistributedObject::set_vconstr(INTDT, coDoInt::virtualCtor);
    coDistributedObject::set_vconstr(BYTEDT, coDoByte::virtualCtor);
}

List<VirtualConstructor> *coDistributedObject::vconstr_list = NULL;
int coDistributedObject::xfer_arrays = 1;

int covise_std_compare(char *a, char *b)
{
    return (*a > *b ? 1 : (*a < *b ? -1 : 0));
}
