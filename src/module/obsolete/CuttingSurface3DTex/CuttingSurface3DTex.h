/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _CUTTINGSURFACE_3DTEX_H
#define _CUTTINGSURFACE_3DTEX_H

// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// ++                                                        (C)2000 RUS  ++
// ++ Description: Combine uniform grid with attributes of Cuttingsurface ++
// ++                                                                     ++
// ++ Author:                                                             ++
// ++                                                                     ++
// ++                            Daniela Rainer                           ++
// ++               Computer Center University of Stuttgart               ++
// ++                            Allmandring 30                           ++
// ++                           70550 Stuttgart                           ++
// ++                                                                     ++
// ++**********************************************************************/

#include <api/coModule.h>
using namespace covise;

class CuttingSurface3DTex : public coModule
{

private:
    virtual int compute(const char *);

    coInputPort *port_samplegrid; // uniform grid created by sample module
    coInputPort *port_samplegrid_colors; // colors on the uniform grid
    coInputPort *port_cuttingsurface; // polygon of the cuttingsurface
    coInputPort *port_cuttingsurface_colors; // colors on the poly
    coOutputPort *port_cut3DTex; // type geometry with attributes

public:
    CuttingSurface3DTex(int argc, char *argv[]);
};
#endif
