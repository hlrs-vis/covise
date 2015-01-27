/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#if !defined(_CELLTOVERT_H)
#define _CELLTOVERT_H
/**************************************************************************\ 
 **                                                           (C)1997 RUS  **
 **                                                                        **
 ** Description: Interpolation from Cell Data to Vertex Data               **
 **                                                                        **
 **                                                                        **
 **                                                                        **
 **                                                                        **
 **                                                                        **
 ** Author:                                                                **
 **                                                                        **
 **                            Andreas Werner                              **
 **                Computer Center University of Stuttgart                 **
 **                            Allmandring 30                              **
 **                            70550 Stuttgart                             **
 **                                                                        **
 ** Date:  05.01.97  V0.1                                                  **
 ** Date:  02.12.98  V1.0 (lt_te)                                          **
\**************************************************************************/

#include <api/coSimpleModule.h>
using namespace covise;
#include <util/coviseCompat.h>

class CellToVert;

class CellToVert : public coSimpleModule
{
private:
    coInputPort *grid_in;
    coInputPort *data_in;
    coOutputPort *data_out;
    coChoiceParam *algorithm;

public:
    CellToVert(int argc, char *argv[]);

    virtual ~CellToVert(){};

    int compute(const char *port);
};
#endif // _CELLTOVERT_H
