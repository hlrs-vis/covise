/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _READ_DASIM_H
#define _READ_DASIM_H
/**************************************************************************\
**                                                   	      (C)2002 RUS **
**                                                                        **
** Description: READ Dasim result files             	                  **
**                                                                        **
**                                                                        **
**                                                                        **
**                                                                        **
**                                                                        **
** Author: Uwe Woessner                                                   **                             **
**                                                                        **
\**************************************************************************/

#include <api/coModule.h>
using namespace covise;
#include <util/coviseCompat.h>

class ReadDasim : public coModule
{

private:
    //  member functions
    virtual int compute(const char *port);

    //  member data
    coOutputPort *p_mesh;
    coOutputPort *p_data;
    coFileBrowserParam *p_fileParam;

public:
    ReadDasim(int argc, char *argv[]);
    virtual ~ReadDasim();
};

#endif
