/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _RWUSG_H
#define _RWUSG_H
/**************************************************************************\ 
 **                                                           (C)1995 RUS  **
 **                                                                        **
 ** Description: Read module for COVISE USG data        	                  **
 **                                                                        **
 **                                                                        **
 **                                                                        **
 **                                                                        **
 **                                                                        **
 ** Author:                                                                **
 **                                                                        **
 **                             Uwe Woessner                               **
 **                Computer Center University of Stuttgart                 **
 **                            Allmandring 30                              **
 **                            70550 Stuttgart                             **
 **                                                                        **
 ** Date:  17.11.95  V1.0                                                  **
\**************************************************************************/

#include <api/coModule.h>
using namespace covise;
#include <util/coviseCompat.h>
#include <reader/CoviseIO.h>

class RWCovise : public CoviseIO, public coModule
{

private:
    //  member functions

    virtual int compute(const char *port);
    virtual void postInst();
    virtual void param(const char *, bool inMapLoading);

    //  Parameter names
    std::string grid_Path;
    coFileBrowserParam *p_grid_path;
    coIntScalarParam *_p_firstStep;
    coIntScalarParam *_p_numSteps;
    coIntScalarParam *_p_skipStep;
    coIntScalarParam *_p_step;
    coBooleanParam *_p_rotate;
    coBooleanParam *_p_force;
    coChoiceParam *_p_RotAxis;
    coBooleanParam *_p_increment_suffix;

    char *s_RotAxis[3];
    coFloatParam *_p_rot_speed;

    //  Ports
    coInputPort *p_mesh_in;
    coOutputPort *p_mesh;

    int useMagic; // true if Magic should be written.

    int _fd;
    int _number_of_elements;
    bool _trueOpen;

    int suffix_number;

protected:
    virtual int covOpenInFile(const char *grid_Path);
    virtual int covCloseInFile(int fd);

public:
    RWCovise(int argc, char *argv[]);
    virtual ~RWCovise();
};
#endif // _RWUSG_H
