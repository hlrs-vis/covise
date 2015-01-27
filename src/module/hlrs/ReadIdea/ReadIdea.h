/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _READ_IDEA_H
#define _READ_IDEA_H
/**************************************************************************\
**                                                   	      (C)2008 HLRS  **
**                                                                        **
** Description: READ Idea (MPA)                                           **
**                                                                        **
**                                                                        **
**                                                                        **
**                                                                        **
**                                                                        **
** Author: Martin Becker                                                  **
**                                                                        **
\**************************************************************************/

#include <stdlib.h>
#include <stdio.h>
#ifndef _WIN32
#include <unistd.h>
#endif
#include <api/coModule.h>
using namespace covise;

#define ULTRASOUND 0
#define RADAR 1

class ReadIdea : public coModule
{

private:
    //  member functions
    virtual int compute(const char *port);

    //  member data
    coOutputPort *p_mesh;
    coOutputPort *p_polygons;

    coOutputPort *p_intensity;

    coFileBrowserParam *p_mainFile;

    coChoiceParam *p_method;

    coFloatParam *p_freqStart;
    coFloatParam *p_freqEnd;

    coFloatParam *p_zScale;
    coFloatParam *p_zMax;
    coFloatParam *p_minAngle;

    coBooleanParam *p_normalize;
    coBooleanParam *p_inverse_frequency;

    //coFloatParam *p_getZLevel;

    int *get_triangles(int n, int *n_trias);
    void removeOuterEdges(int n);

public:
    ReadIdea(int argc, char *argv[]);
    virtual ~ReadIdea();
};

#endif
