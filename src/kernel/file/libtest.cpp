/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

//+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
//+                                                     (C)2002 VirCinity  ++
//+                                                                        ++
//+ This is an example to study the usage of the COVISE file library.      ++
//+ We want to write a polygon object with three parts. More over we store ++
//+ two time steps of this object.                                         ++
//+                                                                        ++
//+ Author:  Sven Kufer (sk@vircinity.com)                                 ++
//+                                                                        ++
//+               VirCinity GmbH                                           ++
//+               Nobelstrasse 15                                          ++
//+               70569 Stuttgart                                          ++
//+                                                                        ++
//+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

//include file for writing COVISE files
#include "covWriteFiles.h"

#include <stdio.h>

void WriteStep(int fd, int step);
void WritePart(int fd, int step, int part);

void main()
{
    // open file and remember file descriptor
    int fd = covOpenOutFile("libtest.covise");

    /*
      In COVISE a series of time steps is handled by a set which elements
      are the objects in every time step. Every object in a time step
      can be again a set of elements, but COVISE always expects that the set
      of time steps is the outer one.
      To store two time steps we open a set of two elements.
    */

    covWriteSetBegin(fd, 2);

    /* now we store the content of this set of time steps */
    WriteStep(fd, 1);
    WriteStep(fd, 2);

    /* As every object in COVISE a set has attributes. These are stored
      directly behind the object itself. To tell COVISE that
      this set is a series of time steps we add the attribute TIMESTEP 1 <#time steps>
    */

    char *ANames[] = { "TIMESTEP" };
    char *AValues[] = { "1 2" };

    covWriteSetEnd(fd, ANames, AValues, 1);

    //close the COVISE file
    covCloseOutFile(fd);
}

void WriteStep(int fd, int step)
{
    // We want to store three parts in every time step. This is done by a set of parts

    covWriteSetBegin(fd, 3);

    /* the three parts */
    WritePart(fd, step, 1);
    WritePart(fd, step, 2);
    WritePart(fd, step, 3);

    // You MUST add attributes to every COVISE object although you don't really have one!

    covWriteSetEnd(fd, NULL, NULL, COUNT_ATTR);
}

void WritePart(int fd, int step, int part)
{
    int i;

    // Polygon list and corner list as described in the programming guide for the coDoPolygons object
    int pl[] = { 0, 5 };
    int cl[] = { 0, 1, 6, 2, 3, 0, 5, 6, 3 };

    //coordinates
    float x[] = { 1.0, 2.0, 2.5, 3.0, 2.5, 1.0, 2.0 };
    float y[] = { -2.0, 2.5, 2.5, 1.5, 0.2, 1.0, 1.5 };
    float z[] = { 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 };

    //example
    for (i = 0; i < 7; i++)
    {
        z[i] += step;
        x[i] += 2 * step * part;
    }

    // the attribute vertexOrder=2 forces the Renderer to show the polygon two-site lighted
    char *ANames[] = { "vertexOrder", "PART" };
    char *AValues[2];
    AValues[0] = "2";
    AValues[1] = new char[10];
    sprintf(AValues[1], "%d", part);

    covWritePOLYGN(fd, 2, pl, 9, cl, 7, x, y, z, ANames, AValues, 2);
}
