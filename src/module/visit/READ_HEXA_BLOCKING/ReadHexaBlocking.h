/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _READ_HEXA_BLOCKING_H
#define _READ_HEXA_BLOCKING_H
/**************************************************************************\ 
 **                                                   	      (C)2000 RUS **
 **                                                                        **
 ** Description:  Reader for ICEMCFD Hexa Blocking Files	                  **
 **                                                                        **
 **                                                                        **
 **                                                                        **
 **                                                                        **
 **                                                                        **
 ** Author: Gabor Duroska                                                  **
 ** Date: April 2000                                                       **
 **                                                                        **
\**************************************************************************/

#include <api/coModule.h>
using namespace covise;

class ReadHexaBlocking : public coModule
{

private:
    //  member functions
    virtual int compute();
    virtual void quit();
    virtual void postInst();

    int openFile();
    void readFile();
    int computeCube(const int);
    int computeWireframe(const int);
    coDoPolygons *createCube(char *, const int, const float);
    coDoPolygons *createCube(char *, const int, const int, const float);
    coDoLines *createWireframe(char *, const int, const float);
    coDoLines *createWireframe(char *, const int, const int, const float);

    //  member data
    const char *filename; // block file name
    FILE *fp;

    coFileBrowserParam *param_file;
    coOutputPort *outPort_polySet;
    coOutputPort *outPort_lineSet;

    int numNodes;
    int numGridElements;
    int numBlocks;

    coIntScalarParam *p_min, *p_max;
    coFloatSliderParam *p_scale;
    coChoiceParam *set_type;

public:
    ReadHexaBlocking();
    virtual ~ReadHexaBlocking();

    struct NodesTabEntry
    {
        float x;
        float y;
        float z;
        int number;
    } *nodesTab;

    struct GridElementsTabEntry
    {
        int output_block_nr;
        int material_id;
        int nodes[8];
    } *gridElementsTab;
};
#endif
