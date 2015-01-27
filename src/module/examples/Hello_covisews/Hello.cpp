/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// ++                                                        (C)2000 RUS  ++
// ++ Description: "Hello, world!" in COVISE API                          ++
// ++                                                                     ++
// ++ Author:                                                             ++
// ++                                                                     ++
// ++                            Andreas Werner                           ++
// ++               Computer Center University of Stuttgart               ++
// ++                            Allmandring 30                           ++
// ++                           70550 Stuttgart                           ++
// ++                                                                     ++
// ++ Date:  10.01.2000  V2.0                                             ++
// ++**********************************************************************/

// this includes our own class's headers
#include "Hello.h"

// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// ++++
// ++++  Constructor : This will set up module port structure
// ++++
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
Hello::Hello(int argc, char *argv[])
    : coModule(argc, argv, "Hello, world! program")
{
    // no parameters, no ports...
    p_text = addStringParam("Output_Text", "Text to output");
    p_text->setValue("Hello!");

    p_showText = addBooleanParam("Show_Text", "Show Text?");
    p_showText->setValue(true);

    p_outLines = addOutputPort("linesOut", "Lines", "output lines");
}

void Hello::param(const char *paramName, bool)
{
    if (strcmp(paramName, "Show_Text") == 0)
    {
        if (p_showText->getValue())
        {
            p_text->show();
        }
        else
        {
            p_text->hide();
        }
    }
}

// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// ++++
// ++++  compute() is called once for every EXECUTE message
/// ++++
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

int Hello::compute(const char *port)
{
    (void)port;
    sendInfo(p_text->getValue());

    /*   
   // a simple square: 4 nodes, 5 corners, 1 line
   coDoLines *myLines = new coDoLines(p_outLines->getObjName(),4,5,1);
   
   float *x;
   float *y;
   float *z;
   int *cornerlist;
   int *linelist;
   
   myLines->getAddresses(&x,&y,&z,&cornerlist,&linelist);
          
   cornerlist[0]=0;
   cornerlist[1]=1;
   cornerlist[2]=2;
   cornerlist[3]=3;
   cornerlist[4]=0;
                     
   linelist[0]=0;

   x[0]=0.;
   y[0]=0.;
   z[0]=0.;
   
   x[1]=1.;
   y[1]=0.;
   z[1]=0.;

   x[2]=1.;
   y[2]=1.;
   z[2]=0.;

   x[3]=0.;
   y[3]=1.;
   z[3]=0.;
*/

    // HELLO: 22 nodes, 26 corners, 8 lines
    coDoLines *myLines = new coDoLines(p_outLines->getObjName(), 22, 26, 8);

    float *x;
    float *y;
    float *z;
    int *cornerlist;
    int *linelist;

    myLines->getAddresses(&x, &y, &z, &cornerlist, &linelist);

    // The "H"
    // 4    5
    // |    |
    // 2----3
    // |    |
    // 0    1

    // 3 lines: 0-2-4, 2-3, 1-3-5
    linelist[0] = 0;
    cornerlist[0] = 0;
    cornerlist[1] = 2;
    cornerlist[2] = 4;

    linelist[1] = 3;
    cornerlist[3] = 2;
    cornerlist[4] = 3;

    linelist[2] = 5;
    cornerlist[5] = 1;
    cornerlist[6] = 3;
    cornerlist[7] = 5;

    // The "E"
    // 10---11
    // |
    // 8----9
    // |
    // 6----7

    // 2 lines
    linelist[3] = 8;
    cornerlist[8] = 7;
    cornerlist[9] = 6;
    cornerlist[10] = 8;
    cornerlist[11] = 10;
    cornerlist[12] = 11;

    linelist[4] = 13;
    cornerlist[13] = 8;
    cornerlist[14] = 9;

    // The first "L"
    // 14
    // |
    // |
    // |
    // 12----13

    // one line
    linelist[5] = 15;
    cornerlist[15] = 13;
    cornerlist[16] = 12;
    cornerlist[17] = 14;

    // The second"L"
    // 17
    // |
    // |
    // |
    // 15----16

    // one line
    linelist[6] = 18;
    cornerlist[18] = 16;
    cornerlist[19] = 15;
    cornerlist[20] = 17;

    // The "O"
    // 20----21
    // |      |
    // |      |
    // |      |
    // 18----19

    // 1 line
    linelist[7] = 21;
    cornerlist[21] = 18;
    cornerlist[22] = 20;
    cornerlist[23] = 21;
    cornerlist[24] = 19;
    cornerlist[25] = 18;

    // alltogether, we have:
    // 26 corners
    // 22 nodes
    // 8 lines

    // we need the coordinates ...
    x[0] = 0.;
    y[0] = 0.;
    x[1] = 1.;
    y[1] = 0.;
    x[2] = 0.;
    y[2] = 1.;
    x[3] = 1.;
    y[3] = 1.;
    x[4] = 0.;
    y[4] = 2.;
    x[5] = 1.;
    y[5] = 2.;

    x[6] = 1.5;
    y[6] = 0.;
    x[7] = 2.5;
    y[7] = 0.;
    x[8] = 1.5;
    y[8] = 1.;
    x[9] = 2.5;
    y[9] = 1.;
    x[10] = 1.5;
    y[10] = 2.;
    x[11] = 2.5;
    y[11] = 2.;

    x[12] = 3.;
    y[12] = 0.;
    x[13] = 4.;
    y[13] = 0.;
    x[14] = 3.;
    y[14] = 2.;

    x[15] = 4.5;
    y[15] = 0.;
    x[16] = 5.5;
    y[16] = 0.;
    x[17] = 4.5;
    y[17] = 2.;

    x[18] = 6.;
    y[18] = 0.;
    x[19] = 7.;
    y[19] = 0.;
    x[20] = 6.;
    y[20] = 2.;
    x[21] = 7.;
    y[21] = 2.;

    // we stay in z=0 plane
    for (int i = 0; i < 22; i++)
    {
        z[i] = 0.;
    }

    p_outLines->setCurrentObject(myLines);

    return SUCCESS;
}

MODULE_MAIN(Examples, Hello)
