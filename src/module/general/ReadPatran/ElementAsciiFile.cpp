/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include <api/coModule.h>
#include "ElementAsciiFile.h"
#include "AscStream.h"
#include "istreamFTN.h"
#include <util/coviseCompat.h>

using namespace covise;

ElementAscFile::ElementAscFile(const char *filename, int column)
{
    ifstream input(filename);
    if (input.fail())
    {
        nnodes = 0;
        return;
    }

    AscStream input_line(&input);
    char buffer[MAXLINE];
    int i, j;
    int elem_pos;
    int skip_lines;

    // Read Record 1
    input_line.getline(buffer, MAXLINE); // Title

    input_line.getline(buffer, MAXLINE); // nwidth

    if (sscanf(buffer, "%d", &nwidth) != 1)
    {
        fprintf(stderr, "ElementAscFile::ElementAscFile: sscanf1 failed\n");
    }

    if (nwidth > 300)
        Covise::sendWarning("This is probably not an element results file. Please check.");
    skip_lines = (column % nwidth == 0) ? column / nwidth - 1 : column / nwidth;
    elem_pos = (column % nwidth == 0) ? (nwidth - 1) * 13 : (column % nwidth - 1) * 13;

    input_line.getline(buffer, MAXLINE); // SubTitle1
    input_line.getline(buffer, MAXLINE); // SubTitle2

    // initialize Data Record Table

    for (i = 0; i < ASCMAXLINES; i++)
        dataTab[i].data = 0.0;

    nnodes = 0;
    while (input_line.getline(buffer, MAXLINE))
    {

        // Read Data Record Table
        in.parseString(buffer, 0, 7, &(dataTab[nnodes].id));
        in.parseString(buffer, 8, 15, (int *)(void *)&(dataTab[nnodes].nshape));

        for (j = 1; j <= skip_lines; j++)
        {
            input_line.getline(buffer, MAXLINE);
        }
        input_line.getline(buffer, MAXLINE);
        in.parseString(buffer, elem_pos, elem_pos + 12, &(dataTab[nnodes].data));

        nnodes++;
        if (nnodes >= ASCMAXLINES)
        {
            cerr << "illegal: cannot read because MAXLINES Error"
                 << endl;
            nnodes = 0;
            return;
        }
    }
}

int ElementAscFile::getDataField(int fieldFlag, const int *elementMapping,
                                 int col, float *f1, const int diff, const int maxeid)
{
    (void)col;
    int i;
    int elementId;
    int elemNo; // COVISE element number

    if (elementMapping == NULL)
        return -1;

    // Element stresses
    if (fieldFlag == ELEMENTSTRESS)
    {

        // initializing element data (also from missing elements)
        for (i = 0; i < nnodes + diff; i++)
            f1[i] = 0.0;

        // ( unused elements allowed )
        for (i = 0; i < nnodes; i++)
        {
            elementId = dataTab[i].id;
            // omit unused elements
            if (elementId >= 0 && elementId <= maxeid)
            {
                elemNo = elementMapping[elementId];
                if (elemNo != -1) // element used in FE model
                {
                    // if (col<1 || col > nwidth)     // wrong column number
                    // return -1;
                    f1[elemNo] = dataTab[i].data;
                }
            }
        }
    }
    return 0;
}
