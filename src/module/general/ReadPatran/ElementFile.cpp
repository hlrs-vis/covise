/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "ElementFile.h"
#include "istreamFTN.h"
#include <util/coviseCompat.h>

ElementFile::ElementFile(int fd)
{
    istreamFTN input(fd);
    if (input.fail())
    {
        numlines = 0;
        return;
    }

    int i, j;
    int dataSize;
    int n_bytes;

    // Read Record 1
    input.readFTN(&header, sizeof(Record1));
    header.title[319] = '\0';

    if (header.nwidth > EMAXCOL)
    {
        cerr << "illegal: cannot read because EMAXCOL Error"
             << endl;
        numlines = 0;
        return;
    }

    // Read Record 2 and 3
    input.readFTN(subtitle1, 80 * sizeof(int));
    input.readFTN(subtitle2, 80 * sizeof(int));
    subtitle1[319] = '\0';
    subtitle2[319] = '\0';

    // initialize Data Record Table
    for (i = 0; i < MAXLINES; i++)
        for (j = 0; j < EMAXCOL; j++)
            dataTab[i].data[j] = 0.0;

    // Read Data Record Table
    numlines = 0;
    dataSize = 2 * sizeof(int) + header.nwidth * sizeof(float);
    n_bytes = input.readFTN((void *)(dataTab), (size_t)dataSize);
    //  while (dataTab[numlines].id != EOF && dataTab[numlines].id != 0)
    while (n_bytes != -1)
    {
        numlines++;
        if (numlines > MAXLINES)
        {
            cerr << "illegal: cannot read because MAXLINES Error"
                 << endl;
            numlines = 0;
            return;
        }
        n_bytes = input.readFTN((void *)(dataTab + numlines), (size_t)dataSize);
        /*
      cerr << dataTab[numlines].data[0] << "\t" << dataTab[numlines].data[1] << "\t"
           << dataTab[numlines].data[2] << "\t" << dataTab[numlines].data[3] << "\t"
           << dataTab[numlines].data[4] << "\t" << dataTab[numlines].data[5] << "\t"
           << dataTab[numlines].data[6] << "\t" << dataTab[numlines].data[7] << "\t"
           << dataTab[numlines].data[8] << "\t" << dataTab[numlines].data[9] << "\t"
           << dataTab[numlines].data[10] << "\t" << dataTab[numlines].data[11] << "\t"
           << dataTab[numlines].data[12] << "\t" << dataTab[numlines].data[13] << "\t"
           << dataTab[numlines].data[14] << "\t" << endl;
      */
    }
}

int ElementFile::getDataField(int fieldFlag, const int *elementMapping,
                              int col, float *f1, const int diff, const int maxeid)
{
    int i;
    int elementId;
    int elemNo; // COVISE element number

    if (elementMapping == NULL)
        return -1;

    // Element stresses
    if (fieldFlag == ELEMENTSTRESS)
    {

        // initializing element data (also from missing elements)
        for (i = 0; i < numlines + diff; i++)
            f1[i] = 0.0;

        // ( unused elements allowed )
        for (i = 0; i < numlines; i++)
        {
            elementId = dataTab[i].id;
            // omit unused elements
            if (elementId >= 0 && elementId <= maxeid)
            {
                elemNo = elementMapping[elementId];
                if (elemNo != -1) // element used in FE model
                {
                    if (col < 1 || col > header.nwidth) // wrong column number
                        return -1;
                    f1[elemNo] = dataTab[i].data[col - 1];
                }
            }
        }
    }
    return 0;
}
