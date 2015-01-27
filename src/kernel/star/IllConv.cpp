/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "IllConv.h"
#include <covise/covise.h>

#undef VERBOSE

using namespace covise;

/// ----- Prevent auto-generated functions by assert -------

/// Copy-Constructor: NOT IMPLEMENTED
IllConv::IllConv(const IllConv &)
{
    assert(0);
}

/// Assignment operator: NOT  IMPLEMENTED
IllConv &IllConv::operator=(const IllConv &)
{
    assert(0);
    return *this;
}

/// ----- Never forget the Destructor !! -------

IllConv::~IllConv()
{
}

const IllConv::ConvertILL IllConv::s_convILL[4096] = {
#include "IllSplit.inc"
};

/// Default constructor: create translation table
IllConv::IllConv()
{
// using static table: no initialisation code here

#ifdef VERBOSE
    int i;
    int maxParts = 0;
    for (i = 0; i < 4096; i++)
        if (s_convILL[i].numParts > maxParts)
            maxParts = s_convILL[i].numParts;
    cout << "IllConv::maxParts="
         << maxParts
         << endl;
    cout << "sizeof(IllConv::s_convILL) ="
         << sizeof(IllConv::s_convILL)
         << endl;
#endif
}

/////////////////////////////////////////////////////////////////////////////

void IllConv::convertIll(StarModelFile::CellTabEntry *&cellTab,
                         StarModelFile::CellTypeEntry *cellType,
                         int *&convertMap,
                         int &numElem, int mxtb,
                         void (*dumper)(const char *))
{
    (void)cellType;
    int cell, numNewCell = 0;

    // create a list with ILL types
    unsigned int *ctype = new unsigned int[numElem];

    // if no converMap given, create one
    if (!convertMap)
    {
        convertMap = new int[numElem];
        for (cell = 0; cell < numElem; cell++)
            convertMap[cell] = cell;
    }

    FILE *debug = NULL;
    if (getenv("VERBOSE_STARLIB_ILL"))
        debug = fopen("zzILL", "w");

    for (cell = 0; cell < numElem; cell++)
    {
        if (cellTab[cell].ictID > 0 && cellTab[cell].ictID <= mxtb)
        //       && (cellType[cellTab[cell].ictID-1].ctype<3)

        {
            int &v0 = cellTab[cell].vertex[0];
            int &v1 = cellTab[cell].vertex[1];
            int &v2 = cellTab[cell].vertex[2];
            int &v3 = cellTab[cell].vertex[3];
            int &v4 = cellTab[cell].vertex[4];
            int &v5 = cellTab[cell].vertex[5];
            int &v6 = cellTab[cell].vertex[6];
            int &v7 = cellTab[cell].vertex[7];

            ctype[cell] = ((v3 == v7) ? 2048 : 0)
                          | ((v2 == v6) ? 1024 : 0)
                          | ((v1 == v5) ? 512 : 0)
                          | ((v0 == v4) ? 256 : 0)
                          | ((v7 == v4) ? 128 : 0)
                          | ((v6 == v7) ? 64 : 0)
                          | ((v5 == v6) ? 32 : 0)
                          | ((v4 == v5) ? 16 : 0)
                          | ((v3 == v0) ? 8 : 0)
                          | ((v2 == v3) ? 4 : 0)
                          | ((v1 == v2) ? 2 : 0)
                          | ((v0 == v1) ? 1 : 0);

            // make sure we know this type
            if (s_convILL[ctype[cell]].numParts >= 0)
            {
                numNewCell += s_convILL[ctype[cell]].numParts;
            }
            else
            {
                /*
            // unknown types pass by
            cout << "Unknown: ";
            unsigned int ct = ctype[cell];
            if ( ct & 2048 ) cout << " 3-7";
            if ( ct & 1024 ) cout << " 2-6";
            if ( ct &  512 ) cout << " 1-5";
            if ( ct &  256 ) cout << " 0-4";
            if ( ct &  128 ) cout << " 4-7";
            if ( ct &   64 ) cout << " 6-7";
            if ( ct &   32 ) cout << " 5-6";
            if ( ct &   16 ) cout << " 4-5";
            if ( ct &    8 ) cout << " 0-3";
            if ( ct &    4 ) cout << " 2-3";
            if ( ct &    2 ) cout << " 1-2";
            if ( ct &    1 ) cout << " 0-1";
            cout << endl;
            */
                ctype[cell] = 0;
                numNewCell++; // we'll copy this cell
            }
        }
        else
            ctype[cell] = UNUSED;

        if (cell % 500000 == 499999)
        {
            char tick[512];
            sprintf(tick, "processed %d cells", cell + 1);
            dumper(tick);
        }
    }

    StarModelFile::CellTabEntry *newCell
        = new StarModelFile::CellTabEntry[numNewCell];

    int *newToOldCell = new int[numNewCell];

    numNewCell = 0;

    /// Convert all elements
    for (cell = 0; cell < numElem; cell++)
    {
        // standard cell or unknown
        if ((ctype[cell] == 0 // hexa
             || ctype[cell] == 0xf0 // pyra
             || ctype[cell] == 0xf4 // tetra
             || ctype[cell] == 0x44 // prism
             || s_convILL[ctype[cell]].numParts == -1)

            && ctype[cell] != UNUSED)
        {
            newCell[numNewCell] = cellTab[cell];
            newToOldCell[numNewCell] = convertMap[cell];
            numNewCell++;
        }

        // ILL
        else if (ctype[cell] != UNUSED)
        {
            const int *cellVert = cellTab[cell].vertex;
            const ConvertILL &convert = s_convILL[ctype[cell]];
            int iPart;

            if (debug)
            {
                unsigned int ct = ctype[cell];
                fprintf(debug,
                        "\n From: #%-7d: %7d %7d %7d %7d %7d %7d %7d %7d\n", cell,
                        cellVert[0], cellVert[1], cellVert[2], cellVert[3],
                        cellVert[4], cellVert[5], cellVert[6], cellVert[7]);

                fprintf(debug, " Type= %d", ct);
                if (ct & 2048)
                    fprintf(debug, " 3-7");
                if (ct & 1024)
                    fprintf(debug, " 2-6");
                if (ct & 512)
                    fprintf(debug, " 1-5");
                if (ct & 256)
                    fprintf(debug, " 0-4");
                if (ct & 128)
                    fprintf(debug, " 4-7");
                if (ct & 64)
                    fprintf(debug, " 6-7");
                if (ct & 32)
                    fprintf(debug, " 5-6");
                if (ct & 16)
                    fprintf(debug, " 4-5");
                if (ct & 8)
                    fprintf(debug, " 0-3");
                if (ct & 4)
                    fprintf(debug, " 2-3");
                if (ct & 2)
                    fprintf(debug, " 1-2");
                if (ct & 1)
                    fprintf(debug, " 0-1");
                fprintf(debug, "\n ----> %d Parts:\n", convert.numParts);
                if (convert.numParts == 0)
                    fprintf(debug, " ===> removed before [%d]\n", numNewCell);
            }

            // loop over ILL separation parts
            for (iPart = 0; iPart < convert.numParts; iPart++)
            {
                const char *CXX = convert.conv[iPart];

                // loop over part's vertices
                for (int iVert = 0; iVert < 8; iVert++)
                    newCell[numNewCell].vertex[iVert] = cellVert[int(CXX[iVert])];

                // set the cell type
                newCell[numNewCell].ictID = cellTab[cell].ictID;

                newToOldCell[numNewCell] = convertMap[cell];

                if (debug)
                {
                    fprintf(debug,
                            "       %7d : %7d %7d %7d %7d %7d %7d %7d %7d\n",
                            numNewCell,
                            newCell[numNewCell].vertex[0],
                            newCell[numNewCell].vertex[1],
                            newCell[numNewCell].vertex[2],
                            newCell[numNewCell].vertex[3],
                            newCell[numNewCell].vertex[4],
                            newCell[numNewCell].vertex[5],
                            newCell[numNewCell].vertex[6],
                            newCell[numNewCell].vertex[7]);
                }

                numNewCell++;
            }

            if (debug)
                fprintf(debug, "---\n");
        }
    }

    delete[] ctype;

    if (debug)
        fclose(debug);

    // set new values and delete old cell table
    numElem = numNewCell;
    delete[] cellTab;
    cellTab = newCell;

    delete[] convertMap;
    convertMap = newToOldCell;
    return;
}
