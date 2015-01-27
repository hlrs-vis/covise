/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include <covise/covise.h>

#include "File09.h"
#include "istreamBLK.h"

#include <sys/types.h>
#include <sys/stat.h>

#undef VERBOSE

using namespace covise;

File09::File09(int fd)
    : input(fd, 8192)

{
#ifdef VERBOSE
    cerr << "File09::File09" << endl;
#endif
    struct stat status;
    if (fstat(fd, &status) < 0)
    {
        ncell = 0;
        d_inode = 0;
        d_device = 0;
        return;
    }
    d_inode = status.st_ino;
    d_device = status.st_dev;

#ifdef VERBOSE
    cerr << "stat : inode=" << status.st_ino
         << " device=" << status.st_dev
         << endl;
#endif

    // Header
    if (input.read((void *)&iter, sizeof(Header)) < 0)
    {
        cerr << "could not read in File09" << endl;
        ncell = 0;
        return;
    }

    // check: we might be byte-swapping
    if (lvers < 2210 || lvers >= 36000)
    {
        input.rewind();
        input.setByteSwap(1);
        if ((input.read((void *)&iter, sizeof(Header)) < 0)
            || (lvers < 2210) || (lvers > 36000))
        {
            ncell = 0;
            return;
        }
#ifdef VERBOSE
        cerr << "Determined BYTE-SWAPPING" << endl;
#endif
    }

#ifdef VERBOSE
    cerr << "Read Header: Star version " << lvers << endl;
    cerr << "ITER   = " << iter << endl;
    cerr << "TIME   = " << time << endl;
    cerr << "NCELL  = " << ncell << " Cells" << endl;
    cerr << "NNODE  = " << nnode << " Vertices" << endl;
    cerr << "NBC    = " << nbc << " Boundaries" << endl;
    cerr << "STEADY = " << steady << endl;
    cerr << "LMVGRD = " << lmvgrd << endl;
#endif

    if (lvers < 0)
    {
        cerr << "Trying to read 29 file with 09 reader"
             << endl;
        ncell = 0;
    }
}

ChoiceList *File09::get_choice(const char **scalarName, int maxList) const
{
    ChoiceList *choice = new ChoiceList("---", 0);

    /// Data fields: pre-2264

    int i;
    if (lvers < 2264)
    {
        choice->add("Velocity", 1);
        choice->add("V-Magnitude", 2);
        choice->add("U", 3);
        choice->add("V", 4);
        choice->add("W", 5);
        choice->add("P", 6);
        choice->add("TE", 7);
        choice->add("ED", 8);
        choice->add("T-Vis", 9);
        choice->add("T", 10);
        choice->add("Den", 11);
        choice->add("Lam-Vis", 12);
        choice->add("CP", 13);
        choice->add("Cond", 14);
    }

    /// Data fields: post-2264

    //  =============== LSTAR field ===============
    //  1-BOUNDARIES   6-ED    11-CP    16-GVLO1 21-FLUXI  26-TWPHL
    //  2-6 FLUXES     7-T     12-COND  17-RSMX1 22-FLUXJ  27-MOVGR
    //  3-3 VELOCITIES 8-TVIS  13-DENDP 18-PP    23-FB
    //  4-PRESSURE     9-DEN   14-DENO  19-T4PAT 24-FBSI
    //  5-TE          10-LVIS  15-SNRI1 20-HRPAT 25-TWOLYR

    else
    {
        if (lstar[3])
        {
            choice->add("Velocity", 1);
            choice->add("V-Magnitude", 2);
            choice->add("U", 3);
            choice->add("V", 4);
            choice->add("W", 5);
        }
        if (lstar[4])
            choice->add("P", 6);
        if (lstar[5])
            choice->add("TE", 7);
        if (lstar[6])
            choice->add("ED", 8);
        if (lstar[8])
            choice->add("T-Vis", 9);
        if (lstar[7])
            choice->add("T", 10);
        if (lstar[9])
            choice->add("Den", 11);
        if (lstar[10])
            choice->add("Lam-Vis", 12);
        if (lstar[11])
            choice->add("CP", 13);
        if (lstar[12])
            choice->add("Cond", 14);
    }

    /// Scalar values

    for (i = 1; i <= numcon; i++)
    {
        if ((i <= maxList) && (*(scalarName[i])))
        {
            choice->add(scalarName[i], 14 + i);
        }
        else
        {
            char buffer[16];
            sprintf(buffer, "Scalar_%d", i);
            choice->add(buffer, SCALAR - 1 + i);
        }
    }

    return choice;
}

int File09::readField(int fieldNo, float *f1, float *f2, float *f3)
{
    int i;

    //  =============== LSTAR field ===============
    //  1-BOUNDARIES   6-ED    11-CP    16-GVLO1 21-FLUXI  26-TWPHL
    //  2-6 FLUXES     7-T     12-COND  17-RSMX1 22-FLUXJ  27-MOVGR
    //  3-3 VELOCITIES 8-TVIS  13-DENDP 18-PP    23-FB
    //  4-PRESSURE     9-DEN   14-DENO  19-T4PAT 24-FBSI
    //  5-TE          10-LVIS  15-SNRI1 20-HRPAT 25-TWOLYR

    // Which LSTAR variable to ask for existance of field i
    // only vers>=2264, not for scalars
    // -1 fuer FORTRAN

    static const int lstarIdx[] = { 0, 3, 3, 3, 3, 3, 4, 5, 6, 8, 7, 9, 10, 11, 12 };
    //                           0 1 2 3 4 5 6 7 8 9 10 11 12 13 14

    // Velocities

    input.seekBlock(2);
    input.skipFloatBlk(nbc); // Skip boundary

    if ((lvers < 2264)
        || ((lvers >= 2264) && (lstar[2])))
    {
        input.skipFloatBlk(ncell); //      Flux 1
        input.skipFloatBlk(ncell);
        input.skipFloatBlk(ncell);
        input.skipFloatBlk(ncell);
        input.skipFloatBlk(ncell);
        input.skipFloatBlk(ncell); //      Flux 6
    }

    // velocity requested but not there
    if ((fieldNo < 6) && (lvers >= 2264) && (lstar[3] == 0))
        return -1;

    // if velocity saved: read or skip
    if ((lstar[3]) || (lvers < 2264))
    {
        // read veloity vector
        if (fieldNo == VELOCITY)
        {
            input.skipFloat(nbc + 1);
            input.read(f1, ncell);
            input.skipFloat(nbc + 1);
            input.read(f2, ncell);
            input.skipFloat(nbc + 1);
            input.read(f3, ncell);
            if (input.fail())
                return -1;
            else
                return ncell;
        }

        // read velocity magnitude
        else if (fieldNo == VMAG)
        {
            float *tmp = new float[ncell];

            input.skipFloat(nbc + 1);
            input.read(tmp, ncell);
            for (i = 0; i < ncell; i++)
                f1[i] = tmp[i] * tmp[i];

            input.skipFloat(nbc + 1);
            input.read(tmp, ncell);
            for (i = 0; i < ncell; i++)
                f1[i] += tmp[i] * tmp[i];

            input.skipFloat(nbc + 1);
            input.read(tmp, ncell);
            for (i = 0; i < ncell; i++)
                f1[i] += tmp[i] * tmp[i];

            for (i = 0; i < ncell; i++)
#ifdef __sgi
                f1[i] = fsqrt(f1[i]);
#else
                f1[i] = sqrt(tmp[i]);
#endif
            delete[] tmp;
            if (input.fail())
                return -1;
            else
                return ncell;
        }

        // read any single velocity component
        else if (fieldNo <= W)
        {
            for (i = 0; i < fieldNo - U; i++)
                input.skipFloatBlk(ncell + nbc + 1);
            input.skipFloat(nbc + 1);
            input.read(f1, ncell);
            if (input.fail())
                return -1;
            else
                return ncell;
        }

        // skip all components
        else
        {
            input.skipFloatBlk(ncell + nbc + 1);
            input.skipFloatBlk(ncell + nbc + 1);
            input.skipFloatBlk(ncell + nbc + 1);
        }
    }

    // Pressure: Always saved as a DOUBLE field

    // if pressure saved: read or skip
    if ((lstar[4]) || (lvers < 2264))
    {
        if (fieldNo == PRESSURE)
        {
            input.skipDouble(nbc + 1);
            double *tmp = new double[ncell];
            input.read(tmp, ncell);
            double *tmpPtr = tmp;
            float *f1Ptr = f1;
            for (i = 0; i < ncell; i++)
            {
                *f1Ptr = (float)*tmpPtr;
                f1Ptr++;
                tmpPtr++;
            }
            delete[] tmp;
            if (input.fail())
                return -1;
            else
                return ncell;
        }
        else
            input.skipDoubleBlk(ncell + nbc + 1);
    }

    ////// other fields: all single precision and scalar vars //////

    int actField = TE;

#ifdef VERBOSE
    static const char *const fName[] = { // 7 8
                                         "none", "VELOCITY", "VMAG", "U", "V", "W", "PRESSURE", "TE", "ED",
                                         // 15
                                         "TVIS", "TEMPERATURE", "DENSITY", "LAMVIS", "CP", "COND", "SCALAR"
    };
#endif
    //cerr << " SCALAR = " << SCALAR << endl;
    while ((actField < fieldNo) && (actField < SCALAR - 1))
    {
        if ((lvers < 2264)
            || ((lvers >= 2264) && (lstar[lstarIdx[actField]])))
        {
            input.skipFloatBlk(ncell + nbc + 1);
#ifdef VERBOSE
            cerr << "ignored block: "
                 << "fieldNo=" << fieldNo
                 << "  actField=" << actField
                 << "[" << fName[actField] << "]"
                 << "LSTAR(" << lstarIdx[actField] << ")"
                 << endl;
        }
        else
        {
            cerr << "skipped block: "
                 << "fieldNo=" << fieldNo
                 << "  actField=" << actField
                 << "[" << fName[actField] << "]"
                 << " LSTAR(" << lstarIdx[actField] << ")"
                 << endl;
#endif
        }
        actField++;
    }

    if (actField == fieldNo) // reached my target field yet

        if ((lvers < 2264)
            || ((lvers >= 2264) && (lstar[lstarIdx[actField]])))
        {

            input.skipFloat(nbc + 1);
            input.read(f1, ncell);
            if (input.fail())
                return -1;
            else
                return ncell;
        }

        else

            return -1;

    else // so it is a scalar...
    {

        // cerr << "Scalar" << endl;

        fieldNo -= SCALAR; // now, it's the scalar # (1..x)
        if (fieldNo > numcon)
        {
            return -1;
        }

        for (i = 1; i < fieldNo; i++) // skip everything before mine..
            input.skipFloatBlk(ncell + nbc + 1);

        input.skipFloat(nbc + 1);
        input.read(f1, ncell);

        if (input.fail())
            return -1;
        else
            return ncell;
    }
}

File09::~File09()
{
}

int File09::isFile(const char *filename)
{
    struct stat status;
    if (stat(filename, &status) < 0)
        return -1;

    if (d_inode == status.st_ino
        && d_device == status.st_dev)
        return 1;
    else
        return 0;
}
