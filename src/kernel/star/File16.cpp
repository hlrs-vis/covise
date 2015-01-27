/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */


#include <covise/covise.h>
#include <ctype.h>

#include "File16.h"
#include "istreamFTN.h"
#include "SammConv.h"
#include "IllConv.h"

#include <sys/stat.h>

#ifdef _WIN32
#include <io.h>
#include <fcntl.h>
#endif

#undef VERBOSE
#undef VERBOSE_BLOCKS
#undef VERBOSE_CPMATCH
#undef VERBOSE_REGI
#undef VERBOSE_SAMMIDX
#undef VERBOSE_SAMMTAB
#undef VERBOSE_CELLTAB

#define DO_SAMM
#undef VERBOSE_SAMM

#undef VERBOSE_ILL
#define DO_ILL

//// Corrected for V3000     21-23.05.97
//// Corrected for V3100B    02/2001

#ifndef _MSC_VER
inline int max(int i, int j) { return (i > j) ? i : j; }
inline int max(int i, int j, int k) { return max(i, max(j, k)); }
#endif

namespace covise
{
const int File16::numVert[8] = { 0, 0, 0, 0, 4, 5, 6, 8 };
}

using namespace covise;

File16::File16(int fd, void (*dumpFunct)(const char *))
    :

    oldOfNewCell(NULL)
    ,

    covToStar(NULL)
    , covToPro(NULL)
    ,

    dumper((dumpFunct) ? dumpFunct : (File16::printStderr))
    , cellTab(NULL)
    , bounTab(NULL)
    , regionType(NULL)
    , vertexTab(NULL)
    , cellType(NULL)
    , cp22(NULL)
    , cp23(NULL)
    , regionSize(NULL)
    , cellShapeArr(NULL)
{

    struct stat status;
    if (fstat(fd, &status) < 0)
    {
        maxn = 0;
        d_inode = 0;
        d_device = 0;
        return;
    }
    d_inode = status.st_ino;
    d_device = status.st_dev;

    /// read 1st word to determine byte order
    int first;
    ssize_t retval;
    retval = read(fd, &first, sizeof(int));
    if (retval == -1)
    {
        std::cerr << "File16::File16: read failed" << std::endl;
        return;
    }
    lseek(fd, 0, SEEK_SET);

    // open our "fortran" files
    istreamFTN input(fd);

    // set byte order if necessary
    if (first < 0 || first > 1000)
        input.setByteSwap(1);

    cells_used = NULL;
    if (input.fail())
    {
        maxn = 0;
        return;
    }

    int i;

////////////////////////////////////////////////////////

#ifdef VERBOSE_BLOCKS
    cerr << "======= Base Header ======= "
         << input.getActualBlockNo()
         << endl;
#endif
    if (input.readFTN_BS((void *)&maxn, sizeof(Header1)) < 0)
    {
        maxn = 0;
        return;
    }

#ifdef VERBOSE
    cerr << " JVERS = " << jvers << endl;
    cerr << " MAXN  = " << maxn << endl;
    cerr << " MAXE  = " << maxe << endl;
    cerr << " MAXR  = " << maxr << endl;
    cerr << " MAXB  = " << maxb << endl;
#endif

////////////////////////////////////////////////////////

#ifdef VERBOSE_BLOCKS
    cerr << "======= TITLE ======= "
         << input.getActualBlockNo()
         << endl;
#endif
    input.readFTN(&title, sizeof(title));
    title.main[79] = '\0';
    title.sub1[79] = '\0';
    title.sub2[79] = '\0';
    i = 78;
    while ((i >= 0) && (title.main[i] == ' '))
        title.main[i--] = '\0';
    i = 78;
    while ((i >= 0) && (title.sub1[i] == ' '))
        title.sub1[i--] = '\0';
    i = 78;
    while ((i >= 0) && (title.sub2[i] == ' '))
        title.sub2[i--] = '\0';

    if (title.main[0])
        dumper(title.main);
    else
        dumper("Reading Star-CD model file");

#ifdef VERBOSE
    cerr << title.main << endl;
#endif

    ////////////////////////////////////////////////////////

    if (jvers < 2300) // ======================= v 2.2
    {
#ifdef VERBOSE_BLOCKS
        cerr << "======= UNKNOWN v2.2 skip 9 Blocks ======= "
             << input.getActualBlockNo()
             << endl;
#endif
        input.skipBlocks(8); // < 2.300
    }
    else if (jvers < 3000) // ======================= v 2.3
    {
#ifdef VERBOSE_BLOCKS
        cerr << "======= UNKNOWN v2.3 skip 9 Blocks ======= "
             << input.getActualBlockNo()
             << endl;
#endif
        input.skipBlocks(9);
    }
    else if (jvers < 3100) // ======================= v 2.3
    {
        input.readFTN_BS(&ncydmf, sizeof(CycCoup));
#ifdef VERBOSE_BLOCKS
        cerr << "======= UNKNOWN v3.1 skip 9 Blocks ======= "
             << input.getActualBlockNo()
             << endl;
#endif
        input.skipBlocks(12);
    }
    else if (jvers < 3200) // ======================= v 2.3
    {
        input.readFTN_BS(&ncydmf, sizeof(CycCoup));
#ifdef VERBOSE_BLOCKS
        cerr << "======= UNKNOWN v3.1 skip 9 Blocks ======= "
             << input.getActualBlockNo()
             << endl;
#endif
        input.skipBlocks(12);
    }
    else // ======================= v 3.0
    {
#ifdef VERBOSE_BLOCKS
        cerr << "======= NCYDMF,NCPDMF (v3.x) ======= "
             << input.getActualBlockNo()
             << endl;
#endif
        input.readFTN_BS(&ncydmf, sizeof(CycCoup));
#ifdef VERBOSE
        cerr << "NCYDMF = " << ncydmf << " (max. Cells in cyclic sets)" << endl;
        cerr << "NCPDMF = " << ncpdmf << " (max. Cells in coupled sets)" << endl;
#endif

#ifdef VERBOSE_BLOCKS
        cerr << "======= UNKNOWN v3.x skip 12 Blocks ======= "
             << input.getActualBlockNo()
             << endl;
#endif
        input.skipBlocks(15); //3240
    }

    ////////////////////////////////////////////////////////

    // Read Cell table
    cellTab = new CellTabEntry[maxe];
    if (maxe > 500000)
        dumper("Reading Cell Table");

#ifdef VERBOSE_BLOCKS
    cerr << "======= Cell Table ======= "
         << input.getActualBlockNo()
         << endl;
#endif
    input.readFTNfloat((float *)cellTab, maxe * 9, 900);
    if (input.fail())
    {
        maxn = 0;
        return;
    }

#ifdef VERBOSE_CELLTAB
    {
        int i;
        FILE *debug = fopen("CELLTAB", "w");

        for (i = 0; i < maxe; i++)
        {
            fprintf(debug,
                    "       %7d : %7d %7d %7d %7d %7d %7d %7d %7d - %d\n", i,
                    cellTab[i].vertex[0], cellTab[i].vertex[1],
                    cellTab[i].vertex[2], cellTab[i].vertex[3],
                    cellTab[i].vertex[3], cellTab[i].vertex[5],
                    cellTab[i].vertex[4], cellTab[i].vertex[7],
                    cellTab[i].ictID);
        }
        fclose(debug);
    }
#endif

    ////////////////////////////////////////////////////////

    // Read Vertex Table
    vertexTab = new VertexTabEntry[maxn];
    if (maxe > 500000)
        dumper("Reading Vertex Table");

#ifdef VERBOSE_BLOCKS
    cerr << "======= Vertex Table ======= "
         << input.getActualBlockNo()
         << endl;
#endif
    input.readFTNfloat((float *)vertexTab, maxn * 3, 900);
    if (input.fail())
    {
        maxn = 0;
        return;
    }

    ////////////////////////////////////////////////////////

    // Boundary Definition
    bounTab = new BounTabEntry[maxb];
#ifdef VERBOSE_BLOCKS
    cerr << "======= Boundary Face List ======= "
         << input.getActualBlockNo()
         << endl;
#endif
    input.readFTNfloat((float *)bounTab, maxb * 6, 900);
    //bounTab = NULL;
    //input.skipBlocks(maxb,150); // skipped
    if (input.fail())
    {
        maxn = 0;
        return;
    }

////////////////////////////////////////////////////////

// Skip region Definitions
#ifdef VERBOSE_BLOCKS
    cerr << "======= Region Parameter List ======= "
         << input.getActualBlockNo()
         << endl;
#endif
    input.skipBlocks(maxr + 1, 16);
    if (input.fail())
    {
        maxn = 0;
        return;
    }

////////////////////////////////////////////////////////

#ifdef VERBOSE_BLOCKS
    cerr << "======= Region Type List ======= "
         << input.getActualBlockNo()
         << endl;
#endif
    regionType = new int[maxr + 1];
    if (jvers < 3100)
        input.readFTNint(regionType, maxr + 1, maxr + 1);
    //input.skipBlocks(maxr+1,maxr+1);
    else
        input.readFTNint(regionType, maxr + 1, 900);
    //input.skipBlocks(maxr+1,900);

    if (input.fail())
    {
        maxn = 0;
        return;
    }

    ////////////////////////////////////////////////////////

    if (jvers >= 3100)
    {
#ifdef VERBOSE_BLOCKS
        cerr << "======= Boundary region string fields ======= "
             << input.getActualBlockNo()
             << endl;
#endif
        input.skipBlocks(maxr + 1, 100);
        input.skipBlocks(maxr + 1, 100);
    }

////////////////////////////////////////////////////////

#ifdef VERBOSE_BLOCKS
    cerr << "======= Solver Settings ======= "
         << input.getActualBlockNo()
         << endl;
#endif
    input.skipBlocks(1);
    if (input.fail())
    {
        maxn = 0;
        return;
    }

////////////////////////////////////////////////////////

// LS[1..29],IS[29],LS[30],NPRSF[3],NPROBS   3 Elemente uebrig >=2.3
#ifdef VERBOSE_BLOCKS
    cerr << "======= LS-Block ======= "
         << input.getActualBlockNo()
         << endl;
#endif
    if (jvers < 2300)
        input.scanForSizeInt((int *)&LSrec, 133);
    else if (jvers < 3100)
        input.scanForSizeInt((int *)&LSrec, 136);
    else
        input.scanForSizeInt((int *)&LSrec, 139);

    // input.readFTNint((int*)&LSrec,133,99999);
    if (input.fail())
    {
        maxn = 0;
        return;
    }

    ////////////////////////////////////////////////////////

    // /*

    if (jvers < 2300)
    {
        if ((LSrec.nprsf[1])
            || (LSrec.nprsf[2])
            || (LSrec.nprsf[2]))
        {
#ifdef VERBOSE_BLOCKS
            cerr << "======= v2.3 max(ni,nj,nk) - Block ======= "
                 << input.getActualBlockNo()
                 << endl;
#endif
#ifndef _WIN32
            input.skipBlocks((max(ni, nj, nk) - 1) / 300 + 1);
#else
            int nijk = std::max(ni, nj);
            nijk = std::max(nk, nijk);
            input.skipBlocks((nijk - 1) / 300 + 1);
#endif
            if (input.fail())
            {
                maxn = 0;
                return;
            }
        }
        if (LSrec.nprobs > 0)
            cerr << "illegal: cannot read because MAXOBF Error"
                 << endl;
    }
    else
    {
        if (LSrec.nprobs > 0 && jvers < 3100)
        {
#ifdef VERBOSE_BLOCKS
            cerr << "======= MAXOBF V2.3 and above ======= "
                 << input.getActualBlockNo()
                 << endl;
#endif
            input.skipBlocks(1);
            if (input.fail())
            {
                maxn = 0;
                return;
            }
        }
    }
// */

////////////////////////////////////////////////////////
// Properties Information Block
#ifdef VERBOSE_BLOCKS
    cerr << "======= Property Info Block ======= "
         << input.getActualBlockNo()
         << endl;
#endif
    input.readFTN_BS(&propInfo, sizeof(PropInfo));

    //   if (jvers<3000)
    //      input.scanForSizeInt((int*)&propInfo,400);
    //   else
    //       input.scanForSizeInt((int*)&propInfo,496);

    if (input.fail())
    {
        maxn = 0;
        return;
    }

    numMaterials = 0;
    for (i = 0; i < 99; i++)
        if (propInfo.lmdef[i])
            numMaterials++;

/*
      if (jvers>=3040) {
   #ifdef VERBOSE_BLOCKS
          cerr << "======= INOX Info Block ======= "
              << input.getActualBlockNo()
              << endl;
   #endif
         int inox;
         input.readFTN(&inox,sizeof(int));
         if (input.fail()) { maxn=0; return; }
         if (inox) {
   #ifdef VERBOSE_BLOCKS
   cerr << "======= INOX Data ======= "
   << input.getActualBlockNo()
   << endl;
   #endif
   input.skipBlocks(2);
   }
   }

   ////////////////////////////////////////////////////////

   // Properties
   for (i=0;i<99;i++)
   if (propInfo.lmdef[i]) {
   #ifdef VERBOSE
   cerr << "Reading Material #" << i << endl;
   #endif
   if (jvers<=2200) {
   #ifdef VERBOSE_BLOCKS
   cerr << "======= v2.x Property #" <<i <<" ======= "
   << input.getActualBlockNo()
   << endl;
   #endif
   input.skipBlocks(1);
   if (input.fail()) { maxn=0; return; }
   }
   else if (jvers<3040) {
   #ifdef VERBOSE_BLOCKS
   cerr << "======= v3.000 Property #" <<i <<" ======= "
   << input.getActualBlockNo()
   << endl;
   #endif
   input.skipBlocks(2);
   if (input.fail()) { maxn=0; return; }
   }
   else {
   #ifdef VERBOSE_BLOCKS
   cerr << "======= v3.050 Property #" << i <<" ======= "
   << input.getActualBlockNo()
   << endl;
   #endif
   input.skipBlocks(3);
   if (input.fail()) { maxn=0; return; }
   }
   }
   */
////////////////////////////////////////////////////////

#ifdef VERBOSE_BLOCKS
    cerr << "======= ROTA ======= "
         << input.getActualBlockNo()
         << endl;
#endif
    //  if (jvers>2201)
    //    input.skipBlocks(1);

    input.skipForSize(693 * sizeof(int));

    if (input.fail())
    {
        maxn = 0;
        return;
    }

////////////////////////////////////////////////////////

#ifdef VERBOSE_BLOCKS
    cerr << "======= Two Layer ======= "
         << input.getActualBlockNo()
         << endl;
#endif
    if (jvers > 2201)
        input.skipBlocks(1);
    if (input.fail())
    {
        maxn = 0;
        return;
    }

    ////////////////////////////////////////////////////////

    if (jvers <= 2330) ///////// pre-3.000 Boundary List
    {

// NBND
#ifdef VERBOSE_BLOCKS
        cerr << "======= NBND Sizes ( vers <= 2.330 ) ======= "
             << input.getActualBlockNo()
             << endl;
#endif
        input.readFTN_BS(nbnd, 6 * sizeof(float));
        if (input.fail())
        {
            maxn = 0;
            return;
        }

// Boundary Data
#ifdef VERBOSE_BLOCKS
        cerr << "======= NBND Data ( vers <= 2.330 ) ======= "
             << input.getActualBlockNo()
             << endl;
#endif

#ifndef _WIN32
        input.skipBlocks((max(max(nbnd[0], nbnd[1], nbnd[2]),
                              max(nbnd[3], nbnd[4], nbnd[5]))
                          - 1) / 150 + 1);
#else
        int nijk = std::max(nbnd[0], nbnd[1]);
        nijk = std::max(nijk, nbnd[2]);
        int nijk2 = std::max(nbnd[3], nbnd[4]);
        nijk2 = std::max(nijk2, nbnd[5]);

        input.skipBlocks((std::max(nijk, nijk2) - 1) / 150 + 1);
#endif
        if (input.fail())
        {
            maxn = 0;
            return;
        }
    }

    ////////////////////////////////////////////////////////

    // NEWKEY, maxcy-Data, Post-Proc

    if (jvers <= 2300) // ======================  V2.2
    {
#ifdef VERBOSE_BLOCKS
        cerr << "======= V2.2 NKEYS ======= "
             << input.getActualBlockNo()
             << endl;
#endif
        input.skipBlocks(1);
        if (input.fail())
        {
            maxn = 0;
            return;
        }
#ifdef VERBOSE_BLOCKS
        cerr << "======= V2.2 cyclic ======="
             << input.getActualBlockNo()
             << endl;
#endif
        input.skipBlocks(maxcy, 450);
        if (input.fail())
        {
            maxn = 0;
            return;
        }
    }
    else if (jvers <= 2330) // ====================== V2.3
    {
#ifdef VERBOSE_BLOCKS
        cerr << "======= V2.3 NKEY ======= "
             << input.getActualBlockNo()
             << endl;
#endif
        input.skipBlocks(1);
        if (input.fail())
        {
            maxn = 0;
            return;
        }
#ifdef VERBOSE_BLOCKS
        cerr << "======= V2.3 cyclic ======= "
             << input.getActualBlockNo()
             << endl;
#endif
        input.skipBlocks(maxcy, 18);
        if (input.fail())
        {
            maxn = 0;
            return;
        }
    }
    else // ========================= V3.0
    {
#ifdef VERBOSE_BLOCKS
        cerr << "======= V3.0 NKEY======= "
             << input.getActualBlockNo()
             << endl;
#endif
        input.skipBlocks(1); // NKEY
        if (input.fail()) // LSWTCH(100),RSWTCH(100)
        {
            maxn = 0;
            return;
        }

#ifdef VERBOSE_BLOCKS
        cerr << "======= V3.0 LSWTCH(100),RSWTCH(100) ======= "
             << input.getActualBlockNo()
             << endl;
#endif
        input.skipBlocks(1); // NKEY
        if (input.fail()) // LSWTCH(100),RSWTCH(100)
        {
            maxn = 0;
            return;
        }

        int NN1 = 2048 / ncydmf; // Cyclic Data
#ifdef VERBOSE_BLOCKS
        if (maxcy)
            cerr << "======= V3.0 Cyclic Data ======= "
                 << input.getActualBlockNo()
                 << endl;
#endif
        input.skipBlocks(maxcy, NN1);
        if (input.fail())
        {
            maxn = 0;
            return;
        }
    }

////////////////////////////////////////////////////////

#ifdef VERBOSE_BLOCKS
    cerr << "======= Data Base sets ======= " // C DATA BASE SETS
         << input.getActualBlockNo()
         << endl;
#endif
    input.skipBlocks(1); // Data base sets
    if (input.fail())
    {
        maxn = 0;
        return;
    }

////////////////////////////////////////////////////////

#ifdef VERBOSE_BLOCKS
    cerr << "======= Cell selection ======= "
         << input.getActualBlockNo()
         << endl;
#endif
    input.skipBlocks(maxe, 900);
    if (input.fail())
    {
        maxn = 0;
        return;
    }

////////////////////////////////////////////////////////

#ifdef VERBOSE_BLOCKS
    cerr << "======= Vertex selection ======= "
         << input.getActualBlockNo()
         << endl;
#endif
    input.skipBlocks(maxn, 900);
    if (input.fail())
    {
        maxn = 0;
        return;
    }

////////////////////////////////////////////////////////

#ifdef VERBOSE_BLOCKS
    cerr << "======= Boundary selection ======= "
         << input.getActualBlockNo()
         << endl;
#endif
    input.skipBlocks(maxb, 900);
    if (input.fail())
    {
        maxn = 0;
        return;
    }

////////////////////////////////////////////////////////

#ifdef VERBOSE_BLOCKS
    if (maxs)
        cerr << "======= Spline selection ======= "
             << input.getActualBlockNo()
             << endl;
#endif
    input.skipBlocks(maxs, 900);
    if (input.fail())
    {
        maxn = 0;
        return;
    }

////////////////////////////////////////////////////////

#ifdef VERBOSE_BLOCKS
    if (mxbl)
        cerr << "======= Block selection ======= "
             << input.getActualBlockNo()
             << endl;
#endif
    input.skipBlocks(mxbl, 900);
    if (input.fail())
    {
        maxn = 0;
        return;
    }

////////////////////////////////////////////////////////

#ifdef VERBOSE_BLOCKS
    if (maxs)
        cerr << "======= Spline Data ======= "
             << input.getActualBlockNo()
             << endl;
#endif
    input.skipBlocks(maxs, 8);
    if (input.fail())
    {
        maxn = 0;
        return;
    }

////////////////////////////////////////////////////////

#ifdef VERBOSE_BLOCKS
    if (mxbl)
        cerr << "======= Block Data ======= "
             << input.getActualBlockNo()
             << endl;
#endif
    input.skipBlocks(mxbl, 31);
    if (input.fail())
    {
        maxn = 0;
        return;
    }

////////////////////////////////////////////////////////

#ifdef VERBOSE_BLOCKS
    cerr << "======= Read KEYS2 ======= "
         << input.getActualBlockNo()
         << endl;
#endif

    input.readFTN_BS((void *)&mxtb, sizeof(Header2));
    if (input.fail())
    {
        maxn = 0;
        return;
    }

#ifdef VERBOSE
    cout << "MXTB   = " << mxtb << endl;
    cerr << "LTURBI = " << lturbi << endl;
    cerr << "LTURBP = " << lturbp << endl;
    cerr << "SETADD = " << setadd << endl;
    cerr << "NSENS  = " << nsens << endl;
    cerr << "NPART  = " << npart << endl;
    cout << "SCALE8 = " << scale8 << endl;
    cerr << "MAXCP  = " << maxcp << endl;
    cerr << "LOC180 = " << loc180 << endl;
    cerr << "MVER   = " << mver << endl;
    cerr << "MAXSCL = " << maxscl << endl;
    cerr << "ISTYPE = " << istype << endl;
    cerr << "MXSTB  = " << mxstb << endl;
    cerr << "NUMCP  = " << numcp << endl;
    cerr << "IOPTBC = " << ioptbc << endl;
    cerr << "LTURBF = " << lturbf << endl;
    cerr << "LTURBT = " << lturbt << endl;
    cerr << "MAXCRS = " << maxcrs << endl;
    cerr << "MXSAM  = " << pbtol << endl;
#endif

    ////////////////////////////////////////////////////////

    struct
    {
        int mxcptb, icptac;
        float cpgtin, cpgtpl, cpgtan;
    } cpg;

    if (jvers >= 3040)
    {

#ifdef VERBOSE_BLOCKS
        cerr << "======= >V3.04 CPG-Block ======= "
             << input.getActualBlockNo()
             << endl;
#endif
        input.readFTN_BS((void *)&cpg, sizeof(cpg));

#ifdef VERBOSE
        cerr << "MXCPTB = " << cpg.mxcptb << endl;
#endif

        ////////////////////////////////////////////////////////

        if (maxcp > 0)
        {
#ifdef VERBOSE_BLOCKS
            cerr << "======= >V3.04 CPTB data ======= "
                 << input.getActualBlockNo()
                 << endl;
#endif
            input.skipBlocks(maxcp, 900);
        }
    }
    else
    {
        cpg.mxcptb = 0;
    }

    ////////////////////////////////////////////////////////

    int &mxsam = pbtol;

    ////////////////////////////////////////////////////////

    ///   SAMM

    if ((jvers > 2310) && (mxsam > 0))
    {
        sammTab = new SammTabEntry[mxsam];

#ifdef VERBOSE_BLOCKS
        cerr << "====== Unknown skip SAMM data ======"
             << input.getActualBlockNo()
             << endl;
#endif

        // SAMM indices: sammIdx[i] = cellNo of i'th SAMM

        int *sammIdx = new int[mxsam];
        input.readFTNint(sammIdx, mxsam, 2048);

#ifdef VERBOSE_SAMMIDX
        {
            FILE *debug = fopen("SAMMIDX", "w");
            for (i = 0; i < mxsam; i++)
                fprintf(debug, "%7d : %7d\n", i, sammIdx[i]);
            fclose(debug);
        }
#endif

        if (input.fail())
        {
            maxn = 0;
            return;
        }

#ifdef VERBOSE
        cerr << "Reading SAMM " << mxsam << " cells" << endl;
#endif

#ifdef VERBOSE_BLOCKS
        cerr << "====== SAMM cells ======"
             << input.getActualBlockNo()
             << endl;
#endif

        input.readFTNfloat((float *)sammTab, mxsam * 13, 1950);

        // mark SAMM definitions of inactive cells
        for (i = 0; i < mxsam; i++)
        {
            int cellIdx = sammIdx[i] - 1; // C counting in our tabs
            if (cellTab[cellIdx].ictID <= 0)
                sammTab[i].ictID = -1;
        }
        delete[] sammIdx; // don't need it any further

#ifdef VERBOSE_SAMMTAB
        {
            int i;
            FILE *debug = fopen("SAMMTAB", "w");

            for (i = 0; i < mxsam; i++)
            {
                fprintf(debug,
                        "       %7d : %7d %7d %7d %7d %7d %7d %7d %7d - %d\n", i,
                        sammTab[i].vertex[0], sammTab[i].vertex[1],
                        sammTab[i].vertex[2], sammTab[i].vertex[3],
                        sammTab[i].vertex[3], sammTab[i].vertex[5],
                        sammTab[i].vertex[4], sammTab[i].vertex[7],
                        sammTab[i].ictID);
                if (sammTab[i].vertex[8])
                    fprintf(debug,
                            "                 %7d %7d %7d %7d\n",
                            sammTab[i].vertex[8], sammTab[i].vertex[9],
                            sammTab[i].vertex[10], sammTab[i].vertex[11]);
            }

            fclose(debug);
        }
#endif

        if (input.fail())
        {
            maxn = 0;
            return;
        }
    }

    ////////////////////////////////////////////////////////

    // Cell Types
    cellType = new CellTypeEntry[mxtb];
#ifdef VERBOSE_BLOCKS
    cerr << "======= Cell Table ======= "
         << input.getActualBlockNo()
         << endl;
#endif
    input.readFTNint((int *)cellType, mxtb * 10, 900);
    if (input.fail())
    {
        maxn = 0;
        return;
    }

    ////////////////////////////////////////////////////////

    if (jvers >= 3060)
    {
#ifdef VERBOSE_BLOCKS
        cerr << "======= Cell Type names (>=3.100) ======= "
             << input.getActualBlockNo()
             << endl;
#endif
        input.skipBlocks((mxtb - 1) / 100 + 1); // 8000 char blocks, 80 char/name
    }
    else
    {
#ifdef VERBOSE_BLOCKS
        cerr << "======= Cell Types Names (<3.100) ======= "
             << input.getActualBlockNo()
             << endl;
#endif
        input.skipBlocks(1);
    }
    if (input.fail())
    {
        maxn = 0;
        return;
    }

////////////////////////////////////////////////////////

#ifdef VERBOSE_BLOCKS
    cerr << "======= Spline tables ======= "
         << input.getActualBlockNo()
         << endl;
#endif
    input.skipBlocks(5 * mxstb, 900);
    if (input.fail())
    {
        maxn = 0;
        return;
    }

    ////////////////////////////////////////////////////////

    if ((jvers >= 3040) && (cpg.mxcptb > 0))
    {
#ifdef VERBOSE_BLOCKS
        cerr << "======= v3.04 Couple tables ======= "
             << input.getActualBlockNo()
             << endl;
#endif
        input.skipBlocks(5 * cpg.mxcptb, 900); // ICPTAB(5*MXCPTB)
        input.skipBlocks(1); // CPBTOL(MXCPTB)

        if (jvers > 3060)
            input.skipBlocks(cpg.mxcptb, 100); // CPTNAME(MXCPTB)
    }

    ////////////////////////////////////////////////////////

    if (nsens > 0)
    {
#ifdef VERBOSE_BLOCKS
        cerr << "======= Probe data ======= "
             << input.getActualBlockNo()
             << endl;
#endif
        input.skipBlocks(2 * nsens, 900);
    }
    if (input.fail())
    {
        maxn = 0;
        return;
    }

    ////////////////////////////////////////////////////////

    if (npart > 0)
    {
#ifdef VERBOSE_BLOCKS
        cerr << "======= Particle Data ======= "
             << input.getActualBlockNo()
             << endl;
#endif
        if (jvers < 2237)
            input.skipBlocks(npart, 450);
        else
            input.skipBlocks(npart, 100);
    }
    if (input.fail())
    {
        maxn = 0;
        return;
    }

////////////////////////////////////////////////////////

#ifdef VERBOSE_BLOCKS
    cerr << "======= Porosity Data ======= "
         << input.getActualBlockNo()
         << endl;
#endif
    {
        input.skipBlocks(1);
    }
    //input.skipBlocks(1);
    if (input.fail())
    {
        maxn = 0;
        return;
    }

////////////////////////////////////////////////////////

#ifdef VERBOSE_BLOCKS
    cerr << "======= Coupled cell sets ======= "
         << input.getActualBlockNo()
         << endl;
#endif
    // Coupled Cells: V2.2=1+24  V2.3=1+50 V30=1+x, side info
    if (maxcp == 0)
    {
        cp22 = NULL;
        cp23 = NULL;
        cp30 = NULL;
    }
    else if (jvers >= 2325)
    {
        cp22 = NULL;
        cp23 = NULL;
        int buffer[2048];
        int nn1 = 2048 / ncpdmf;
        int blocks = maxcp / nn1 + 1;
        int cp = 0;
        int j;

        if (ncpdmf < MAX_CP)
        {
            cp30 = new CoupledCells30[maxcp];

            int block;
            for (block = 0; block < blocks; block++)
            {
                input.readFTNint(buffer, 2048, 2048);
                if (input.fail())
                {
                    maxn = 0;
                    return;
                }
                int *bufPtr = buffer;
                for (i = 0; ((i < nn1) && (cp < maxcp)); i++)
                {
                    cp30[cp].master = *bufPtr++;
                    cp30[cp].masterSide = (char)(cp30[cp].master % 10);
                    cp30[cp].master /= 10;
                    for (j = 0; j < ncpdmf - 1; j++)
                    {
                        cp30[cp].slave[j] = *bufPtr++;
                        cp30[cp].slaveSide[j] = (char)(cp30[cp].slave[j] % 10);
                        cp30[cp].slave[j] /= 10;
                    }
                    cp++;
                }
            }

/*****+ write debug file*/
#ifdef VERBOSE_CPMATCH
            FILE *fi = fopen("CP_MATCH", "w");
            for (i = 0; i < maxcp; i++)
            {
                if (cp30[i].master)
                {
                    fprintf(fi, "MASTER: %8d Side %1d\n",
                            cp30[i].master, cp30[i].masterSide);
                    for (j = 0; j < ncpdmf - 1; j++)
                        fprintf(fi, "    --> %8d Side %1d\n",
                                cp30[i].slave[j], cp30[i].slaveSide[j]);
                }
            }
            fclose(fi);
#endif
            /************/

            // compress list
            numRealCP = 0;
            for (i = 0; i < maxcp; i++)
            {
                if (cp30[i].master)
                {
                    cp30[numRealCP] = cp30[i];
                    numRealCP++;
                }
            }
        }
        else
        {
            cp30 = NULL;
            maxcp = 0; // ignore Couples in future
            input.skipBlocks(blocks);
            char buffer[1024];
            sprintf(buffer, "Unable to read CP matches: NCPDMF=%d too high", ncpdmf);
            dumper(buffer);
        }

    } // version v3.x

    else if (jvers >= 2300)
    {
        cp22 = NULL;
        cp23 = new CoupledCells23[maxcp];
        cp30 = NULL;
        input.readFTNint((int *)cp23, 51 * maxcp, 17 * 51);
        if (input.fail())
        {
            maxn = 0;
            return;
        }
    }
    else
    {
        cp22 = new CoupledCells22[maxcp];
        cp23 = NULL;
        cp30 = NULL;
        input.readFTNint((int *)cp22, 25 * maxcp, 36 * 25);
        if (input.fail())
        {
            maxn = 0;
            return;
        }
    }

    if (jvers > 3040)
    {

#ifdef VERBOSE_BLOCKS
        cerr << "======= Coupling types (>3040) ======= "
             << input.getActualBlockNo()
             << endl;
#endif
        input.skipBlocks(1); // ICPTYP(MAXCP)
    }

    ////////////////////////////////////////////////////////

    if (mver == 1)
    {
#ifdef VERBOSE_BLOCKS
        cerr << "======= ANSYS nodal rotations (vers > v3000) ======= "
             << input.getActualBlockNo()
             << endl;
        input.skipBlocks(maxn, 300);
#endif
    }

    ////////////////////////////////////////////////////////

    if (maxscl)
    {
#ifdef VERBOSE_BLOCKS
        cerr << "======= SCALR(15,50) ======= "
             << input.getActualBlockNo()
             << endl;
#endif
        input.skipBlocks(1);
        if (input.fail())
        {
            maxn = 0;
            return;
        }
    }
    ////////////////////////////////////////////////////////

    if (maxscl)
    {
#ifdef VERBOSE_BLOCKS
        cerr << "======= Scalar Names ======= "
             << input.getActualBlockNo()
             << endl;
#endif
        char *buf = new char[20 * maxscl];
        if (input.readFTN(buf, 20 * maxscl) < 0)
        {
            maxn = 0;
            return;
        }
        if (input.fail())
        {
            maxn = 0;
            return;
        }

        scalName = new char *[maxscl + 1];
        for (i = 1; i <= maxscl; i++)
        {
            scalName[i] = new char[21];
            strncpy(scalName[i], buf + 20 * (i - 1), 20);
            scalName[i][20] = '\0';
            int j = 19;
            while (j && isspace(scalName[i][j]))
            {
                scalName[i][j] = '\0';
                j--;
            }
        }
        for (i = 1; i <= maxscl; i++)
        {
            char *chPtr = scalName[i];
            while (*chPtr)
            {
                if (*chPtr == ' ')
                    *chPtr = '_';
                chPtr++;
            }
        }
    }
////////////////////////////////////////////////////////

#ifdef VERBOSE_BLOCKS
    cerr << "stopped before block "
         << input.getActualBlockNo()
         << endl;
#endif

////////////////////////////////////////////////////////
////
////   SAMM handling
////
////////////////////////////////////////////////////////

#ifdef DO_SAMM
    if ((jvers > 2310) && (mxsam > 0))
    {
        dumper("Converting SAMM");
        SammConv conv;
        oldOfNewCell = conv.convertSamm(cellTab, sammTab, cellType, maxe, mxtb, dumper);

#ifdef VERBOSE_SAMM
        ofstream map("zzMapSAMM");
        int last = -9999;
        for (i = 0; i < maxe; i++)
        {
            if (last != oldOfNewCell[i])
            {
                last = oldOfNewCell[i];
                map << "\n" << last << "\t\t";
            }
            map << i << " ";
        }
        map << endl;
#endif
    }
    else // no SAMM : create dummy map
    {
#endif
        oldOfNewCell = new int[maxe];
        for (i = 0; i < maxe; i++)
            oldOfNewCell[i] = i;
#ifdef DO_SAMM
    }
#endif

////////////////////////////////////////////////////////
////
////   Illegal cell form handling
////
////////////////////////////////////////////////////////

#ifdef DO_ILL
    IllConv illConv;

    if (maxe > 250000)
        dumper("Converting ILL-shaped");

    illConv.convertIll(cellTab, cellType, oldOfNewCell, maxe, mxtb, dumper);
#endif

    if (getenv("VERBOSE_STARLIB_PROSTAR"))
    {
        ofstream map("zzPROSTAR");
        int last = -9999;
        for (i = 0; i < maxe; i++)
        {
            if (last != oldOfNewCell[i])
            {
                last = oldOfNewCell[i];
                map << "\n" << last << "\t\t";
            }
            map << i << " ";
        }
        map << endl;
    }

    /////////////////////////////////////////////////////////
    // Calculate Region sizes
    if (maxr == 0)
        regionSize = NULL;
    else
    {
        regionSize = new RegionSize[maxr + 1];
        for (i = 0; i <= maxr; i++)
        {
            regionSize[i].numPoly = 0;
            regionSize[i].numTria = 0;
        }
        for (i = 0; i < maxb; i++)
        {
            if (bounTab[i].vertex[0] > 0)
            {
                int reg = bounTab[i].region;
                regionSize[reg].numPoly++;
                if (bounTab[i].vertex[2] == bounTab[i].vertex[3])
                    regionSize[reg].numTria++;
            }
        }
    }
}

void File16::createMap(int calcSolids)
{
    ////////////////////////////////////////////////////////
    ///  Now calculate some things we need to know

    int idx, i;
    numCovCells = 0;
    numCovConn = 0;

    // build up lists of cell shapes - in ProStar numbering
    cellShapeArr = new int[maxe];

    ////////////////////////////////////////////////////////
    if (cells_used)
        delete[] cells_used;
    cells_used = new int[maxe];

    /////////////////////////////////////////////////////////
    // First: find out, how many Cells are REALLY used
    //        on the side: find #connections and shapes

    for (i = 0; i < maxe; i++)
        if ((cellTab[i].ictID > 0) && (cellTab[i].ictID <= mxtb)
            && (cellTab[i].vertex[0] > 0)
            && ((cellType[cellTab[i].ictID - 1].ctype == 1)
                || ((cellType[cellTab[i].ictID - 1].ctype == 2) && calcSolids)))
        {
            // find shape of this cell
            int *vertex = &(cellTab[i].vertex[0]) - 1; // -1 fuer FORTRAN[]
            int shape;
            if (vertex[3] == vertex[4])
                if (vertex[6] == vertex[7])
                    shape = TETRAHEDRON;
                else
                    shape = PRISM;
            else if (vertex[6] == vertex[7])
                shape = PYRAMID;
            else
                shape = HEXAGON;

            cellShapeArr[i] = shape;
            cells_used[numCovCells] = i;
            numCovCells++;
            numCovConn += numVert[shape];
        }

    /////////////////////////////////////////////////////////
    // Second: Prostar -> Star mapping
    delete[] covToPro;
    delete[] covToStar;
    covToPro = new int[numCovCells];
    covToStar = new int[numCovCells];

    // Ignoring Versions < 2.1.1.5 !!!
    // Now, the Fluids
    idx = 0;
    int dataidx = -1;
    int lastOld = -99999;
    for (i = 0; i < maxe; i++)
        if ((cellTab[i].ictID > 0) && (cellTab[i].ictID <= mxtb)
            && (cellTab[i].vertex[0] > 0)
            && (cellType[cellTab[i].ictID - 1].ctype == 1)
            && (cellTab[i].vertex[0] > 0))
        {
            covToPro[idx] = i;
            if (lastOld != oldOfNewCell[i]) // increment data mapping ony
            { // when changing the orig cell
                dataidx++;
                lastOld = oldOfNewCell[i];
            }
            covToStar[idx] = dataidx;
            idx++;
        }

    // if calculated, do the solids now
    if (calcSolids)
        for (i = 0; i < maxe; i++)
            if ((cellTab[i].ictID > 0) && (cellTab[i].ictID <= mxtb)
                && (cellType[cellTab[i].ictID - 1].ctype == 2)
                && (cellTab[i].vertex[0] > 0))
            {
                covToPro[idx] = i;
                if (lastOld != oldOfNewCell[i]) // increment data mapping ony
                { // when changing the orig cell
                    dataidx++;
                    lastOld = oldOfNewCell[i];
                }
                covToStar[idx] = dataidx;
                idx++;
            }

    // this is how many cells in the original Star data set
    numOrigStarCells = dataidx + 1;

    // we do not need the rest yet
    return;
}

// get the sizes for allocation
void File16::getMeshSize(int &numCells, int &numConn, int &numV)
{
    numCells = numCovCells;
    numConn = numCovConn;
    numV = maxn; // @@@@@@@@ do sth here !!!
}

File16::~File16()
{
    delete[] cellTab;
    delete[] bounTab;
    delete[] vertexTab;
    delete[] cellType;
    delete[] cp22;
    delete[] cp23;
    delete[] covToPro;
    delete[] cellShapeArr;
    delete[] regionSize;
}

// get my own mesh - for non-moving grid
void File16::getMesh(int *elPtr, int *clPtr, int *tlPtr,
                     float *xPtr, float *yPtr, float *zPtr,
                     int *typPtr)
{
    float zoom;
    if (jvers >= 3000)
        zoom = 1.0;
    else
        zoom = 1.0f / scale8;
    int i, elmNr = 0;
    for (i = 0; i < numCovCells; i++)
    {
        int proIdx = covToPro[i];
        /// read type info
        int cTabNo = *typPtr = cellTab[proIdx].ictID;
        typPtr++;
        cTabNo--; //decrement to use c Array indexingon Fortran Fields
        *tlPtr = cellShapeArr[proIdx];
        *elPtr++ = elmNr;
        int *vertex = &cellTab[proIdx].vertex[0];
        *clPtr = vertex[0] - 1;
        clPtr++;
        *clPtr = vertex[1] - 1;
        clPtr++;
        *clPtr = vertex[2] - 1;
        clPtr++;
        switch (*tlPtr)
        {
        case HEXAGON:
            *clPtr = vertex[3] - 1;
            clPtr++;
            *clPtr = vertex[4] - 1;
            clPtr++;
            *clPtr = vertex[5] - 1;
            clPtr++;
            *clPtr = vertex[6] - 1;
            clPtr++;
            *clPtr = vertex[7] - 1;
            clPtr++;
            elmNr += 8;
            break;
        case PRISM:
            *clPtr = vertex[4] - 1;
            clPtr++;
            *clPtr = vertex[5] - 1;
            clPtr++;
            *clPtr = vertex[6] - 1;
            clPtr++;
            elmNr += 6;
            break;
        case PYRAMID:
            *clPtr = vertex[3] - 1;
            clPtr++;
            *clPtr = vertex[4] - 1;
            clPtr++;
            elmNr += 5;
            break;
        case TETRAHEDRON:
            *clPtr = vertex[4] - 1;
            clPtr++;
            elmNr += 4;
            break;
        }
        tlPtr++;
    }

    for (i = 0; i < maxn; i++)
    {
        *xPtr++ = vertexTab[i].coord[0] * zoom;
        *yPtr++ = vertexTab[i].coord[1] * zoom;
        *zPtr++ = vertexTab[i].coord[2] * zoom;
    }
}

void File16::getReducedMesh(int *el, int *cl, int *tl,
                            int *redCovToStar,
                            float *vx, float *vy, float *vz,
                            int *eLenPtr, int *cLenPtr, int *vLenPtr,
                            int *typPtr)
{
    float zoom;
    if (jvers >= 3000)
        zoom = 1.0;
    else
        zoom = 1.0f / scale8;
    int *elPtr = el, *tlPtr = tl, *clPtr = cl;
    int &eLen = *eLenPtr, &cLen = *cLenPtr, &vLen = *vLenPtr;
    eLen = cLen = 0;
    int covElem;

    for (covElem = 0; covElem < numCovCells; covElem++)
    {
        int proElem = covToPro[covElem];
        float &vxc0 = vx[cellTab[proElem].vertex[0] - 1],
              &vyc0 = vy[cellTab[proElem].vertex[0] - 1],
              &vzc0 = vz[cellTab[proElem].vertex[0] - 1];
        float &vxc1 = vx[cellTab[proElem].vertex[1] - 1],
              &vyc1 = vy[cellTab[proElem].vertex[1] - 1],
              &vzc1 = vz[cellTab[proElem].vertex[1] - 1];
        float &vxc2 = vx[cellTab[proElem].vertex[2] - 1],
              &vyc2 = vy[cellTab[proElem].vertex[2] - 1],
              &vzc2 = vz[cellTab[proElem].vertex[2] - 1];
        float &vxc3 = vx[cellTab[proElem].vertex[3] - 1],
              &vyc3 = vy[cellTab[proElem].vertex[3] - 1],
              &vzc3 = vz[cellTab[proElem].vertex[3] - 1];
        float &vxc4 = vx[cellTab[proElem].vertex[4] - 1],
              &vyc4 = vy[cellTab[proElem].vertex[4] - 1],
              &vzc4 = vz[cellTab[proElem].vertex[4] - 1];
        float &vxc5 = vx[cellTab[proElem].vertex[5] - 1],
              &vyc5 = vy[cellTab[proElem].vertex[5] - 1],
              &vzc5 = vz[cellTab[proElem].vertex[5] - 1];
        float &vxc6 = vx[cellTab[proElem].vertex[6] - 1],
              &vyc6 = vy[cellTab[proElem].vertex[6] - 1],
              &vzc6 = vz[cellTab[proElem].vertex[6] - 1];
        float &vxc7 = vx[cellTab[proElem].vertex[7] - 1],
              &vyc7 = vy[cellTab[proElem].vertex[7] - 1],
              &vzc7 = vz[cellTab[proElem].vertex[7] - 1];

        if ((cellTab[proElem].vertex[0] < 0)
            || ((vxc0 == vxc3) && (vxc1 == vxc2) && (vxc4 == vxc7) && (vxc5 == vxc6)
                && (vyc0 == vyc3) && (vyc1 == vyc2) && (vyc4 == vyc7) && (vyc5 == vyc6)
                && (vzc0 == vzc3) && (vzc1 == vzc2) && (vzc4 == vzc7) && (vzc5 == vzc6))

            || ((vxc0 == vxc1) && (vxc3 == vxc2) && (vxc4 == vxc5) && (vxc7 == vxc6)
                && (vyc0 == vyc1) && (vyc3 == vyc2) && (vyc4 == vyc5) && (vyc7 == vyc6)
                && (vzc0 == vzc1) && (vzc3 == vzc2) && (vzc4 == vzc5) && (vzc7 == vzc6))

            || ((vxc0 == vxc4) && (vxc1 == vxc5) && (vxc3 == vxc7) && (vxc2 == vxc6)
                && (vyc0 == vyc4) && (vyc1 == vyc5) && (vyc3 == vyc7) && (vyc2 == vyc6)
                && (vzc0 == vzc4) && (vzc1 == vzc5) && (vzc3 == vzc7) && (vzc2 == vzc6))

            || (vxc0 > 1e29) || (vxc1 > 1e29) || (vxc2 > 1e29) || (vxc3 > 1e29)
            || (vxc4 > 1e29) || (vxc5 > 1e29) || (vxc6 > 1e29) || (vxc7 > 1e29))
        {
            // this cell is not used, so no entry for this mapping direction
            // redCovToStar[eLen]=-1;
        }
        else
        {
            redCovToStar[eLen] = covToStar[covElem];
            eLen++;
            *tlPtr = cellShapeArr[proElem];
            *elPtr++ = cLen;
            int *vertex = &cellTab[proElem].vertex[0];

            /// read type info

            //-1 FORTRAN
            int cTabNo = cellTab[covToPro[covElem]].ictID - 1;
            // CellTypeEntry *ct = findCellType(cTabNo+1);

            *typPtr = cTabNo + 1;
            typPtr++;

            *clPtr = vertex[0] - 1;
            clPtr++;
            *clPtr = vertex[1] - 1;
            clPtr++;
            *clPtr = vertex[2] - 1;
            clPtr++;
            switch (*tlPtr)
            {
            case HEXAGON:
                *clPtr = vertex[3] - 1;
                clPtr++;
                *clPtr = vertex[4] - 1;
                clPtr++;
                *clPtr = vertex[5] - 1;
                clPtr++;
                *clPtr = vertex[6] - 1;
                clPtr++;
                *clPtr = vertex[7] - 1;
                clPtr++;
                cLen += 8;
                break;
            case PRISM:
                *clPtr = vertex[4] - 1;
                clPtr++;
                *clPtr = vertex[5] - 1;
                clPtr++;
                *clPtr = vertex[6] - 1;
                clPtr++;
                cLen += 6;
                break;
            case PYRAMID:
                *clPtr = vertex[3] - 1;
                clPtr++;
                *clPtr = vertex[4] - 1;
                clPtr++;
                cLen += 5;
                break;
            case TETRAHEDRON:
                *clPtr = vertex[4] - 1;
                clPtr++;
                cLen += 4;
                break;
            }
            tlPtr++;
        }
    }

    // Vertex coordinates from File29 -> supplied

    // Now do Vertex compression
    int i;
    int *table = new int[vLen];
    memset(table, 0, sizeof(int) * vLen);

    // first mark used vertices
    clPtr = cl;
    for (i = 0; i < cLen; i++)
    {
        table[*clPtr] = -1;
        clPtr++;
    }

    // count and build transition table      // table [alt] = neu
    int numUsed = 0;
    for (i = 0; i < vLen; i++)
        if (table[i])
        {
            table[i] = numUsed;
            numUsed++;
        }
        else
            table[i] = -1;

    // compress and scale vertices in situ
    for (i = 0; i < vLen; i++)
        if (table[i] >= 0)
        {
            vx[table[i]] = vx[i] * zoom;
            vy[table[i]] = vy[i] * zoom;
            vz[table[i]] = vz[i] * zoom;
        }

    // correct connectivity table
    clPtr = cl;
    for (i = 0; i < cLen; i++)
    {
        *clPtr = table[*clPtr];
        clPtr++;
    }
    delete[] table;

    vLen = numUsed;
}

// Whwnever a file is invalid, we put maxn=0
int File16::isValid()
{
    return (maxn > 0);
}

// get the sizes for Region patches: Does not re-use vertices
void File16::getRegionPatchSize(int region, int &numPoly, int &numConn, int &numVert)
{
    if (region > 0 && region <= maxr)
    {
        numPoly = regionSize[region].numPoly;
        numConn = 4 * regionSize[region].numPoly - regionSize[region].numTria;
        numVert = 4 * numPoly - regionSize[region].numTria;
    }
    else
        numPoly = numConn = numVert = 0;
}

// get the patches for  this region
void File16::getRegionPatch(int reqRegion, int *polyPtr, int *connPtr,
                            float *x, float *y, float *z)
{
    float zoom;
    if (jvers >= 3000)
    {
        zoom = 1.0;
    }
    else
    {
        zoom = 1.0f / scale8;
    }
    BounTabEntry *boun = bounTab;
    int i, vert = 0;
    int vertIndex;
#ifdef VERBOSE_REGI
    cerr << "---------------------------- Polygons for region " << reqRegion << endl;
#endif
    for (i = 0; i < maxb; i++)
    {
#ifdef VERBOSE_REGI
        cerr << "Boundary patch " << i << " for region " << boun->region << endl;
#endif
        if (boun->region == reqRegion)
        {
            // PolygonList entry
            *polyPtr++ = vert;
#ifdef VERBOSE_REGI
            cerr << "Polygon starts at vertex " << vert << endl;
#endif

            // Vertex 0
            vertIndex = boun->vertex[0] - 1;
            *x = zoom * vertexTab[vertIndex].coord[0];
            x++;
            *y = zoom * vertexTab[vertIndex].coord[1];
            y++;
            *z = zoom * vertexTab[vertIndex].coord[2];
            z++;
#ifdef VERBOSE_REGI
            cerr << "  Vertex " << vert << " at "
                 << x[-1] << "/" << y[-1] << "/" << z[-1] << endl;
#endif
            *connPtr = vert++;
            connPtr++;

            // Vertex 1
            vertIndex = boun->vertex[1] - 1;
            *x = zoom * vertexTab[vertIndex].coord[0];
            x++;
            *y = zoom * vertexTab[vertIndex].coord[1];
            y++;
            *z = zoom * vertexTab[vertIndex].coord[2];
            z++;
#ifdef VERBOSE_REGI
            cerr << "  Vertex " << vert << " at "
                 << x[-1] << "/" << y[-1] << "/" << z[-1] << endl;
#endif
            *connPtr = vert++;
            connPtr++;

            // Vertex 2
            vertIndex = boun->vertex[2] - 1;
            *x = zoom * vertexTab[vertIndex].coord[0];
            x++;
            *y = zoom * vertexTab[vertIndex].coord[1];
            y++;
            *z = zoom * vertexTab[vertIndex].coord[2];
            z++;
#ifdef VERBOSE_REGI
            cerr << "  Vertex " << vert << " at "
                 << x[-1] << "/" << y[-1] << "/" << z[-1] << endl;
#endif
            *connPtr = vert++;
            connPtr++;

            // Vertex 3 only if not a triangle
            if (boun->vertex[3] != boun->vertex[2])
            {
                vertIndex = boun->vertex[3] - 1;
                *x = zoom * vertexTab[vertIndex].coord[0];
                x++;
                *y = zoom * vertexTab[vertIndex].coord[1];
                y++;
                *z = zoom * vertexTab[vertIndex].coord[2];
                z++;
#ifdef VERBOSE_REGI
                cerr << "  Vertex " << vert << " at "
                     << x[-1] << "/" << y[-1] << "/" << z[-1] << endl;
#endif
                *connPtr = vert++;
                connPtr++;
            }
        }
        boun++;
    }
}

//////////////////////////////////////////////////////////////////////////////

int File16::getMaxProstarIdx()
{
    return maxe;
}

const char *File16::getScalName(int i)
{
    if (i > 0 && i <= maxscl)
        return scalName[i];
    else
        return NULL;
}

//////////////////////////////////////////////////////////////////////////////
// check whether the file we read before is the same as this one: inode/device
int File16::isFile(const char *filename)
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

//////////////////////////////////////////////////////////////////////////////
// set the dump device: must be a pointer to 'void funct(const char *)'
// all message output is directed to this 'device', Modules can set it to
// CoModule::sendInfo
void File16::setDumper(void (*newDumper)(const char *))
{
    dumper = newDumper;
}

// this is the 'default' dumper device
void File16::printStderr(const char *text)
{
    cerr << text << endl;
}

//////////////////////////////////////////////////////////////////////////////
// CP matches: ONLY 3.x ye

// Utility function: get a specified face from a certain cell
void File16::getface(int cellNo, int faceNo, int &shape, int poly[4])
{
    const CellTabEntry &ctab = cellTab[cellNo]; // apply SAMM here..
    const int *vert = ctab.vertex;

    switch (faceNo)
    {
    case 1:
        poly[0] = vert[0];
        poly[1] = vert[1];
        poly[2] = vert[2];
        poly[3] = vert[3];
        break;
    case 2:
        poly[0] = vert[7];
        poly[1] = vert[6];
        poly[2] = vert[5];
        poly[3] = vert[4];
        break;
    case 3:
        poly[0] = vert[1];
        poly[1] = vert[0];
        poly[2] = vert[4];
        poly[3] = vert[5];
        break;
    case 4:
        poly[0] = vert[3];
        poly[1] = vert[2];
        poly[2] = vert[6];
        poly[3] = vert[7];
        break;
    case 5:
        poly[0] = vert[0];
        poly[1] = vert[3];
        poly[2] = vert[7];
        poly[3] = vert[4];
        break;
    case 6:
        poly[0] = vert[1];
        poly[1] = vert[5];
        poly[2] = vert[6];
        poly[3] = vert[2];
        break;

    // any other face is incorrect/empty
    default:
        shape = 0;
        return;
    }

    shape = 4;

    // eliminate all polys wioth 0s in it
    int i, j, k;

    for (i = 0; i < 4; i++)
        if (poly[i] == 0)
        {
            shape = 0;
            return;
        }

    // eliminate duplicate vertices:

    for (k = 0; k < shape - 1; k++)
        for (i = 0; i < shape - 1; i++)
            if (poly[i] == poly[i + 1])
            {
                for (j = i + 1; j < shape - 2; j++)
                    poly[i] = poly[i + 1];
                shape--;
            }

    return;
}

////////////////////////////////////////////////////////////////////////////
/// finde new (SAMM-corrected) celltab entry for given one

int File16::findNewCell(int oldCell)
{
    int mini = 0;
    int maxi = maxe;
    int mid = (mini + maxi) / 2;
    while (oldOfNewCell[mid] != oldCell && maxi > mini)
    {
        if (oldOfNewCell[mid] < oldCell)
            mini = mid + 1;
        else
            maxi = mid - 1;
        mid = (mini + maxi) / 2;
    }
    if (oldOfNewCell[mid] != oldCell)
        return -1;
    else
        return mid;
}

///////////////////////////////////////////////////////////////////////////
/// Get allocation size of Coupling Polygons
void File16::getCPsizes(int &numVert, int &numConn, int &numPoly)
{
    numConn = numVert = numPoly = 0;
    int i, j, v;

    ////////////////////////// CP Matches
    // only if we have couples in v3000 or above: need face info
    if (cp30)
    {
        int *vertTrafo = new int[maxn];
        for (i = 0; i < maxn; i++)
            vertTrafo[i] = -1;

        for (i = 0; i < numRealCP; i++) // loop all CP sets
        {
            int shape, vert[4]; // Master Cell
            CoupledCells30 &cp = cp30[i];
            getface(findNewCell(cp.master - 1), cp.masterSide, shape, vert);
            if (shape)
            {
                numConn += shape;
                numPoly++;
                for (v = 0; v < shape; v++)
                    if (vertTrafo[vert[v] - 1] == -1)
                        vertTrafo[vert[v] - 1] = numVert++;
            }
            for (j = 0; j < ncpdmf - 1; j++) // all slave cells
            {
                if (cp.slave[j])
                {
                    getface(findNewCell(cp.slave[j] - 1), cp.slaveSide[j], shape, vert);
                    if (shape)
                    {
                        numConn += shape;
                        numPoly++;
                        for (v = 0; v < shape; v++)
                            if (vertTrafo[vert[v] - 1] == -1)
                                vertTrafo[vert[v] - 1] = numVert++;
                    }
                }
            }
        }
        delete[] vertTrafo;
    }

    ////////////////////////// ATTACH boundaries
    for (i = 0; i <= maxr; i++)
    {
        if (regionType[i] == 13) // ATTACH
        {
            int numRegiPoly, numRegiConn, numRegiVert;
            getRegionPatchSize(i, numRegiPoly, numRegiConn, numRegiVert);
            numPoly += numRegiPoly;
            numConn += numRegiConn;
            numVert += numRegiVert;
        }
    }
}

/////////////////////////////////////////////////////////////////////
/// Build CP match polygons
void File16::getCPPoly(float *xVert, float *yVert, float *zVert,
                       float *xPoly, float *yPoly, float *zPoly,
                       int *polyTab, int *connTab)
{
    int numConn = 0;
    int numVert = 0;
    int i, j, v;

    if (cp30) // only if we have couples and V3000 or above
    {
        int *vertTrafo = new int[maxn];
        for (i = 0; i < maxn; i++)
            vertTrafo[i] = -1;

        for (i = 0; i < numRealCP; i++) // Master Cells
        {
            int shape, vert[4];
            CoupledCells30 &cp = cp30[i];
            getface(findNewCell(cp.master - 1), cp.masterSide, shape, vert);
            if (shape)
            {
                *polyTab++ = numConn;
                numConn += shape;
                for (v = 0; v < shape; v++)
                {
                    if (vertTrafo[vert[v] - 1] == -1) // new vertex
                    {
                        // add to Polygon
                        vertTrafo[vert[v] - 1] = numVert++;
                        *xPoly++ = xVert[vert[v] - 1];
                        *yPoly++ = yVert[vert[v] - 1];
                        *zPoly++ = zVert[vert[v] - 1];
                    }
                    *connTab++ = vertTrafo[vert[v] - 1]; // add to ConnList
                }
            }
            for (j = 0; j < ncpdmf - 1; j++) // Slave Cells
            {
                if (cp.slave[j])
                {
                    getface(findNewCell(cp.slave[j] - 1), cp.slaveSide[j], shape, vert);
                    if (shape)
                    {
                        *polyTab++ = numConn;
                        numConn += shape;
                        for (v = 0; v < shape; v++)
                        {
                            if (vertTrafo[vert[v] - 1] == -1) // new vertex
                            {
                                // add to Polygon
                                vertTrafo[vert[v] - 1] = numVert++;
                                *xPoly++ = xVert[vert[v] - 1];
                                *yPoly++ = yVert[vert[v] - 1];
                                *zPoly++ = zVert[vert[v] - 1];
                            }
                            // add to ConnList
                            *connTab++ = vertTrafo[vert[v] - 1];
                        }
                    }
                }
            }
        }
        delete[] vertTrafo;
    }

    //////////////////////////////////////////////////// ATTACH boundaries
    for (i = 0; i <= maxr; i++)
    {
        if (regionType[i] == 13) // ATTACH boundaries
        {
            int numRegiPoly, numRegiConn, numRegiVert;
            getRegionPatchSize(i, numRegiPoly, numRegiConn, numRegiVert);

            // get patches: Vertices=0..x polyTab=0..x
            // append to Polygons collected so far
            getMovedRegionPatch(i, xVert, yVert, zVert,
                                polyTab, connTab, xPoly, yPoly, zPoly);
            for (j = 0; j < numRegiConn; j++) // transform vertex to start at numVert
                connTab[j] += numVert;

            for (j = 0; j < numRegiPoly; j++) // transform poly to start at numConn
                polyTab[j] += numConn;

            polyTab += numRegiPoly; // set pointers behind firlds for next loop
            connTab += numRegiConn;
            xPoly += numRegiVert;
            yPoly += numRegiVert;
            zPoly += numRegiVert;

            numConn += numRegiConn;
            numVert += numRegiVert;
        }
    }
}

/////////////////////////////////////////////////////////////////////////////
// Moving-grid version of getRegionPatch: get vertex coordinates as parameters
// get the patches for  this region
void File16::getMovedRegionPatch(int reqRegion, float *xv, float *yv, float *zv,
                                 int *polyPtr, int *connPtr,
                                 float *x, float *y, float *z)
{
    float zoom;
    if (jvers >= 3000)
    {
        zoom = 1.0f;
    }
    else
    {
        zoom = 1.0f / scale8;
    }
    BounTabEntry *boun = bounTab;
    int i, vert = 0;
    int vertIndex;
#ifdef VERBOSE_REGI
    cerr << "---------------------------- Polygons for region " << reqRegion << endl;
#endif
    for (i = 0; i < maxb; i++)
    {
#ifdef VERBOSE_REGI
        cerr << "Boundary patch " << i << " for region " << boun->region << endl;
#endif
        if (boun->region == reqRegion)
        {
            // PolygonList entry
            *polyPtr++ = vert;
#ifdef VERBOSE_REGI
            cerr << "Polygon starts at vertex " << vert << endl;
#endif

            // Vertex 0
            vertIndex = boun->vertex[0] - 1;
            *x = zoom * xv[vertIndex];
            x++;
            *y = zoom * yv[vertIndex];
            y++;
            *z = zoom * zv[vertIndex];
            z++;
#ifdef VERBOSE_REGI
            cerr << "  Vertex " << vert << " at "
                 << x[-1] << "/" << y[-1] << "/" << z[-1] << endl;
#endif
            *connPtr = vert++;
            connPtr++;

            // Vertex 1
            vertIndex = boun->vertex[1] - 1;
            *x = zoom * xv[vertIndex];
            x++;
            *y = zoom * yv[vertIndex];
            y++;
            *z = zoom * zv[vertIndex];
            z++;
#ifdef VERBOSE_REGI
            cerr << "  Vertex " << vert << " at "
                 << x[-1] << "/" << y[-1] << "/" << z[-1] << endl;
#endif
            *connPtr = vert++;
            connPtr++;

            // Vertex 2
            vertIndex = boun->vertex[2] - 1;
            *x = zoom * xv[vertIndex];
            x++;
            *y = zoom * yv[vertIndex];
            y++;
            *z = zoom * zv[vertIndex];
            z++;
#ifdef VERBOSE_REGI
            cerr << "  Vertex " << vert << " at "
                 << x[-1] << "/" << y[-1] << "/" << z[-1] << endl;
#endif
            *connPtr = vert++;
            connPtr++;

            // Vertex 3 only if not a triangle
            if (boun->vertex[3] != boun->vertex[2])
            {
                vertIndex = boun->vertex[3] - 1;
                *x = zoom * xv[vertIndex];
                x++;
                *y = zoom * yv[vertIndex];
                y++;
                *z = zoom * zv[vertIndex];
                z++;
#ifdef VERBOSE_REGI
                cerr << "  Vertex " << vert << " at "
                     << x[-1] << "/" << y[-1] << "/" << z[-1] << endl;
#endif
                *connPtr = vert++;
                connPtr++;
            }
        }
        boun++;
    }
}
