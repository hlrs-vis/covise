/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */


#include <covise/covise.h>
#include "File29.h"
#include "File16.h"
#include "File09.h"
#include <sys/stat.h>

#ifdef _WIN32
#include <io.h>
#include <fcntl.h>
#endif

#ifdef _STANDARD_C_PLUS_PLUS
using std::flush;
#endif

namespace covise
{
void printStdout(const char *text)
{
    cout << text << endl;
}

void printNothing(const char *)
{
}

// ===========================================================================
int info29(const char *filename)
{
#ifdef _WIN32
    int fd29 = open(filename, O_RDONLY, O_BINARY);
#else
    int fd29 = open(filename, O_RDONLY);
#endif

    if (fd29 < 0)
    {
        perror(filename);
        exit(0);
    }

    File29 file29(fd29, printNothing);

    if (!file29.isValid())
    {
        close(fd29);
        return 0;
    }

    cout << "Type:                           Transient StarCD v" << -file29.lvers << " Data file\n"
         << "Title                           '" << file29.title << "'\n"
         << "Number of Cells:         (MAXE) " << file29.ncell << "\n"
         << "Number of Vertices:      (MAXN) " << file29.nnode << "\n"
         << "Number of Timesteps:            " << file29.get_num_steps()
         << endl;
    cout << flush;
    cerr << flush;

    int i;
    printf("%10s %8s %7s\n", "Block", "time", "Drops");
    for (i = 1; i <= file29.get_num_steps(); i++)
    {
        file29.skip_to_step(i);
        printf("%10i  %8e  %i\n",
               file29.headerBlock[i - 1] - 1, file29.getRealTime(i), file29.getNumDrops(i));
    }

#ifdef __sgi
    struct stat64 filestat;
    fstat64(fd29, &filestat);
#else
    struct stat filestat;
    fstat(fd29, &filestat);
#endif
    long numBlk = filestat.st_size / 8192;

    if (numBlk == file29.headerBlock[i - 1])
        printf("%10i = termination block\n", file29.headerBlock[i - 1] - 1);
    else
        printf(" no termination block\n");

    return 1;
}

// ===========================================================================
int info16(const char *filename)
{
#ifdef _WIN32
    int fd16 = open(filename, O_RDONLY, O_BINARY);
#else
    int fd16 = open(filename, O_RDONLY);
#endif
    if (fd16 < 0)
    {
        perror(filename);
        return 0;
    }

    File16 file16(fd16, printStdout);

    if (!file16.isValid())
    {
        close(fd16);
        return 0;
    }

    cout << "Type:                           StarCD v" << file16.jvers << " Mesh file\n"
         << "title                           '" << file16.title.main << "'\n"
         << "Subtitle1                       '" << file16.title.sub1 << "'\n"
         << "Subtitle1                       '" << file16.title.sub2 << "'\n"
         << "Number of Cells:         (MAXE) " << file16.maxe << "\n"
         << "Number of Vertices:      (MAXN) " << file16.maxn << "\n"
         << "Number of Regions:       (MAXR) " << file16.maxr << "\n"
         << "Number of Boundaries:    (MAXB) " << file16.maxb << "\n"
         << "Number of Couples:      (MAXCP) " << file16.maxcp << "\n"
         << "Geometry Scale factor: (SCALE8) " << file16.scale8 << "\n";
    if (file16.jvers > 2310)
        cout
            << "Number of SAMM cells:   (MXSAM) " << file16.pbtol << "\n";
    cout << endl;

    cout << flush;
    cerr << flush;

    /////////////////////////////////////////////////////////////////////////////////////

    FILE *cells = fopen("cells", "w");
    FILE *vertices = fopen("vertices", "w");
    int c, v;

    // vertex usage list
    char *useVert = new char[file16.maxn + 1];
    for (v = 0; v <= file16.maxn; v++)
        useVert[v] = 0;

    int s = 0; // SAMM table entry index

    // loop over cell table
    for (c = 0; c < file16.maxe; c++)
    {
        // if cell used
        if ((file16.cellTab[c].ictID > 0)
            && (file16.cellTab[c].ictID <= file16.mxtb)
            && (file16.cellTab[c].vertex[0] >= 0)
                   // volume cells
            && (file16.cellType[file16.cellTab[c].ictID - 1].ctype < 3))
        {
            int vert[20]; // max 8 corners + 12 SAMM points
            int numSAMM = 0; // number of cut-away corners
            int numVert = 0;
            for (v = 0; v < 8; v++)
                if (file16.cellTab[c].vertex[v] == 0)
                    numSAMM++;
                else
                    vert[numVert++] = file16.cellTab[c].vertex[v];

            ////////////////////////////////////////////////////////////////
            // 'normal' irregular cases
            if (numSAMM == 0 // regular Prism
                && numVert == 8
                && file16.cellTab[c].vertex[2] == file16.cellTab[c].vertex[3]
                && file16.cellTab[c].vertex[6] == file16.cellTab[c].vertex[7])
            {
                printNothing("");
                continue;
            }
            if (numSAMM == 0 // regular Pyra
                && numVert == 8
                && file16.cellTab[c].vertex[4] == file16.cellTab[c].vertex[5]
                && file16.cellTab[c].vertex[5] == file16.cellTab[c].vertex[6]
                && file16.cellTab[c].vertex[6] == file16.cellTab[c].vertex[7])
            {
                printNothing("");
                continue;
            }

            if (numSAMM == 0 // regular Tetra
                && numVert == 8
                && file16.cellTab[c].vertex[2] == file16.cellTab[c].vertex[3]
                && file16.cellTab[c].vertex[4] == file16.cellTab[c].vertex[5]
                && file16.cellTab[c].vertex[5] == file16.cellTab[c].vertex[6]
                && file16.cellTab[c].vertex[6] == file16.cellTab[c].vertex[7])
            {
                printNothing("");
                continue;
            }

            if ((numSAMM == 4) // regular Quad element
                && numVert == 4
                && file16.cellTab[c].vertex[4] == 0
                && file16.cellTab[c].vertex[5] == 0
                && file16.cellTab[c].vertex[6] == 0
                && file16.cellTab[c].vertex[7] == 0)
            {
                printNothing("");
                continue;
            }

            ////////////////////////////////////////////////////////////////
            // add SAMM vertices if SAMM case

            if (numSAMM)
            {
                for (v = 0; v < 12; v++)
                    if (file16.sammTab[s].vertex[v] == 0)
                        vert[numVert++] = file16.cellTab[s].vertex[v];
            }

            ////////////////////////////////////////////////////////////////
            // Irregular? - any two vertices same?
            int irregular = 0;
            int v0, v1;
            for (v0 = 0; v0 < numVert - 1; v0++)
                for (v1 = v0 + 1; v1 < numVert; v1++)
                    if (vert[v0] == vert[v1] && vert[v0])
                    {
                        irregular = 1;
                        v0 = numVert;
                        v1 = numVert;
                        break;
                    }

            ////////////////////////////////////////////////////////////////
            // Print cell for irregular and SAMM cells
            if (irregular)
                fprintf(cells, "Irregular ");

            if (numSAMM)
                fprintf(cells, "SAMM-%d ", numSAMM);

            if (irregular || numSAMM)
            {
                fprintf(cells,
                        "\n %7d : %7d %7d %7d %7d %7d %7d %7d %7d\n", c,
                        file16.cellTab[c].vertex[0] - 1, file16.cellTab[c].vertex[1] - 1,
                        file16.cellTab[c].vertex[2] - 1, file16.cellTab[c].vertex[3] - 1,
                        file16.cellTab[c].vertex[4] - 1, file16.cellTab[c].vertex[5] - 1,
                        file16.cellTab[c].vertex[6] - 1, file16.cellTab[c].vertex[7] - 1);
                for (v = 0; v < numVert; v++)
                    useVert[vert[v]] = 1;
            }

            if (numSAMM)
                fprintf(cells,
                        " %7s : %7d %7d %7d %7d %7d %7d %7d %7d \n", "",
                        file16.sammTab[s].vertex[0] - 1, file16.sammTab[s].vertex[1] - 1,
                        file16.sammTab[s].vertex[2] - 1, file16.sammTab[s].vertex[3] - 1,
                        file16.sammTab[s].vertex[4] - 1, file16.sammTab[s].vertex[5] - 1,
                        file16.sammTab[s].vertex[6] - 1, file16.sammTab[s].vertex[7] - 1);
            if (numSAMM == 8)
                fprintf(cells,
                        " %7s : %7d %7d %7d %7d \n", "",
                        file16.sammTab[s].vertex[8] - 1, file16.sammTab[s].vertex[9] - 1,
                        file16.sammTab[s].vertex[10] - 1, file16.sammTab[s].vertex[11] - 1);

            // next SAMM table entry
            if (numSAMM)
                s++;

        } // if cell used

    } // loop over cell table

    /////////////////////////////////////////////////////////////////////////////////////

    fclose(cells);

    for (v = 1; v < file16.maxn; v++)
        if (useVert[v])
            fprintf(vertices,
                    " %7i : %12.6f %12.6f %12.6f\n", v - 1,
                    file16.vertexTab[v - 1].coord[0],
                    file16.vertexTab[v - 1].coord[1],
                    file16.vertexTab[v - 1].coord[2]);

    fclose(vertices);
    return 1;
}

// ===========================================================================
int info09(const char *filename)
{
#ifdef _WIN32
    int fd29 = open(filename, O_RDONLY, O_BINARY);
#else
    int fd29 = open(filename, O_RDONLY);
#endif
    if (fd29 < 0)
    {
        perror(filename);
        return 0;
    }

    File09 file09(fd29);

    if (!file09.isValid())
    {
        close(fd29);
        return 0;
    }

    cout << "StarCD stationary result file\n"
         << "\n"
         << "Version number:      (LVERS) " << file09.lvers << "\n"
         << "Iteration number:     (ITER) " << file09.iter << "\n"
         << "Iteration time:       (TIME) " << file09.time << "\n"
         << "Number of Cells:     (NCELL) " << file09.ncell << "\n"
         << "Number of Vertices:  (NNODE) " << file09.nnode << "\n"
         << "Number of Boundaries:  (NBC) " << file09.nbc
         << endl;

    if (file09.lvers >= 2264)
    {
        cout << "Saved species: ";
        if (file09.lstar[3])
            cout << " U,V,W";
        if (file09.lstar[4])
            cout << " P";
        if (file09.lstar[5])
            cout << " TE";
        if (file09.lstar[6])
            cout << " ED";
        if (file09.lstar[8])
            cout << " T-Vis";
        if (file09.lstar[7])
            cout << " T";
        if (file09.lstar[9])
            cout << " Den";
        if (file09.lstar[10])
            cout << " Lam-Vis";
        if (file09.lstar[11])
            cout << " CP";
        if (file09.lstar[12])
            cout << " Cond";
        if (file09.numcon > 0)
            cout << "+ " << file09.numcon << " Scalar values" << endl;
        cout << "\n" << endl;
    }
    return 1;
}
}

// ===========================================================================
using namespace covise;
int main(int argc, char *argv[])
{

    if (argc < 2)
    {
        cerr << "call: " << argv[0] << " <filename>" << endl;
        exit(0);
    }

    int i;
    for (i = 1; i < argc; i++)
    {
        cout << "=======================================================================\n"
             << "Analysis of '" << argv[i] << "'\n"
             << "-----------------------------------------------------------------------"
             << endl;
        if (info29(argv[i]) == 0 && info16(argv[i]) == 0 && info09(argv[i]) == 0)
        {
            cerr << "File " << argv[1] << " is neither of mdl, pst or pstt file" << endl;
        }
    }
    return 0;
}
