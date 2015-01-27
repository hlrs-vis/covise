/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */


#include <covise/covise.h>
#include <sys/stat.h>
#include <util/ChoiceList.h>
#include "File29.h"
#include "istreamBLK.h"
#include <sys/stat.h>

#undef VERBOSE
#undef VERBOSE_ARR
#undef VERBOSE_ARR_ANALYSE

///////////////////////////////////////////////////////////////////
//
//    Constructor
//
///////////////////////////////////////////////////////////////////

using namespace covise;

File29::File29(int fd, void (*dumpFunct)(const char *))
    : input(fd, 8192)
{
    d_actDrop = NULL;
    d_lastDropStep = -1;
    d_numActDrops = 0;

    // pre-set Droplet elimination factor, getEntry doesn't change if not found
    d_elimSizeFactor = 20;
    //CoviseConfig::getEntry("StarLib.ElimSizeFactor",&d_elimSizeFactor);
    if (fd < 0)
    {
        ncell = 0;
        d_inode = (ino_t)-1;
        d_device = (dev_t)-1;
        return;
    }
    struct stat status;
    if (fstat(fd, &status) < 0)
    {
        ncell = 0;
        d_inode = (ino_t)-1;
        d_device = (dev_t)-1;
        return;
    }
    d_inode = status.st_ino;
    d_device = status.st_dev;

    if (input.fail())
    {
        ncell = 0;
        return;
    }

    if (dumpFunct)
        dumper = dumpFunct;
    else
        dumper = File29::printStderr;

    readHeader(1);

    // check: we might be byte-swapping
    if ((lvers < -40000) || (lvers > -2209) || (iter == -1))
    {
        input.rewind();
        input.setByteSwap(1);
        readHeader(1);
        if ((lvers < -40000) || (lvers > -2209) || (iter == -1))
        {
            ncell = 0;
            return;
        }
    }

    lvers *= -1;

    /// NEW starting v3100: scan for blocks
    findHeaders();

    readHeader(1);
    analyseHeader();
}

// this is the 'default' dumper device
void File29::printStderr(const char *text)
{
    cerr << text << endl;
}

ChoiceList *File29::get_choice(const char **scalarName, int maxList) const
{
    ChoiceList *choice = new ChoiceList("---", 0);
    if ((lct[7]) || (lct[8]) || (lct[9]))
    {
        choice->add("Velocity", 1);
        choice->add("V-Magnitude", 2);
    }

    if (lct[7])
        choice->add("U", 3);
    if (lct[8])
        choice->add("V", 4);
    if (lct[9])
        choice->add("W", 5);
    if (lct[10])
        choice->add("P", 6);
    if (lct[11])
        choice->add("TE", 7);
    if (lct[12])
        choice->add("ED", 8);
    if (lct[13])
        choice->add("TVis", 9);
    if (lct[14])
        choice->add("T", 10);
    if (lct[15])
        choice->add("Dens", 11);

    if (lct[16])
        choice->add("LamVis", 12);
    if (lct[17])
        choice->add("CP", 13);
    if (lct[18])
        choice->add("Cond", 14);

    if (ndrec > 0)
    {
        choice->add("DropCoord", 15);
        choice->add("DropVel", 16);
        choice->add("DropDens", 17);
        choice->add("DropDiam", 18);
        choice->add("DropTemp", 19);
        choice->add("DropNo", 20);
        choice->add("DropMass", 21);
    }

    int i;
    for (i = 1; i <= 50; i++)
    {
        if (lct[i + 20])
        {
            if ((i <= maxList) && (*(scalarName[i])))
            {
                choice->add(scalarName[i], SCALAR - 1 + i);
            }
            else
            {
                char buffer[16];
                sprintf(buffer, "Scalar_%d", i);
                choice->add(buffer, 14 + i);
            }
        }
    }

    return choice;
}

///////////////////////////////////////////////////////////////////
//
//    readHeader
//
///////////////////////////////////////////////////////////////////

long File29::readHeader(long blockNo)
{
    if (input.seekBlock(blockNo) < 0)
        return -1;
    actualHeaderBlock = blockNo;
    long res = input.read(&iter, 2048);
    if ((res < 0) || (iter == -1))
        return -1;
    title[51] = '\0';
    return res;
}

///////////////////////////////////////////////////////////////////
//
//    analyseHeader
//
///////////////////////////////////////////////////////////////////

void File29::analyseHeader()
{

    // gather flag fields

    //cerr << "ndrop9=" << ndrop9 << endl;

    int i;
    for (i = 1; i <= 200; i++)
        lct[i] = lsrf[i] = ndata[i][1] = ndata[i][2] = ndata[i][3] = 0;

    // Cell Data

    for (i = 1; i < 7; i++) // Flux (6 comp)
        lct[i] = lct1_6;
    for (i = 0; i < 9; i++) // U,V,W,P,TE,ED,VS,T,Dens
        lct[i + 7] = lct_field1[i];
    for (i = 0; i < 3; i++)
        lct[i + 21] = lct_field1[i + 9]; // Scalars 1-3
    for (i = 0; i < 3; i++)
        lct[i + 16] = lct_field2[i]; //  LamVis,CP,COND
    lct[19] = lct_field1[7]; //  Enthalpy
    lct[20] = lct20; //  Void Fraction, flux ??
    for (i = 4; i <= 50; i++) // Scalars 4-50
        lct[i + 20] = lct_field3[i - 4];

    // Vertex Data --> never in 29 file !!

    // Wall Data

    lsrf[1] = lsrf_field1[0]; // FX,FY,FZ
    lsrf[2] = lsrf_field1[1];
    lsrf[3] = lsrf_field1[2];

    lsrf[6] = lsrf[7] = lsrf_field1[3]; // HTR,TEMP,MTRAN1-3
    lsrf[8] = lsrf[113] = lsrf_field1[4];
    lsrf[9] = lsrf[114] = lsrf_field1[5];
    lsrf[10] = lsrf[115] = lsrf_field1[6];

    lsrf[4] = lsrf_field1[7]; // YPLUS,YNORM
    lsrf[5] = lsrf_field1[8];

    lsrf[58] = lsrf_field1[9]; // HFLUX,MFLUX1-3
    lsrf[59] = lsrf_field1[10];
    lsrf[60] = lsrf_field1[11];
    lsrf[61] = lsrf_field1[12];

    lsrf[109] = lsrf[110] = lsrf_field1[15]; // HRAD
    lsrf[111] = lsrf[112] = lsrf_field1[16]; // HSOL

    for (i = 4; i <= 50; i++)
    {
        lsrf[i + 7] = lsrf_field2[i - 4]; // SCALAR WALL DATA - MTRANS,MFLUX,MCOEF
        lsrf[i + 58] = lsrf_field3[i - 4];
        lsrf[i + 112] = lsrf[i + 7];
    }

#ifdef VERBOSE_ARR
    cerr << "i\tLCT\tlsrf" << endl;
    //for (i=1;i<200;i++)
    for (i = 1; i < 40; i++)
        cerr << i << "\t" << lct[i]
             << "\t" << lsrf[i] << endl;
#endif

    // maybe correct fields for non-moving grid
    if (lmvgrd == 0)
        nwprsm = nprsm = nbsi = ias = 0;

    long actBlk = actualHeaderBlock;

    ndata[1][3] = actBlk; // Header

    // !!!!! Warning: postconv saves block#-1, I got block# !!!!!!!
    actBlk += input.numRec(nbc) + 4; // ??? + materials, molec.W., Add.Const.

    if (lmvgrd)
    {
        ndata[1][2] = actBlk;
        actBlk += input.numRec(nnode);
        ndata[2][2] = actBlk;
        actBlk += input.numRec(nnode);
        ndata[3][2] = actBlk;
        actBlk += input.numRec(nnode);
    }

    for (i = 1; i < 7; i++)
        if (lct[i])
        {
            ndata[i][1] = actBlk;
            actBlk += input.numRec(ncell);
        }

    for (i = 7; i < 71; i++) /// Cell Data
        if (lct[i])
        {
            ndata[i][1] = actBlk;
            if (i == 14)
            {
                actBlk += input.numRec(nbc + ntcell + 1);
                if ((lvers >= 3040) && (iunben > 0)) /// >>>>>>>>>>>>>>>>>
                    actBlk += input.numRec(nbc + ntcell + 1);
            }
            else if ((i == 17) || (i == 18) || (i == 19))
                actBlk += input.numRec(nbc + ntcell + 1);
            else if (i == 20)
            {
                if (0) // (twphl[1]>0)
                    actBlk += input.numRec(ncell);
                if (ndrop9 > 0)
                    actBlk += input.numRec(ncell);
            }
            else
                actBlk += input.numRec(nbc + ncell + 1);
        }

    // skip trying to read non-existing Vertex data

    if (nsitem > 0) /// Surface Data
        for (i = 1; i <= 162; i++)
            if (lsrf[i])
            {
                ndata[i][3] = actBlk;
                actBlk += input.numRec(nbc);
            }

    if (ndrop9 > 0) // Droplet location !!!!
    {
        ndrec = actBlk;
        actBlk += (ndrop9 - 1) / 146 + 1;
    }

    if (ias)
    {
        irasi = actBlk;
        actBlk += input.numRec(2 * nbsi);
    }

    if ((nwprsm == 1) && (nprsm > 0))
    {
        irpsm = actBlk;
        actBlk += (9 * nprsm - 1) / 2043 + 1;
    }

    if (lmvgrd)
    {
        ircnd = actBlk;
        actBlk++;
    }

    nextHeaderBlock = actBlk;

#ifdef VERBOSE_ARR_ANALYSE
    cerr << " NDATA [7..24][1]: "
         << ndata[7][1] << " " << ndata[8][1] << " " << ndata[9][1] << " "
         << ndata[10][1] << " " << ndata[11][1] << " " << ndata[12][1] << " "
         << ndata[13][1] << " " << ndata[14][1] << " " << ndata[15][1] << " "
         << ndata[16][1] << " " << ndata[17][1] << " " << ndata[18][1] << " "
         << ndata[19][1] << " " << ndata[20][1] << " " << ndata[21][1] << " "
         << ndata[22][1] << " " << ndata[23][1] << " " << ndata[24][1] << " "
         << endl;
    cerr << " NDATA [7..24][1]: "
         << ndata[7][1] << " " << ndata[8][1] << " " << ndata[9][1] << " "
         << ndata[10][1] << " " << ndata[11][1] << " " << ndata[12][1] << " "
         << ndata[13][1] << " " << ndata[14][1] << " " << ndata[15][1] << " "
         << ndata[16][1] << " " << ndata[17][1] << " " << ndata[18][1] << " "
         << ndata[19][1] << " " << ndata[20][1] << " " << ndata[21][1] << " "
         << ndata[22][1] << " " << ndata[23][1] << " " << ndata[24][1] << " "
         << endl;
#endif
}

///////////////////////////////////////////////////////////////////
//
//    skip_to_next_field
//
///////////////////////////////////////////////////////////////////

long File29::skip_to_next_field()
{
    long res = readHeader(nextHeaderBlock);
    if (res > 0)
        analyseHeader();
    return res;
}

long File29::skip_to_step(int stepNo)
{
    if ((stepNo > numSteps) || (stepNo < 1))
        return -1;
    long res = readHeader(headerBlock[stepNo - 1]);
    if (res > 0)
        analyseHeader();
    return res;
}

///////////////////////////////////////////////////////////////////
//
//    getRealTime
//
///////////////////////////////////////////////////////////////////

float File29::getRealTime(int step)
{
    step--; // Star counts from 1
    if (actualHeaderBlock != headerBlock[step])
    {
        readHeader(headerBlock[step]);
        analyseHeader();
    }
    return time;
}

///////////////////////////////////////////////////////////////////
//
//    getVertexCoordinates
//
///////////////////////////////////////////////////////////////////

int File29::getVertexCoordinates(int step,
                                 float *x, float *y, float *z,
                                 int *len)
{
    step--; // Star counts from 1
    if (actualHeaderBlock != headerBlock[step])
    {
        readHeader(headerBlock[step]);
        analyseHeader();
    }

    if ((step >= numSteps)
        || (ndata[1][2] == 0)
        || (ndata[2][2] == 0)
        || (ndata[3][2] == 0))
        return -1;
    input.seekBlock(ndata[1][2]); //+step*blk_per_step);
    input.read(x, nnode);
    input.seekBlock(ndata[2][2]); //+step*blk_per_step);
    input.read(y, nnode);
    input.seekBlock(ndata[3][2]); //+step*blk_per_step);
    input.read(z, nnode);
    *len = nnode;
    return ((input.fail()) ? -1 : 1);
}

///////////////////////////////////////////////////////////////////
//
// Operators for droplet handling
namespace covise
{

inline int dropValid(File29::DropRec &drop, float maxDropSize, int zeroMass)
{
    if (zeroMass)
        return (drop.iorg > 0)
               && (drop.idt > 0)
               && (drop.idc > 0)
               && (drop.diam <= maxDropSize);
    else
        return (drop.iorg > 0)
               && (drop.idt > 0)
               && (drop.idc > 0)
               && (drop.diam > 0)
               && (drop.mass > 0)
               && (drop.dens > 0)
               && (drop.diam <= maxDropSize);
}

///////////////////////////////////////////////////////////////////
//
//    readField
//
///////////////////////////////////////////////////////////////////

inline float sqr(float x) { return x * x; }

int File29::readField(int step, int field,
                      int *redCovToStar, int elements,
                      float *f1, float *f2, float *f3)
{
    // we start over, so this is ok
    input.resetErrorFlag();

    step--; // Star counts from 1
    if (actualHeaderBlock != headerBlock[step])
    {
        readHeader(headerBlock[step]);
        analyseHeader();
    }

    int i;
    float *starField = new float[ntcell];

    if (step >= numSteps)
        return -1;

    // Velocity vector

    if (field == VELOCITY)
    {

        if (ndata[7][1] || ndata[8][1] || ndata[9][1])
        {

            if (ndata[7][1])
            {
                input.seekBlock(ndata[7][1]); // +step*blk_per_step);
                input.skipFloat(nbc + 1);
                input.read(starField, ntcell);
                for (i = 0; i < elements; i++)
                    if (i >= 0)
                        f1[i] = starField[redCovToStar[i]];
            }
            else
            {
                memset(f1, 0, sizeof(float) * elements);
            }

            if (ndata[8][1])
            {
                input.seekBlock(ndata[8][1]); // +step*blk_per_step);
                input.skipFloat(nbc + 1);
                input.read(starField, ntcell);
                for (i = 0; i < elements; i++)
                    if (i >= 0)
                        f2[i] = starField[redCovToStar[i]];
            }
            else
            {
                memset(f2, 0, sizeof(float) * elements);
            }

            if (ndata[9][1])
            {
                input.seekBlock(ndata[9][1]); // +step*blk_per_step);
                input.skipFloat(nbc + 1);
                input.read(starField, ntcell);
                for (i = 0; i < elements; i++)
                    if (i >= 0)
                        f3[i] = starField[redCovToStar[i]];
            }
            else
            {
                memset(f3, 0, sizeof(float) * elements);
            }
        }
        else
        {
            delete[] starField;
            return -1;
        }
    }

    // Velocity Magintude

    else if (field == VMAG)
    {

        if (ndata[7][1] || ndata[8][1] || ndata[9][1])
        {

            if (ndata[7][1])
            {
                input.seekBlock(ndata[7][1]); // +step*blk_per_step);
                input.skipFloat(nbc + 1);
                input.read(starField, ntcell);
                for (i = 0; i < elements; i++)
                    if (i >= 0)
                        f1[i] = sqr(starField[redCovToStar[i]]);
            }
            else
            {
                memset(f1, 0, sizeof(float) * elements);
            }

            if (ndata[8][1])
            {
                input.seekBlock(ndata[8][1]); // +step*blk_per_step);
                input.skipFloat(nbc + 1);
                input.read(starField, ntcell);
                for (i = 0; i < elements; i++)
                    if (i >= 0)
                        f1[i] += sqr(starField[redCovToStar[i]]);
            }

            if (ndata[9][1])
            {
                input.seekBlock(ndata[9][1]); // +step*blk_per_step);
                input.skipFloat(nbc + 1);
                input.read(starField, ntcell);
                for (i = 0; i < elements; i++)
                    if (i >= 0)
                        f1[i] += sqr(starField[redCovToStar[i]]);
            }
            for (i = 0; i < elements; i++)
                f1[i] = sqrt(f1[i]);
        }
        else
        {
            delete[] starField;
            return -1;
        }
    }
    else if (field == DROP_COORD || field == DROP_VEL || field == DROP_DENS || field == DROP_DIAM || field == DROP_TEMP || field == DROP_NO || field == DROP_MASS)
    {

        if (ndrec)
        {
            // Droplet stored in d_actDrop are not from current timestep
            if (d_lastDropStep != step)
            {
                d_lastDropStep = step;

                input.seekBlock(ndrec);

                delete[] d_actDrop;
                d_actDrop = new DropRec[ndrop9 + 146];
                for (i = 0; i < ndrop9 + 146; i++)
                    d_actDrop[i].iorg = 0;
                int dropsLeft = ndrop9;
                int dropsread = 0;
                while (dropsLeft)
                {
                    int readDrops = (dropsLeft > 146) ? 146 : dropsLeft;
                    input.read(&d_actDrop[dropsread], 2048);
                    dropsLeft -= readDrops;
                    dropsread += readDrops;
                }
                d_numActDrops = dropsread;

                // find avg size of all valid droplets : double for sum of small numbers
                // check, whether droplets are mass-less: in that case no check for
                // dens>0, mass>0, size>0, but still for size<avg*fact
                double avgSizeDbl = 0.0;
                int numValid = 0;
                int zeroMass = 1;

                for (i = 0; i < ndrop9; i++)
                {
                    if (dropValid(d_actDrop[i], FLT_MAX, zeroMass))
                    {
                        avgSizeDbl += d_actDrop[i].diam;
                        numValid++;
                        if (d_actDrop[i].mass > 0.0) // if any droplets have a mass
                            zeroMass = 0;
                    }
                }

                // make float for faster comparison
                float maxSize = (float)avgSizeDbl * d_elimSizeFactor;
                if (numValid > 0)
                    maxSize /= numValid;
                if (maxSize == 0.0)
                    maxSize = FLT_MAX;

                // remove non-used Droplets

                d_numActDrops = 0;
                for (i = 0; i < ndrop9; i++)
                {
                    if (dropValid(d_actDrop[i], maxSize, zeroMass)
                        && d_actDrop[i + 1].iorg == d_actDrop[i].iorg + 1)
                    {
                        d_actDrop[d_numActDrops] = d_actDrop[i];
                        d_numActDrops++;
                    }
                }
            }

            // read requested field into output array
            switch (field)
            {
            case DROP_COORD:
                for (i = 0; i < d_numActDrops; i++)
                {
                    f1[i] = d_actDrop[i].x;
                    f2[i] = d_actDrop[i].y;
                    f3[i] = d_actDrop[i].z;
                }
                break;
            case DROP_VEL:
                for (i = 0; i < d_numActDrops; i++)
                {
                    f1[i] = d_actDrop[i].u;
                    f2[i] = d_actDrop[i].v;
                    f3[i] = d_actDrop[i].w;
                }
                break;
            case DROP_DENS:
                for (i = 0; i < d_numActDrops; i++)
                    f1[i] = d_actDrop[i].dens;
                break;
            case DROP_DIAM:
                for (i = 0; i < d_numActDrops; i++)
                    f1[i] = d_actDrop[i].diam;
                break;
            case DROP_MASS:
                for (i = 0; i < d_numActDrops; i++)
                    f1[i] = d_actDrop[i].mass;
                break;
            case DROP_NO:
                for (i = 0; i < d_numActDrops; i++)
                    f1[i] = d_actDrop[i].coun;
                break;
            case DROP_TEMP:
                for (i = 0; i < d_numActDrops; i++)
                    f1[i] = d_actDrop[i].temp;
                break;
            }
            return d_numActDrops;
        }
        else
            return 0; // no drops in this step
    }

    // any kind of scalar field
    else
    {

        if (field < SCALAR)
            field += 7 - U; // correct for U=7
        else
            field += 21 - SCALAR; // correct for SCALAR1=21
        int i;

        if ((field > 200) || (ndata[field][1] < 1))
        {
            delete[] starField;
            return -1;
        }

        input.seekBlock(ndata[field][1]); // +step*blk_per_step);
        if (field != 20)
            input.skipFloat(nbc + 1);
        input.read(starField, ntcell);
        for (i = 0; i < elements; i++)
            if (i >= 0)
                f1[i] = starField[redCovToStar[i]];
    }

    delete[] starField;

    if (input.fail())
        return -1;
    else
        return 0;
}
}

///////////////////////////////////////////////////////////////////
//
//    Destructor
//
///////////////////////////////////////////////////////////////////

void File29::findHeaders()
{
    FILE *analysis = NULL;

    dumper("Analysing transient StarCD file");

    input.seekBlock(1);
    headerBlock[0] = 1;
    char buffer[128];
    bool verboseAnalysis;
    if (getenv("READSTAR_VERBOSE_ANALYSIS"))
        verboseAnalysis = true;
    else
        verboseAnalysis = false;

    if (verboseAnalysis)
    {
        analysis = fopen("analysis.txt", "w");
        fprintf(analysis, "%c %-6s %-6s %-8s %-8s %-8s %-8s\n", ' ', "blk", "res", "ve", "nc", "nb", "iter");
        fprintf(analysis, "===============================================\n");
    }

    int res = input.read(&iter, 2048);

    // use these field to identify headers
    int ve = lvers;
    int nc = ncell;
    int nb = nbc;
    int aktBlk = 1;

    if (verboseAnalysis)
    {
        fprintf(analysis, "%c %-6d %-6d %-8d %-8d %-8d %-8d\n", 'M', aktBlk, res, ve, nc, nb, iter);
    }

    numSteps = 0;

    while (res > 0 && iter != -1)
    {
        headerBlock[numSteps] = aktBlk;
        d_numDrops[numSteps] = ndrop9;
        //cerr << "Found Step #" << iter << " starting at block " << aktBlk << endl;
        numSteps++;

        //pass all fields we know, seek behind for next step
        actualHeaderBlock = aktBlk - 1;
        analyseHeader();
        input.seekBlock(nextHeaderBlock);
        aktBlk = nextHeaderBlock - 1;

        res = input.read(&iter, 2048);
        if (verboseAnalysis)
        {
            fprintf(analysis, "%c %-6d %-6d %-8d %-8d %-8d %-8d\n", '+', aktBlk, res, ve, nc, nb, iter);
        }
        aktBlk++;
        while (res > 0
               && (ve != lvers || nc != ncell || nb != nbc)
               && (iter != -1))
        {
            res = input.read(&iter, 2048);
            if (verboseAnalysis)
            {
                fprintf(analysis, "%c %-6d %-6d %-8d %-8d %-8d %-8d\n", ' ', aktBlk, res, ve, nc, nb, iter);
            }
            aktBlk++;
        }

        //sprintf(buffer,"Step %4d t=%9f",iter,time);
        //dumper(buffer);
    }

    if (res == 8192 && iter == -1)
    {
        sprintf(buffer, "Transient StarCD file with %d steps, correctly terminated",
                numSteps);
        dumper(buffer);
        headerBlock[numSteps] = aktBlk; // save header block for termination, too
    }
    else
    {
        sprintf(buffer, "Transient StarCD file with %d steps, not terminated",
                numSteps - 1);
        dumper(buffer);
        numSteps--; // do not use last step
    }

    // make sure we have a valis starting block
    input.seekBlock(1);
    input.read(&iter, 2048);

    input.resetErrorFlag();
}

// set the dump device: must be a pointer to 'void funct(const char *)'
void File29::setDumper(void (*newDumper)(const char *))
{
    dumper = newDumper;
}

///////////////////////////////////////////////////////////////////
//
//    Destructor
//
///////////////////////////////////////////////////////////////////

File29::~File29()
{
    delete[] d_actDrop;
}

//////////////////////////////////////////////////////////////////////////////

int File29::isFile(const char *filename)
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
