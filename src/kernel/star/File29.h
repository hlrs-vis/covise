/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef __FILE29_H_
#define __FILE29_H_

#include <covise/covise.h>
#include <util/coTypes.h>

#include "istreamBLK.h"
#include "StarFile.h"

namespace covise
{

class STAREXPORT File29 : public StarFile
{
public:
    typedef float float32;
    typedef int int32;
    struct DropRec
    {
        int32 iorg, idt, idc; // FORTRAN header later overwritten by index
        float32 x, y, z, u, v, w;
        float32 dens, diam, mass, coun, temp;
    };

private:
    File29(const File29 &);
    File29 &operator=(const File29 &);
    File29();
    istreamBLK input;

    void analyseHeader();
    long readHeader(long blockNo);
    long actualHeaderBlock, nextHeaderBlock; //,blk_per_step;
    int numSteps;

    // this is a function that dumps a text to somewhere;
    void (*dumper)(const char *);

    // our default: print to stderr
    static void printStderr(const char *text);

    // we need this to check whether file has changed
    ino_t d_inode;
    dev_t d_device;

    // Number of nroplets in timestep
    int d_numDrops[50000];

    // we read Droplets in this structure
    DropRec *d_actDrop;

    // timestep in the structure
    int d_lastDropStep;
    int d_numActDrops;

    // Size factor for elimination or 'bad' particles: eliminate all > fact*avgSize
    float d_elimSizeFactor;

public:
    //    struct Header {

    int iter;
    float time;
    int ncell, //    1
        nbc, //    2
        nbw, //    3
        nbs, //    4
        nbb, //    5
        nnode; //    6
    char title[52]; //    7-
    int nbcyc, //   22
        nbcut, //   23
        nsol, //   24
        numcon, //   25
        lvers, //   26
        ndrop9_, //   27
        ndrop9, //   28
        ncitem, //   29
        nvitem, //   30
        nsitem, //   31
        lmvgrd, //   32
        lct_field1[12], //   33 -  44
        field45_56[12], //   45 -  56
        lsrf_field1[17], //   57 -  73
        lct1_6, //   74
        field75_90[16], //   75 -  90
        lct_field2[3], //   91 -  73
        field94_98[5], //   94 -  98
        ntcell, //   99
        field100, //  100
        lct_field3[47], //  101 - 147
        field148_199[52], //  148 - 199
        nummat, //  200
        nsmat; //  201
    float materials[799]; //  202 - 1000
    int lsrf_field2[47], // 1001 - 1047
        lsrf_field3[47], // 1048 - 1094
        field1095_1345[251], // 1095 - 1345
        lct20, // 1346
        field1347_1799[453], // 1347 - 1799
        lstar[100], // 1800 - 1899  (FTN: lstar[1]->1801
        field1900_1909[10], // 1900 - 1909
        nbsio, // 1910
        nwprsm, // 1911
        nprsm, // 1912
        nbsi, // 1913
        ias, // 1914
        itypen, // 1915
        icoup, // 1916
        mspin, // 1917
        iunben, // 1918
        field1919_2048[130]; // 1919 - 2048

    /// End of Header

    // postconv fields
    int lct[201], lsrf[201], ndata[201][4], ndrec, irasi, irpsm, ircnd;
    /*
            enum { VELOCITY    = 1,             // 1
                   VMAG,                        // 2
                   U,V,W,                       // 3 4 5
                   PRESSURE,                    // 6
                   TE,ED,                       // 7 8
                   TVIS,TEMPERATURE,DENSITY,    // 9 10 11
                   LAMVIS,CP,COND,              // 12 13 14
                   DROP_COORD, DROP_VEL,        // 15 16
                   DROP_DENS, DROP_DIAM,        // 17 18
                   DROP_TEMP, DROP_NO,          // 19 20
      DROP_MASS,                   // 21
      SCALAR } ;                   // 22
      // Scalar_1 = 22
      // Scalar_6 = 27 ...
      */

    ~File29();

    // see setDumper()
    File29(int fd, void (*dumpFunct)(const char *) = NULL);

    int isValid()
    {
        return (ncell != 0);
    }

    long skip_to_step(int stepNo);
    int skip_to_time(float time);
    int skip_to_field(int fieldNo);
    long skip_to_next_field();

    float getRealTime(int step);

    int getVertexCoordinates(int step,
                             float *x, float *y, float *z,
                             int *len);
    int readField(int step, int field,
                  int *starToCov, int elements,
                  float *f1, float *f2, float *f3);
    // Header empty U V W Pressure TE ED VIS Temperature Density Lam_Vis CP COND Scalar
    int headerBlock[50000];

    // get the choice list
    virtual ChoiceList *get_choice(const char **, int) const;

    // number of steps in this file
    int get_num_steps()
    {
        return numSteps;
    }

    // set the dump device: must be a pointer to 'void funct(const char *)'
    void setDumper(void (*newDumper)(const char *));

    void findHeaders();

    /// check whether this is the file we read
    //   1 = true
    //   0 = false
    //  -1 = could not stat file
    int isFile(const char *filename);

    //// number of droplets in given timestep
    // steps are counted 1...n
    int getNumDrops(int step_no) const
    {
        return d_numDrops[step_no - 1];
    }
};
}
#endif
