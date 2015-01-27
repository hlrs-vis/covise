/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef __FILE09_H_
#define __FILE09_H_

#include <covise/covise.h>
#include <util/coTypes.h>

#include "istreamBLK.h"
#include <util/ChoiceList.h>
#include "StarFile.h"

namespace covise
{

class STAREXPORT File09 : public StarFile
{

private:
    File09(const File09 &);
    File09 &operator=(const File09 &);
    File09();
    istreamBLK input;

    // we need this to check whether file has changed
    ino_t d_inode;
    dev_t d_device;

public:
    struct Header
    {
        int iter;
        float time;
        int ncell, nbc, nbw, nbs, nbb, nnode, wpost;
        char title[52];
        int iskip, nbcyc, nbcut, lvers, iretw, ifver, ntrec,
            hrad, npatch, ntpch, numcon, nsitem, field35_50[16],
            lstep, lramp, iterso, itersn, steady, compg,
            nbc012, field58_82[25], mrec29, mrec39;
        float scale9, field86_99[13];
        int ntcell;
        float prepp_2_1;
        int lmvgrd, wsurf;
        float field103_113[11];
        int field114_124[11], npdrop9;
        float field126;
        int nsol, nwset, nbpt;
        int field130, nbnd[6]; // ???
        int nbcuto;
        int field138_199[62];
        int nummat, nsmat;
        float field202_1235[1034];
        int lsrf1236_1345[110],
            lvoid, field1347_1348[2],
            icoup, mspin;
        float field1351_1799[449];
        int lstar[101]; // LSTAR(1...100) -> lstar[0..100]
        int nbncst, nbncen;
        float field1903;
        int nscl, nreg, nbsi, spcyc, mrecmm, nbp23, nclps,
            nwprsm, nprsm, nclp, nclpf, itypen,
            field1916_2048[133];
    };

    int iter;
    float time;
    int ncell, nbc, nbw, nbs, nbb, nnode, wpost;
    char title[52];
    int iskip, nbcyc, nbcut, lvers, iretw, ifver, ntrec,
        hrad, npatch, ntpch, numcon, nsitem, field35_50[16],
        lstep, lramp, iterso, itersn, steady, compg,
        nbc012, field58_82[25], mrec29, mrec39;
    float scale9, field86_99[13];
    int ntcell;
    float prepp_2_1;
    int lmvgrd, wsurf;
    float field103_113[11];
    int field114_124[11], npdrop9;
    float field126;
    int nsol, nwset, nbpt;
    int field130, nbnd[6]; // ???
    int nbcuto;
    int field138_199[62];
    int nummat, nsmat;
    float field202_1235[1034];
    int lsrf1236_1345[110],
        lvoid, field1347_1348[2],
        icoup, mspin;
    float field1351_1800[449];
    int lstar[101];
    int nbncst, nbncen;
    float field1903;
    int nscl, nreg, nbsi, spcyc, mrecmm, nbp23, nclps,
        nwprsm, nprsm, nclp, nclpf, itypen,
        field1916_2048[133];

    ~File09();

    File09(int fd);

    int isValid()
    {
        return (ncell != 0);
    }

    virtual ChoiceList *get_choice(const char **, int) const;

    // read a field: return #of elements read
    int readField(int fieldNo, float *f1,
                  float *f2 = NULL,
                  float *f3 = NULL);

    /// check whether this is the file we read
    //   1 = true
    //   0 = false
    //  -1 = could not stat file
    int isFile(const char *filename);
};
}
#endif
