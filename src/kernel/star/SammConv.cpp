/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "SammConv.h"
#include <covise/covise.h>

#undef VERBOSE

using namespace covise;

/// ----- Prevent auto-generated functions by assert -------

/// Copy-Constructor: NOT IMPLEMENTED
SammConv::SammConv(const SammConv &)
{
    assert(0);
}

/// Assignment operator: NOT  IMPLEMENTED
SammConv &SammConv::operator=(const SammConv &)
{
    assert(0);
    return *this;
}

/// ----- Never forget the Destructor !! -------

SammConv::~SammConv()
{

    delete[] d_convSAMM;
}

static const int s0 = 16, s1 = 17, s2 = 18, s3 = 19, s4 = 20,
                 s5 = 21, s6 = 22, s7 = 23, s8 = 24, s9 = 25,
                 s10 = 26, s11 = 27;

const float SammConv::DEGEN_RATIO = 1.1f; // 10% spare should be enough..

///////////  SAMM conversion base types

const SammConv::ConvertSAMM SammConv::s_samm0 = // Dummy: normal cell
    {
      1,
      { { 0, 1, 2, 3, 4, 5, 6, 7 } },
      0
    };

const SammConv::ConvertSAMM SammConv::s_samm1 = // 1 corner cut off
    {
      8, // # elements = 8
      { // + 8 element definitions
        { 1, 5, 6, 2, s0, s0, s0, s0 },
        { 5, s0, 1, 1, s1, s1, s1, s1 },
        { 1, 5, 4, 0, s1, s1, s1, s1 },
        { 2, s0, s1, s1, 1, 1, 1, 1 },
        { s0, s2, 2, 2, s1, s1, s1, s1 },
        { 1, s2, 0, 0, s1, s1, s1, s1 },
        { 1, 2, s2, s2, s1, s1, s1, s1 },
        { 1, 2, 3, 0, s2, s2, s2, s2 }
      },
      1
    };

const SammConv::ConvertSAMM SammConv::s_samm2 = // 1 edge cut off
    {
      3,
      { { 1, 2, s2, s2, 5, 6, s3, s3 },
        { 1, s2, s1, s1, 5, s3, s0, s0 },
        { 1, s1, 0, 0, 5, s0, 4, 4 } },
      2
    };

const SammConv::ConvertSAMM SammConv::s_samm2a = // 1 edge cut off ######
    {
      3,
      { { 1, 2, s0, s0, 5, 6, s1, s1 },
        { 1, s0, s3, s3, 5, s1, s2, s2 },
        { 1, s3, 0, 0, 5, s2, 4, 4 } },
      2
    };

const SammConv::ConvertSAMM SammConv::s_samm3 = // 2 adjacent edges cut off
    {
      6,
      { { 1, 2, s0, s0, s1, s1, s1, s1 },
        { 1, s0, s4, s4, s1, s1, s1, s1 },
        { ///
          0, 1, s4, s4, s1, s1, s1, s1
        },
        { 1, s1, 0, 0, s2, s2, s2, s2 },
        { ///
          4, 5, 1, 0, s2, s2, s2, s2
        },
        { 0, s1, s4, s4, 4, s2, s3, s3 } },
      3
    };

const SammConv::ConvertSAMM SammConv::s_samm4 = // 3 adjacent edges (one corner)
    {
      5,
      { { s1, s4, s3, s2, 1, 1, 1, 1 },
        { 5, s2, s3, s3, 1, 1, 1, 1 },
        { s1, s0, s5, s4, 1, 1, 1, 1 },
        { s1, 2, s0, s0, 1, 1, 1, 1 },
        { s4, s5, 0, 0, 1, 1, 1, 1 } },
      4
    };

const SammConv::ConvertSAMM SammConv::s_samm5 = // 1 face + 1 vertex
    {
      3,
      { { 5, s3, s2, 1, s4, s4, s4, s4 },
        { 1, s2, s0, s0, s4, s4, s4, s4 },
        { 1, s2, s1, 0, s0, s0, s0, s0 } },
      5
    };

// This is just a dummy for filling the 'standard' table
const SammConv::ConvertSAMM SammConv::s_samm8dummy = {
    5,
    { { s1, s1, s1, s1, s1, s1, s1, s1 },
      { s1, s1, s1, s1, s1, s1, s1, s1 },
      { s1, s1, s1, s1, s1, s1, s1, s1 },
      { s1, s1, s1, s1, s1, s1, s1, s1 },
      { s1, s1, s1, s1, s1, s1, s1, s1 } },
    8
};

const SammConv::ConvertSAMM SammConv::s_samm8[4096] = {
#include "Samm8Split.inc"
};

void SammConv::createSammConv(const ConvertSAMM &base, const char *code,
                              int i0, int i1, int i2, int i3, int i4, int i5, int i6, int i7)
{
    //// convert 8-char 'binary' string to number (e.g. "00000001" -> 1)
    int index = ((code[0] != '0') ? 128 : 0)
                | ((code[1] != '0') ? 64 : 0)
                | ((code[2] != '0') ? 32 : 0)
                | ((code[3] != '0') ? 16 : 0)
                | ((code[4] != '0') ? 8 : 0)
                | ((code[5] != '0') ? 4 : 0)
                | ((code[6] != '0') ? 2 : 0)
                | ((code[7] != '0') ? 1 : 0);

    int part, vert, idx[8] = { i0, i1, i2, i3, i4, i5, i6, i7 };

    int i, j;

    // which points are cut away: fields for the SAMMs make it easier
    int excluded[32] = {
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
    };
    for (i = 0; i < 8; i++)
        if (code[7 - i] != '0')
            excluded[i] = 1;

    ConvertSAMM &samm = d_convSAMM[index];

    // check against double set
    if (samm.numParts)
        cerr << "Tried to set SAMM entry '" << code << "' twice" << endl;

    /// check indices
    for (i = 0; i < 7; i++)
        for (j = i + 1; j < 8; j++)
            if (idx[i] == idx[j])
                cerr << "duplicate rotation index in '" << code << "'" << endl;

    for (part = 0; part < base.numParts; part++)
        for (vert = 0; vert < 8; vert++)
        {
            if (base.conv[part][vert] < 16) // vertex?
                // rotate
                samm.conv[part][vert] = idx[static_cast<int>(base.conv[part][vert])];
            else // no vertex
                // leave it
                samm.conv[part][vert] = base.conv[part][vert];

            /// check that none of the 'excluded' indices is used
            if (excluded[static_cast<int>(samm.conv[part][vert])])
                cerr << "SAMM entry '" << code << "' uses excludede vertex "
                     << base.conv[part][vert] << endl;
        }
    samm.numParts = base.numParts;
    samm.type = base.type;
}

/// Default constructor: create translation table
SammConv::SammConv()
{
    d_convSAMM = new ConvertSAMM[256];

    int i;

#ifdef VERBOSE
    int maxParts = 0;
    for (i = 0; i < 256; i++)
        if (d_convSAMM[i].numParts > maxParts)
            maxParts = d_convSAMM[i].numParts;
    cout << "SammConv::maxParts="
         << maxParts
         << endl;
    cout << "sizeof(SammConv::d_convSAMM) ="
         << sizeof(d_convSAMM)
         << endl;

    cout << "sizeof(SammConv::s_samm8) ="
         << sizeof(SammConv::s_samm8)
         << endl;
#endif

    for (i = 0; i < 256; i++)
        d_convSAMM[i].numParts = 0;

    /// Dummy: normal cell
    createSammConv(s_samm0, "00000000", 0, 1, 2, 3, 4, 5, 6, 7);

    /// Create SAMM-1 cells
    createSammConv(s_samm1, "10000000", 0, 1, 2, 3, 4, 5, 6, 7);
    createSammConv(s_samm1, "01000000", 3, 0, 1, 2, 7, 4, 5, 6);
    createSammConv(s_samm1, "00100000", 2, 3, 0, 1, 6, 7, 4, 5);
    createSammConv(s_samm1, "00010000", 1, 2, 3, 0, 5, 6, 7, 4);
    createSammConv(s_samm1, "00001000", 6, 5, 4, 7, 2, 1, 0, 3);
    createSammConv(s_samm1, "00000100", 5, 4, 7, 6, 1, 0, 3, 2);
    createSammConv(s_samm1, "00000010", 4, 7, 6, 5, 0, 3, 2, 1);
    createSammConv(s_samm1, "00000001", 7, 6, 5, 4, 3, 2, 1, 0);

    /// Create SAMM-2 cells
    createSammConv(s_samm2, "00000011", 3, 7, 4, 0, 2, 6, 5, 1);
    createSammConv(s_samm2, "00010001", 1, 2, 3, 0, 5, 6, 7, 4);
    createSammConv(s_samm2, "00110000", 0, 3, 7, 4, 1, 2, 6, 5);
    createSammConv(s_samm2, "00000110", 0, 4, 5, 1, 3, 7, 6, 2);
    createSammConv(s_samm2, "00100010", 2, 3, 0, 1, 6, 7, 4, 5);
    createSammConv(s_samm2, "01100000", 1, 0, 4, 5, 2, 3, 7, 6);
    createSammConv(s_samm2, "00001100", 1, 5, 6, 2, 0, 4, 7, 3);
    createSammConv(s_samm2, "01000100", 3, 0, 1, 2, 7, 4, 5, 6);
    createSammConv(s_samm2, "11000000", 2, 1, 5, 6, 3, 0, 4, 7);
    createSammConv(s_samm2, "00001001", 4, 5, 1, 0, 7, 6, 2, 3);
    createSammConv(s_samm2, "10001000", 0, 1, 2, 3, 4, 5, 6, 7);
    createSammConv(s_samm2, "10010000", 5, 1, 0, 4, 6, 2, 3, 7);

    /// Create SAMM-3 cells
    createSammConv(s_samm3, "00011001", 2, 6, 7, 3, 1, 5, 4, 0);
    createSammConv(s_samm3, "00010011", 7, 6, 5, 4, 3, 2, 1, 0);
    createSammConv(s_samm3, "00001011", 5, 6, 2, 1, 4, 7, 3, 0);

    createSammConv(s_samm3, "00100011", 3, 7, 4, 0, 2, 6, 5, 1);
    createSammConv(s_samm3, "00100110", 4, 7, 6, 5, 0, 3, 2, 1);
    createSammConv(s_samm3, "00000111", 6, 7, 3, 2, 5, 4, 0, 1);

    createSammConv(s_samm3, "01001100", 5, 4, 7, 6, 1, 0, 3, 2);
    createSammConv(s_samm3, "01000110", 0, 4, 5, 1, 3, 7, 6, 2);
    createSammConv(s_samm3, "00001110", 7, 4, 0, 3, 6, 5, 1, 2);

    createSammConv(s_samm3, "10001001", 6, 5, 4, 7, 2, 1, 0, 3);
    createSammConv(s_samm3, "00001101", 4, 5, 1, 0, 7, 6, 2, 3);
    createSammConv(s_samm3, "10001100", 1, 5, 6, 2, 0, 4, 7, 3);

    createSammConv(s_samm3, "00110001", 6, 2, 1, 5, 7, 3, 0, 4);
    createSammConv(s_samm3, "10010001", 1, 2, 3, 0, 5, 6, 7, 4);
    createSammConv(s_samm3, "10110000", 3, 2, 6, 7, 0, 1, 5, 4);

    createSammConv(s_samm3, "01100010", 7, 3, 2, 6, 4, 0, 1, 5);
    createSammConv(s_samm3, "00110010", 2, 3, 0, 1, 6, 7, 4, 5);
    createSammConv(s_samm3, "01110000", 0, 3, 7, 4, 1, 2, 6, 5);

    createSammConv(s_samm3, "11000100", 4, 0, 3, 7, 5, 1, 2, 6);
    createSammConv(s_samm3, "01100100", 3, 0, 1, 2, 7, 4, 5, 6);
    createSammConv(s_samm3, "11100000", 1, 0, 4, 5, 2, 3, 7, 6);

    createSammConv(s_samm3, "11001000", 0, 1, 2, 3, 4, 5, 6, 7);
    createSammConv(s_samm3, "11010000", 2, 1, 5, 6, 3, 0, 4, 7);
    createSammConv(s_samm3, "10011000", 5, 1, 0, 4, 6, 2, 3, 7);

    /// Create SAMM-4 cells
    createSammConv(s_samm4, "11011000", 0, 1, 2, 3, 4, 5, 6, 7);
    createSammConv(s_samm4, "11100100", 3, 0, 1, 2, 7, 4, 5, 6);
    createSammConv(s_samm4, "01110010", 2, 3, 0, 1, 6, 7, 4, 5);
    createSammConv(s_samm4, "10110001", 1, 2, 3, 0, 5, 6, 7, 4);
    createSammConv(s_samm4, "10001101", 6, 5, 4, 7, 2, 1, 0, 3);
    createSammConv(s_samm4, "01001110", 5, 4, 7, 6, 1, 0, 3, 2);
    createSammConv(s_samm4, "00100111", 4, 7, 6, 5, 0, 3, 2, 1);
    createSammConv(s_samm4, "00011011", 7, 6, 5, 4, 3, 2, 1, 0);

    /// Create SAMM-5 Cells
    createSammConv(s_samm5, "11101100", 4, 0, 3, 7, 5, 1, 2, 6);
    createSammConv(s_samm5, "11100110", 3, 0, 1, 2, 7, 4, 5, 6);
    createSammConv(s_samm5, "11110100", 1, 0, 4, 5, 2, 3, 7, 6);

    createSammConv(s_samm5, "11011100", 0, 1, 2, 3, 4, 5, 6, 7);
    createSammConv(s_samm5, "11111000", 2, 1, 5, 6, 3, 0, 4, 7);
    createSammConv(s_samm5, "11011001", 5, 1, 0, 4, 6, 2, 3, 7);

    createSammConv(s_samm5, "10110011", 6, 2, 1, 5, 7, 3, 0, 4);
    createSammConv(s_samm5, "10111001", 1, 2, 3, 0, 5, 6, 7, 4);
    createSammConv(s_samm5, "11110001", 3, 2, 6, 7, 0, 1, 5, 4);

    createSammConv(s_samm5, "01110110", 7, 3, 2, 6, 4, 0, 1, 5);
    createSammConv(s_samm5, "01110011", 2, 3, 0, 1, 6, 7, 4, 5);
    createSammConv(s_samm5, "11110010", 0, 3, 7, 4, 1, 2, 6, 5);

    createSammConv(s_samm5, "11001110", 5, 4, 7, 6, 1, 0, 3, 2);
    createSammConv(s_samm5, "01101110", 0, 4, 5, 1, 3, 7, 6, 2);
    createSammConv(s_samm5, "01001111", 7, 4, 0, 3, 6, 5, 1, 2);

    createSammConv(s_samm5, "10011101", 6, 5, 4, 7, 2, 1, 0, 3);
    createSammConv(s_samm5, "10001111", 4, 5, 1, 0, 7, 6, 2, 3);
    createSammConv(s_samm5, "11001101", 1, 5, 6, 2, 0, 4, 7, 3);

    createSammConv(s_samm5, "10011011", 2, 6, 7, 3, 1, 5, 4, 0);
    createSammConv(s_samm5, "00111011", 7, 6, 5, 4, 3, 2, 1, 0);
    createSammConv(s_samm5, "00011111", 5, 6, 2, 1, 4, 7, 3, 0);

    /// 763
    createSammConv(s_samm5, "00110111", 3, 7, 4, 0, 2, 6, 5, 1);
    /// 743
    createSammConv(s_samm5, "01100111", 4, 7, 6, 5, 0, 3, 2, 1);
    /// 764
    createSammConv(s_samm5, "00101111", 6, 7, 3, 2, 5, 4, 0, 1);

    /// Create SAMM-8 Cells: not used, just a dummy...
    createSammConv(s_samm8dummy, "11111111", 0, 1, 2, 3, 4, 5, 6, 7);

    // this 'SAMM' does not exist: check it
    assert(d_convSAMM[UNUSED].numParts == 0);
}

/////////////////////////////////////////////////////////////////////////////
namespace covise
{
inline int getSamm8case(const int *v)
{
    int mask = 0;
    if (v[0] == v[1])
        mask |= 1;
    if (v[1] == v[2])
        mask |= 2;
    if (v[2] == v[3])
        mask |= 4;
    if (v[3] == v[4])
        mask |= 8;
    if (v[4] == v[5])
        mask |= 16;
    if (v[5] == v[0])
        mask |= 32;

    if (v[6] == v[7])
        mask |= 64;
    if (v[7] == v[8])
        mask |= 128;
    if (v[8] == v[9])
        mask |= 256;
    if (v[9] == v[10])
        mask |= 512;
    if (v[10] == v[11])
        mask |= 1024;
    if (v[11] == v[6])
        mask |= 2048;

    return mask;
}

// Create debug output for SAMM-8
static void debugSamm8(FILE *debug, const int *sammVert,
                       const SammConv::ConvertSAMM *convert)
{
    (void)convert;
    const int &sv0 = sammVert[0];
    const int &sv1 = sammVert[1];
    const int &sv2 = sammVert[2];
    const int &sv3 = sammVert[3];
    const int &sv4 = sammVert[4];
    const int &sv5 = sammVert[5];
    const int &sv6 = sammVert[6];
    const int &sv7 = sammVert[7];
    const int &sv8 = sammVert[8];
    const int &sv9 = sammVert[9];
    const int &sv10 = sammVert[10];
    const int &sv11 = sammVert[11];

    char outstr[256];

    outstr[0] = '\0';

    int s0type = 0;
    int s1type = 0;
    int s2type = 0;
    if (sv0 == sv1)
    {
        s0type |= 1;
        strcat(outstr, " 0=1");
    }
    else
        strcat(outstr, " 0 1");
    if (sv1 == sv2)
    {
        s0type |= 2;
        strcat(outstr, "=2");
    }
    else
        strcat(outstr, " 2");
    if (sv2 == sv3)
    {
        s0type |= 4;
        strcat(outstr, "=3");
    }
    else
        strcat(outstr, " 3");
    if (sv3 == sv4)
    {
        s0type |= 8;
        strcat(outstr, "=4");
    }
    else
        strcat(outstr, " 4");
    if (sv4 == sv5)
    {
        s0type |= 16;
        strcat(outstr, "=5");
    }
    else
        strcat(outstr, " 5");
    if (sv5 == sv0)
    {
        s0type |= 32;
        strcat(outstr, "=0");
    }
    else
        strcat(outstr, " 0");

    strcat(outstr, " |");

    if (sv6 == sv7)
    {
        s1type |= 1;
        strcat(outstr, " 6=7");
    }
    else
        strcat(outstr, " 6 7");
    if (sv7 == sv8)
    {
        s1type |= 2;
        strcat(outstr, "=8");
    }
    else
        strcat(outstr, " 8");
    if (sv8 == sv9)
    {
        s1type |= 4;
        strcat(outstr, "=9");
    }
    else
        strcat(outstr, " 9");
    if (sv9 == sv10)
    {
        s1type |= 8;
        strcat(outstr, "=10");
    }
    else
        strcat(outstr, " 10");
    if (sv10 == sv11)
    {
        s1type |= 16;
        strcat(outstr, "=11");
    }
    else
        strcat(outstr, " 11");
    if (sv11 == sv6)
    {
        s1type |= 32;
        strcat(outstr, "=6");
    }
    else
        strcat(outstr, " 6");

    strcat(outstr, " |");

    if (sv0 == sv6)
    {
        s2type |= 1;
        strcat(outstr, " 0=6");
    }
    else
        strcat(outstr, "    ");
    if (sv1 == sv7)
    {
        s2type |= 2;
        strcat(outstr, " 1=7");
    }
    else
        strcat(outstr, "    ");
    if (sv2 == sv8)
    {
        s2type |= 4;
        strcat(outstr, " 2=8");
    }
    else
        strcat(outstr, "    ");
    if (sv3 == sv9)
    {
        s2type |= 8;
        strcat(outstr, " 3=9");
    }
    else
        strcat(outstr, "    ");
    if (sv4 == sv10)
    {
        s2type |= 16;
        strcat(outstr, " 4=10");
    }
    else
        strcat(outstr, "     ");
    if (sv5 == sv11)
    {
        s2type |= 32;
        strcat(outstr, " 5=11");
    }
    else
        strcat(outstr, "     ");
    fprintf(debug, "SAMM-8 : %02d-%02d-%02d  %s\n", s0type, s1type, s2type, outstr);
}
}
/////////////////////////////////////////////////////////////////////////////

int *SammConv::convertSamm(StarModelFile::CellTabEntry *&cellTab,
                           StarModelFile::SammTabEntry *sammTab,
                           StarModelFile::CellTypeEntry *cellType,
                           int &numElem, int mxtb,
                           void (*dumper)(const char *))
{
    int verbose = (NULL != getenv("VERBOSE_STARLIB_SAMM"));

    // create a list with SAMM types
    unsigned char *ctype = new unsigned char[numElem];

    int cell, numNewCell = 0;
    for (cell = 0; cell < numElem; cell++)
    {
        if (cellTab[cell].ictID > 0 // allowed type ID
            && cellTab[cell].ictID <= mxtb
               // FLUI or SOLI
            && (cellType[cellTab[cell].ictID - 1].ctype == 1
                || cellType[cellTab[cell].ictID - 1].ctype == 2))
        {
            ctype[cell] = 0;
            if (cellTab[cell].vertex[0] == 0)
                ctype[cell] |= 1;
            if (cellTab[cell].vertex[1] == 0)
                ctype[cell] |= 2;
            if (cellTab[cell].vertex[2] == 0)
                ctype[cell] |= 4;
            if (cellTab[cell].vertex[3] == 0)
                ctype[cell] |= 8;
            if (cellTab[cell].vertex[4] == 0)
                ctype[cell] |= 16;
            if (cellTab[cell].vertex[5] == 0)
                ctype[cell] |= 32;
            if (cellTab[cell].vertex[6] == 0)
                ctype[cell] |= 64;
            if (cellTab[cell].vertex[7] == 0)
                ctype[cell] |= 128;

            // make sure we know this type: SAMM-8 uses dummy here.
            if (d_convSAMM[ctype[cell]].numParts > 0)
                numNewCell += d_convSAMM[ctype[cell]].numParts;

            // some are missing, but not complete upper
            else
            {
                char string[9];
                string[8] = '\0';
                string[0] = (ctype[cell] & 1) ? '1' : '0';
                string[1] = (ctype[cell] & 2) ? '1' : '0';
                string[2] = (ctype[cell] & 4) ? '1' : '0';
                string[3] = (ctype[cell] & 8) ? '1' : '0';
                string[4] = (ctype[cell] & 16) ? '1' : '0';
                string[5] = (ctype[cell] & 32) ? '1' : '0';
                string[6] = (ctype[cell] & 64) ? '1' : '0';
                string[7] = (ctype[cell] & 128) ? '1' : '0';
                ctype[cell] = 0;
                numNewCell++;
                cerr << "Unknown Celltype " << string
                     << " used for SOLI or FLUI: not converted" << endl;
            }
        }
        else if (cellTab[cell].vertex[0]) // 1st !=0 -> may be baffle or so
        {
            ctype[cell] = 0; // keep it
            numNewCell++;
        }
        else
            ctype[cell] = UNUSED; /// discard empties
    }

    // leave some spare for 'degenerated' type cells
    numNewCell = (int)(numNewCell * DEGEN_RATIO);

    StarModelFile::CellTabEntry *newCell
        = new StarModelFile::CellTabEntry[numNewCell];
    int *newToOldCell = new int[numNewCell];

    numNewCell = 0;
    int sammTabIdx = 0;

    FILE *debug = NULL;
    if (verbose)
        debug = fopen("zzSAMM", "w");

    /// Convert all elements
    for (cell = 0; cell < numElem; cell++)
    {
        // standard cell
        if (ctype[cell] == 0)
        {
            newCell[numNewCell] = cellTab[cell];
            newToOldCell[numNewCell] = cell;
            numNewCell++;
        }

        // all other SAMM types
        else if (ctype[cell] != UNUSED)
        {
            // skip SAMM point definitions of inactive cells
            while (sammTab[sammTabIdx].ictID < 0)
                sammTabIdx++;

            const int *sammVert = sammTab[sammTabIdx].vertex;
            const int *cellVert = cellTab[cell].vertex;

            // now fond conversion table: special for SAMM-8
            const ConvertSAMM *convert;
            if (ctype[cell] == 255)
                convert = &s_samm8[getSamm8case(sammVert)];
            else
                convert = &d_convSAMM[ctype[cell]];

            if (debug)
            {

                // SAMM-8 specific remark line
                if (ctype[cell] == 8)
                    debugSamm8(debug, sammVert, convert);

                // cellTab vertices
                fprintf(debug,
                        " From: #%-7d: %7d %7d %7d %7d %7d %7d %7d %7d - SAMM-%d\n", cell,
                        cellVert[0], cellVert[1], cellVert[2], cellVert[3],
                        cellVert[4], cellVert[5], cellVert[6], cellVert[7],
                        convert->type);

                // sammTab vertices
                fprintf(debug,
                        "       %7s : %7d %7d %7d %7d %7d %7d %7d %7d \n", "",
                        sammVert[0], sammVert[1], sammVert[2], sammVert[3],
                        sammVert[4], sammVert[5], sammVert[6], sammVert[7]);

                // 12 points for SAMM-8
                if (convert->type == 8)
                    fprintf(debug,
                            "       %7s : %7d %7d %7d %7d \n", "",
                            sammVert[8], sammVert[9], sammVert[10], sammVert[11]);

                fprintf(debug, " ----> %d Parts:\n", convert->numParts);
            }

            int iPart;

            // loop over SAMM parts
            for (iPart = 0; iPart < convert->numParts; iPart++)
            {
                const char *conv = convert->conv[iPart];

                // loop over part's vertices
                for (int iVert = 0; iVert < 8; iVert++)
                {
                    newCell[numNewCell].vertex[iVert]
                        = (conv[iVert] & 16) ? sammVert[int(conv[iVert] & 15)] : cellVert[int(conv[iVert])];
                }
                // set the cell type
                newCell[numNewCell].ictID = cellTab[cell].ictID;

                if (debug)
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

                newToOldCell[numNewCell] = cell;
                numNewCell++;
            }
            sammTabIdx++;

            if (debug)
                fprintf(debug, "---\n");

        } // loop over SAMM parts

        if (cell % 500000 == 499999)
        {
            char tick[512];
            sprintf(tick, "processed %d cells", cell + 1);
            dumper(tick);
        }
    }

    if (debug)
        fclose(debug);

    delete[] ctype;

    // set new values and delete old cell table
    numElem = numNewCell;
    delete[] cellTab;
    cellTab = newCell;

    return newToOldCell;
}
