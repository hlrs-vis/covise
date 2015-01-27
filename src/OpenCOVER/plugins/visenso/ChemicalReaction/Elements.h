/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _ELEMENTS_H
#define _ELEMENTS_H

#include <osg/Vec4>
#include <math.h>

#define ELEMENT_MAX 30

#define GET_ELEMENT_RADIUS(element, charge) pow(ELEMENT_RADIUS[element][3 + charge], 0.27f) * 0.15f

static const int ELEM_H = 1;
static const int ELEM_He = 2;
static const int ELEM_Li = 3;
static const int ELEM_Be = 4;
static const int ELEM_B = 5;
static const int ELEM_C = 6;
static const int ELEM_N = 7;
static const int ELEM_O = 8;
static const int ELEM_F = 9;
static const int ELEM_Ne = 10;
static const int ELEM_Na = 11;
static const int ELEM_Mg = 12;
static const int ELEM_Al = 13;
static const int ELEM_Si = 14;
static const int ELEM_P = 15;
static const int ELEM_S = 16;
static const int ELEM_Cl = 17;
static const int ELEM_Ar = 18;
static const int ELEM_K = 19;
static const int ELEM_Ca = 20;
static const int ELEM_Sc = 21;
static const int ELEM_Ti = 22;
static const int ELEM_V = 23;
static const int ELEM_Cr = 24;
static const int ELEM_Mn = 25;
static const int ELEM_Fe = 26;
static const int ELEM_Co = 27;
static const int ELEM_Ni = 28;
static const int ELEM_Cu = 29;
static const int ELEM_Zn = 30;

static const std::string ELEMENT_SYMBOLS[] = {
    "",
    "H",
    "He",
    "Li",
    "Be",
    "B",
    "C",
    "N",
    "O",
    "F",
    "Ne",
    "Na",
    "Mg",
    "Al",
    "Si",
    "P",
    "S",
    "Cl",
    "Ar",
    "K",
    "Ca",
    "Sc",
    "Ti",
    "V",
    "Cr",
    "Mn",
    "Fe",
    "Co",
    "Ni",
    "Cu",
    "Zn"
};

static const std::string ELEMENT_NAMES[] = {
    "",
    "Wasserstoff",
    "Helium",
    "Lithium",
    "Beryllium",
    "Bor",
    "Kohlenstoff",
    "Stickstoff",
    "Sauerstoff",
    "Fluor",
    "Neon",
    "Natrium",
    "Magnesium",
    "Aluminium",
    "Silicium",
    "Phosphor",
    "Schwefel",
    "Chlor",
    "Argon",
    "Kalium",
    "Calcium",
    "Scandium",
    "Titan",
    "Vanadium",
    "Chrom",
    "Mangan",
    "Eisen",
    "Cobalt",
    "Nickel",
    "Kupfer",
    "Zink"
};

static const osg::Vec4 DEFAULT_COLOR = osg::Vec4(0.7f, 0.7f, 0.7f, 1.0f);

static const osg::Vec4 ELEMENT_COLORS[] = {
    DEFAULT_COLOR,
    osg::Vec4(0.0f, 0.0f, 1.0f, 1.0f), // H
    DEFAULT_COLOR,
    DEFAULT_COLOR,
    DEFAULT_COLOR,
    DEFAULT_COLOR,
    osg::Vec4(0.2f, 0.2f, 0.2f, 1.0f), // C
    osg::Vec4(0.0f, 0.8f, 0.0f, 1.0f), // N
    osg::Vec4(1.0f, 0.0f, 0.0f, 1.0f), // O
    DEFAULT_COLOR,
    DEFAULT_COLOR,
    osg::Vec4(0.7f, 1.0f, 0.7f, 1.0f), // Na
    osg::Vec4(0.8f, 0.8f, 0.5f, 1.0f), // Mg
    osg::Vec4(1.0f, 1.0f, 1.0f, 1.0f), // Al
    DEFAULT_COLOR,
    DEFAULT_COLOR,
    osg::Vec4(1.0f, 1.0f, 0.2f, 1.0f), // S
    osg::Vec4(0.7f, 0.7f, 1.0f, 1.0f), // Cl
    DEFAULT_COLOR,
    DEFAULT_COLOR,
    DEFAULT_COLOR,
    DEFAULT_COLOR,
    DEFAULT_COLOR,
    DEFAULT_COLOR,
    DEFAULT_COLOR,
    DEFAULT_COLOR,
    osg::Vec4(0.5f, 0.5f, 0.5f, 1.0f), // Fe
    DEFAULT_COLOR,
    DEFAULT_COLOR,
    osg::Vec4(1.0f, 0.6f, 0.3f, 1.0f), // Cu
    DEFAULT_COLOR
};

static const float ELEMENT_RADIUS[][7] = { // {-3, -2, -1, 0, 1, 2, 3}
    { 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f },
    { 30.0f, 30.0f, 208.0f, 30.0f, 30.0f, 30.0f, 30.0f }, // H
    { 93.0f, 93.0f, 93.0f, 93.0f, 93.0f, 93.0f, 93.0f },
    { 152.0f, 152.0f, 152.0f, 152.0f, 60.0f, 152.0f, 152.0f }, // Li
    { 111.0f, 111.0f, 111.0f, 111.0f, 111.0f, 31.0f, 111.0f },
    { 80.0f, 80.0f, 80.0f, 80.0f, 80.0f, 80.0f, 80.0f },
    { 77.0f, 77.0f, 77.0f, 77.0f, 77.0f, 77.0f, 77.0f },
    { 73.0f, 73.0f, 73.0f, 73.0f, 73.0f, 73.0f, 73.0f },
    { 74.0f, 140.0f, 74.0f, 74.0f, 74.0f, 74.0f, 74.0f },
    { 71.0f, 71.0f, 136.0f, 71.0f, 71.0f, 71.0f, 71.0f },
    { 112.0f, 112.0f, 112.0f, 112.0f, 112.0f, 112.0f, 112.0f },
    { 186.0f, 186.0f, 186.0f, 186.0f, 95.0f, 186.0f, 186.0f }, // Na
    { 160.0f, 160.0f, 160.0f, 160.0f, 160.0f, 65.0f, 160.0f },
    { 143.0f, 143.0f, 143.0f, 143.0f, 143.0f, 143.0f, 50.0f },
    { 118.0f, 118.0f, 118.0f, 118.0f, 118.0f, 118.0f, 118.0f },
    { 110.0f, 110.0f, 110.0f, 110.0f, 110.0f, 110.0f, 110.0f },
    { 103.0f, 184.0f, 103.0f, 103.0f, 103.0f, 103.0f, 103.0f },
    { 99.0f, 99.0f, 181.0f, 99.0f, 99.0f, 99.0f, 99.0f },
    { 154.0f, 154.0f, 154.0f, 154.0f, 154.0f, 154.0f, 154.0f },
    { 227.0f, 227.0f, 227.0f, 227.0f, 133.0f, 227.0f, 227.0f }, // K
    { 197.0f, 197.0f, 197.0f, 197.0f, 197.0f, 99.0f, 197.0f },
    { 161.0f, 161.0f, 161.0f, 161.0f, 161.0f, 161.0f, 81.0f },
    { 145.0f, 145.0f, 145.0f, 145.0f, 145.0f, 145.0f, 145.0f },
    { 131.0f, 131.0f, 131.0f, 131.0f, 131.0f, 131.0f, 131.0f },
    { 125.0f, 125.0f, 125.0f, 125.0f, 125.0f, 125.0f, 125.0f },
    { 137.0f, 137.0f, 137.0f, 137.0f, 137.0f, 137.0f, 137.0f },
    { 124.0f, 124.0f, 124.0f, 124.0f, 124.0f, 124.0f, 124.0f },
    { 125.0f, 125.0f, 125.0f, 125.0f, 125.0f, 125.0f, 125.0f },
    { 125.0f, 125.0f, 125.0f, 125.0f, 125.0f, 125.0f, 125.0f },
    { 128.0f, 128.0f, 128.0f, 128.0f, 96.0f, 128.0f, 128.0f },
    { 133.0f, 133.0f, 133.0f, 133.0f, 133.0f, 74.0f, 133.0f }
};

#endif
