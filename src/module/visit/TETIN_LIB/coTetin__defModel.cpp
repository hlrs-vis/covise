/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "coTetin__defModel.h"
#include <string.h>

/// read from file
coTetin__defModel::coTetin__defModel(istream &str, int binary)
    : coTetinCommand(coTetin::DEFINE_MODEL)
{
    if (binary)
    {
    }
    else
    {
        char lineBuf[4096];

        // no flags set so far
        d_flagsSet = 0;

        // first get the size: this MUST be given
        if (!(str >> d_size))
        {
            d_flagsSet = -1; // mark illegal
        }

        // defaults
        d_nat_size = 0.0;
        d_refinement = 1;
        d_edge_crit = 0.2;

        // get first line: parameters
        getLine(lineBuf, 4096, str);
        if (getOption(lineBuf, "natural_size", d_nat_size))
            d_flagsSet |= 1;
        if (getOption(lineBuf, "natural_size_refinemnt", d_refinement))
            d_flagsSet |= 2;
        if (getOption(lineBuf, "reference_size", d_ref_size))
            d_flagsSet |= 4;
        if (getOption(lineBuf, "edge_criterion", d_edge_crit))
            d_flagsSet |= 8;

        if ((d_nat_size < 0.0) || (d_refinement < 1)
            || (d_edge_crit <= 0.0) || (d_edge_crit >= 1.0))
        {
            cerr << "coTetin__defModel: Illegal values" << endl;
            d_flagsSet = -1;
        }
    }
}

/// read from memory
coTetin__defModel::coTetin__defModel(int *&intDat, float *&floatDat, char *&)
    : coTetinCommand(coTetin::DEFINE_MODEL)
{
    d_flagsSet = *intDat++;
    d_size = *floatDat++;
    d_nat_size = *floatDat++;
    d_refinement = *intDat++;
    d_ref_size = *floatDat++;
    d_edge_crit = *floatDat++;
}

/// Destructor
coTetin__defModel::~coTetin__defModel()
{
}

/// check whether Object is valid
int coTetin__defModel::isValid() const
{
    if ((d_flagsSet == -1) || !d_comm)
        return 0;
    else
        return 1;
}

/// count size required in fields
void coTetin__defModel::addSizes(int &numInt, int &numFloat, int &) const
{
    // parameters
    numInt += 3;
    numFloat += 4;
}

/// put my data to a given set of pointers
void coTetin__defModel::getBinary(int *&intDat, float *&floatDat, char *&charDat) const
{
    // copy the command's name
    *intDat++ = d_comm;

    // copy the data
    *intDat++ = d_flagsSet;
    *floatDat++ = d_size;
    *floatDat++ = d_nat_size;
    *intDat++ = d_refinement;
    *floatDat++ = d_ref_size;
    *floatDat++ = d_edge_crit;
}

/// print to a stream in Tetin format
void coTetin__defModel::print(ostream &str) const
{
    if (isValid())
    {
        str << "define_model " << d_size;
        if (d_flagsSet & 1)
            str << " natural_size " << d_nat_size;
        if (d_flagsSet & 2)
            str << " natural_size_refnmnt " << d_refinement;
        if (d_flagsSet & 4)
            str << " reference_size " << d_ref_size;
        if (d_flagsSet & 8)
            str << " edge_criterion " << d_edge_crit;
        str << endl;
    }
    else
        str << "// invalid define_model command skipped" << endl;
}

// ===================== command-specific functions =====================

/// direct constructor
coTetin__defModel::coTetin__defModel(float size, float nat_size, int refinement, float ref_size, float edge_crit)
{
    d_flagsSet = 15;
    d_size = size;
    d_nat_size = nat_size;
    d_refinement = refinement;
    d_ref_size = ref_size;
    d_edge_crit = edge_crit;
}

/// would need the defaults for all values to implement proper get-fct... ICEM
