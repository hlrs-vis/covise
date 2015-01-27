/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _CO_TETIN__DEFINE_MODEL_H_
#define _CO_TETIN__DEFINE_MODEL_H_

#include "iostream.h"
#include "coTetinCommand.h"

// 07.06.99

/**
 * Class coTetin__defModel implements Tetin file "define_model" command
 *
 */
class coTetin__defModel : public coTetinCommand
{

private:
    /// Copy-Constructor: NOT IMPLEMENTED
    coTetin__defModel(const coTetin__defModel &){};

    /// Assignment operator: NOT IMPLEMENTED
    coTetin__defModel &operator=(const coTetin__defModel &)
    {
        return *this;
    };

    /// Default constructor: NOT IMPLEMENTED
    coTetin__defModel(){};

    // ===================== the command's data =====================

    int d_flagsSet; //  vvv --- bit field which are args are given

    float d_size; //        must be given
    float d_nat_size; //   1    float >=0 default?? , >=0
    int d_refinement; //   2    default ??? optional, >=1
    float d_ref_size; //   4    float ???
    float d_edge_crit; //   8

public:
    /// read from file
    coTetin__defModel(istream &str, int binary);

    /// read from memory
    coTetin__defModel(int *&, float *&, char *&);

    /// Destructor
    virtual ~coTetin__defModel();

    /// whether object is valid
    virtual int isValid() const;

    /// count size required in fields
    virtual void addSizes(int &numInt, int &numFloat, int &numChar) const;

    /// put my data to a given set of pointers
    virtual void getBinary(int *&intDat, float *&floatDat, char *&charDat) const;

    /// print to a stream in Tetin format
    virtual void print(ostream &str) const;

    // ===================== command-specific functions =====================

    /// direct constructor
    coTetin__defModel(float size, float nat_size, int refinement,
                      float ref_size, float edge_crit);
};
#endif
