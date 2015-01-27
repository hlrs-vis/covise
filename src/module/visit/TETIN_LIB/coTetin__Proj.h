/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _CO_TETIN__DELETE_PROJECT_H_
#define _CO_TETIN__DELETE_PROJECT_H_

#include "iostream.h"
#include "coTetinCommand.h"

// 07.10.99

/**
 * Class coTetin__Proj implements "project points" command
 *
 */

class coTetin__Proj : public coTetinCommand
{

private:
    /// Copy-Constructor: NOT IMPLEMENTED
    coTetin__Proj(const coTetin__Proj &){};

    /// Assignment operator: NOT IMPLEMENTED
    coTetin__Proj &operator=(const coTetin__Proj &)
    {
        return *this;
    };

    /// Default constructor:
    coTetin__Proj()
        : n_points(0)
        , points(0)
        , dir(0)
        , n_families(0)
        , family_names(0){};

    // ===================== the command's data =====================

    // value
    int n_points; // number of points to project
    float *points; // points to project (3*n_points)
    float *dir; // direction vector to project (3)
    int n_families; // number of families to project to
    char **family_names; // names of families

public:
    /// read from file
    coTetin__Proj(istream &str, int binary, ostream &ostr = cerr);

    /// read from memory
    coTetin__Proj(int *&intDat, float *&floatDat, char *&charDat);

    coTetin__Proj(int n_pnts, float *pnts, float *direct,
                  int n_fam, const char **names);

    /// Destructor
    virtual ~coTetin__Proj();

    /// whether object is valid
    virtual int isValid() const;

    /// count size required in fields
    virtual void addSizes(int &numInt, int &numFloat, int &numChar) const;

    /// put my data to a given set of pointers
    virtual void getBinary(int *&intDat, float *&floatDat, char *&charDat) const;

    /// print to a stream in Tetin format
    virtual void print(ostream &str) const;

    // ===================== command-specific functions =====================
};
#endif
