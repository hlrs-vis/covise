/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _CO_TETIN__DELETE_FAMILY_H_
#define _CO_TETIN__DELETE_FAMILY_H_

#include "iostream.h"
#include "coTetinCommand.h"

// 07.10.99

/**
 * Class coTetin__delFam implements "delete families" command
 *
 */

enum
{
    FAMILY_DEL_INVALID = 0,
    FAMILY_DEL_ALL,
    FAMILY_DEL_SURF,
    FAMILY_DEL_CURVE
};

class coTetin__delFam : public coTetinCommand
{

private:
    /// Copy-Constructor: NOT IMPLEMENTED
    coTetin__delFam(const coTetin__delFam &){};

    /// Assignment operator: NOT IMPLEMENTED
    coTetin__delFam &operator=(const coTetin__delFam &)
    {
        return *this;
    };

    /// Default constructor:
    coTetin__delFam()
        : n_families(0)
        , family_names(0)
        , type_delete(FAMILY_DEL_INVALID){};

    // ===================== the command's data =====================

    // value
    int n_families;
    char **family_names;
    int type_delete;

public:
    /// read from file
    coTetin__delFam(istream &str, int binary, ostream &ostr = cerr);

    /// read from memory
    coTetin__delFam(int *&intDat, float *&floatDat, char *&charDat);

    coTetin__delFam(int n_fam, char **names, int del_type);

    /// Destructor
    virtual ~coTetin__delFam();

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
