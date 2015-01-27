/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _CO_TETIN__TRIANGULATE_FAMILY_H_
#define _CO_TETIN__TRIANGULATE_FAMILY_H_

#include "iostream.h"
#include "coTetinCommand.h"

// 04.06.99

/**
 * Class coTetin__trianFam implements "triangulate families" command
 *
 */
class coTetin__trianFam : public coTetinCommand
{

private:
    /// Copy-Constructor: NOT IMPLEMENTED
    coTetin__trianFam(const coTetin__trianFam &){};

    /// Assignment operator: NOT IMPLEMENTED
    coTetin__trianFam &operator=(const coTetin__trianFam &)
    {
        return *this;
    };

    /// Default constructor:
    coTetin__trianFam()
        : n_families(0)
        , family_names(0){};

    // ===================== the command's data =====================

    // value
    int n_families;
    char **family_names;

public:
    /// read from file
    coTetin__trianFam(istream &str, int binary, ostream &ostr = cerr);

    /// read from memory
    coTetin__trianFam(int *&intDat, float *&floatDat, char *&charDat);

    coTetin__trianFam(int n_fam, char **names);

    /// Destructor
    virtual ~coTetin__trianFam();

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
