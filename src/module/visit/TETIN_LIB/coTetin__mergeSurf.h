/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _CO_TETIN__MERGE_SURFACE_H_
#define _CO_TETIN__MERGE_SURFACE_H_

#include "iostream.h"
#include "coTetinCommand.h"

// 04.06.99

/**
 * Class coTetin__mergeSurf implements Tetin file "merge_surface" command
 *
 */
class coTetin__mergeSurf : public coTetinCommand
{

private:
    /// Copy-Constructor: NOT IMPLEMENTED
    coTetin__mergeSurf(const coTetin__mergeSurf &){};

    /// Assignment operator: NOT IMPLEMENTED
    coTetin__mergeSurf &operator=(const coTetin__mergeSurf &)
    {
        return *this;
    };

    /// Default constructor: NOT IMPLEMENTED
    coTetin__mergeSurf(){};

    // ===================== the command's data =====================

    // the two families
    char *d_fam[2];

public:
    /// read from file
    coTetin__mergeSurf(istream &str, int binary);

    /// read from memory
    coTetin__mergeSurf(int *&, float *&, char *&);

    /// Destructor
    virtual ~coTetin__mergeSurf();

    /// whether object is valid
    virtual int isValid() const;

    /// count size required in fields
    virtual void addSizes(int &numInt, int &numFloat, int &numChar) const;

    /// put my data to a given set of pointers
    virtual void getBinary(int *&intDat, float *&floatDat, char *&charDat) const;

    /// print to a stream in Tetin format
    virtual void print(ostream &str) const;

    // ===================== command-specific functions =====================

    /// construct from two strings
    coTetin__mergeSurf(const char *fam0, const char *fam1);

    /// get the family names : [0..1]
    const char *getFamily(int num) const;

    /// get the family names : [0..1]
    void setFamily(int num, const char *name);
};
#endif
