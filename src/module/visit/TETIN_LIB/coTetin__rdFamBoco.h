/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _CO_TETIN__READ_FAMILY_BOCO_H_
#define _CO_TETIN__READ_FAMILY_BOCO_H_

#include "iostream.h"
#include "coTetinCommand.h"

// 04.06.99

/**
 * Class coTetin__rdFamBoco implements Tetin file "read_family_boco" command
 *
 */
class coTetin__rdFamBoco : public coTetinCommand
{

private:
    /// Copy-Constructor: NOT IMPLEMENTED
    coTetin__rdFamBoco(const coTetin__rdFamBoco &){};

    /// Assignment operator: NOT IMPLEMENTED
    coTetin__rdFamBoco &operator=(const coTetin__rdFamBoco &)
    {
        return *this;
    };

    /// Default constructor: NOT IMPLEMENTED
    coTetin__rdFamBoco(){};

    // ===================== the command's data =====================

    // Flag value
    char *d_boco;

public:
    /// read from file
    coTetin__rdFamBoco(istream &str, int binary);

    /// read from memory
    coTetin__rdFamBoco(int *&intDat, float *&floatDat, char *&charDat);

    /// Destructor
    virtual ~coTetin__rdFamBoco();

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
