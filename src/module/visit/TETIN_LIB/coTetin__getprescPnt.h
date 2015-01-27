/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _CO_TETIN__GET_PRESCPNT_H_
#define _CO_TETIN__GET_PRESCPNT_H_

#include "iostream.h"
#include "coTetinCommand.h"

// 27.04.00

/**
 * Class coTetin__getprescPnt gets prescribed points from database
 *
 */
class coTetin__getprescPnt : public coTetinCommand
{

private:
    /// Copy-Constructor: NOT IMPLEMENTED
    coTetin__getprescPnt(const coTetin__getprescPnt &){};

    /// Assignment operator: NOT IMPLEMENTED
    coTetin__getprescPnt &operator=(const coTetin__getprescPnt &)
    {
        return *this;
    };

    /// Default constructor:
    coTetin__getprescPnt()
        : get_family(0)
        , n_names(0)
        , names(0){};

    // ===================== the command's data =====================

    // value
    int get_family; // if <> 0: interpret as family_names else curve_names
    int n_names;
    char **names;

public:
    /// read from file
    coTetin__getprescPnt(istream &str, int binary, ostream &ostr = cerr);

    /// read from memory
    coTetin__getprescPnt(int *&intDat, float *&floatDat, char *&charDat);

    coTetin__getprescPnt(int get_family_in,
                         int n_names_in, const char **names_in);

    /// Destructor
    virtual ~coTetin__getprescPnt();

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
