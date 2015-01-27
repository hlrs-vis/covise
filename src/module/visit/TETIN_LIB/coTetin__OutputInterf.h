/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _CO_TETIN__OUTPUTINTERF_H_
#define _CO_TETIN__OUTPUTINTERF_H_

#include "iostream.h"
#include "coTetinCommand.h"

// 04.06.99

/**
 * Class coTetin__OutputInterf implements Tetin file "OutputInterf" command
 *
 */
class coTetin__OutputInterf : public coTetinCommand
{

private:
    /// Copy-Constructor: NOT IMPLEMENTED
    coTetin__OutputInterf(const coTetin__OutputInterf &){};

    /// Assignment operator: NOT IMPLEMENTED
    coTetin__OutputInterf &operator=(const coTetin__OutputInterf &)
    {
        return *this;
    };

    /// Default constructor:
    coTetin__OutputInterf()
        : outp_intf(0){};

    // ===================== the command's data =====================

    // value
    char *outp_intf; // name of output interface

public:
    /// read from file
    coTetin__OutputInterf(istream &str, int binary, ostream &ostr = cerr);

    /// read from memory
    coTetin__OutputInterf(int *&intDat, float *&floatDat, char *&charDat);

    coTetin__OutputInterf(char *outp_intff);

    /// Destructor
    virtual ~coTetin__OutputInterf();

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
