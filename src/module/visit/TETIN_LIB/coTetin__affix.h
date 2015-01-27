/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _CO_TETIN__AFFIX_H_
#define _CO_TETIN__AFFIX_H_

#include "iostream.h"
#include "coTetinCommand.h"

// 04.06.99

/**
 * Class coTetin__affix implements Tetin file "affix" command
 *
 */
class coTetin__affix : public coTetinCommand
{

private:
    /// Copy-Constructor: NOT IMPLEMENTED
    coTetin__affix(const coTetin__affix &){};

    /// Assignment operator: NOT IMPLEMENTED
    coTetin__affix &operator=(const coTetin__affix &)
    {
        return *this;
    };

    /// Default constructor: NOT IMPLEMENTED
    coTetin__affix(){};

    // ===================== the command's data =====================

    // Flag value
    int d_flag;

public:
    /// read from file
    coTetin__affix(istream &str, int binary);

    /// read from memory
    coTetin__affix(int *&intDat, float *&floatDat, char *&charDat);

    /// Destructor
    virtual ~coTetin__affix();

    /// whether object is valid
    virtual int isValid() const;

    /// count size required in fields
    virtual void addSizes(int &numInt, int &numFloat, int &numChar) const;

    /// put my data to a given set of pointers
    virtual void getBinary(int *&intDat, float *&floatDat, char *&charDat) const;

    /// print to a stream in Tetin format
    virtual void print(ostream &str) const;

    // ===================== command-specific functions =====================

    /// construct from flag value
    coTetin__affix(int val);

    /// get the flag's value
    int getValue() const
    {
        return d_flag;
    }

    /// set the flag's value
    void setValue(int val);
};
#endif
