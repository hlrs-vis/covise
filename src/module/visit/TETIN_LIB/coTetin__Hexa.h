/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _CO_TETIN__HEXA_H_
#define _CO_TETIN__HEXA_H_

#include "iostream.h"
#include "coTetinCommand.h"

// 04.06.99

/**
 * Class coTetin__Hexa implements Tetin file "Hexa" command
 *
 */
class coTetin__Hexa : public coTetinCommand
{

private:
    /// Copy-Constructor: NOT IMPLEMENTED
    coTetin__Hexa(const coTetin__Hexa &){};

    /// Assignment operator: NOT IMPLEMENTED
    coTetin__Hexa &operator=(const coTetin__Hexa &)
    {
        return *this;
    };

    /// Default constructor:
    coTetin__Hexa()
        : tetin(0)
        , replay(0)
        , config_dir(0)
        , outp_intf(0)
        , write_blocking(0){};

    // ===================== the command's data =====================

    // value
    char *tetin; // name of tetin file
    char *replay; // name of replay file
    char *config_dir; // path of configuration directory
    char *outp_intf; // name of output interface
    int write_blocking; // if <> 0: write blocking file after mesh creation

public:
    /// read from file
    coTetin__Hexa(istream &str, int binary, ostream &ostr = cerr);

    /// read from memory
    coTetin__Hexa(int *&intDat, float *&floatDat, char *&charDat);

    coTetin__Hexa(char *tetinf, char *replayf, char *config_d,
                  char *outp_intff, int write_bl = 0);

    /// Destructor
    virtual ~coTetin__Hexa();

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
