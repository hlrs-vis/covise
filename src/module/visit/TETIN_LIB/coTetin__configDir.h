/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _CO_TETIN__CONFIGDIR_NAME_H_
#define _CO_TETIN__CONFIGDIR_NAME_H_

#include "iostream.h"
#include "coTetinCommand.h"

// 04.06.99

/**
 * Class coTetin__configDir implements Tetin file "config_dir_name" command
 *
 */
class coTetin__configDir : public coTetinCommand
{

private:
    /// Copy-Constructor: NOT IMPLEMENTED
    coTetin__configDir(const coTetin__configDir &){};

    /// Assignment operator: NOT IMPLEMENTED
    coTetin__configDir &operator=(const coTetin__configDir &)
    {
        return *this;
    };

    /// Default constructor:
    coTetin__configDir()
        : name(0){};

    // ===================== the command's data =====================

    // value
    char *name;

public:
    /// read from file
    coTetin__configDir(istream &str, int binary, ostream &ostr = cerr);

    /// read from memory
    coTetin__configDir(int *&intDat, float *&floatDat, char *&charDat);

    coTetin__configDir(char *filename);

    /// Destructor
    virtual ~coTetin__configDir();

    /// whether object is valid
    virtual int isValid() const;

    /// count size required in fields
    virtual void addSizes(int &numInt, int &numFloat, int &numChar) const;

    /// put my data to a given set of pointers
    virtual void getBinary(int *&intDat, float *&floatDat, char *&charDat) const;

    /// print to a stream in Tetin format
    virtual void print(ostream &str) const;

    // ===================== command-specific functions =====================

    /// get the value
    char *getValue() const
    {
        return name;
    }
};
#endif
