/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _CO_TETIN__BOCO_FILE_H_
#define _CO_TETIN__BOCO_FILE_H_

#include "iostream.h"
#include "coTetinCommand.h"

// 04.06.99

/**
 * Class coTetin__bocoFile implements Tetin file "boco_file" command
 *
 */
class coTetin__bocoFile : public coTetinCommand
{

private:
    /// Copy-Constructor: NOT IMPLEMENTED
    coTetin__bocoFile(const coTetin__bocoFile &){};

    /// Assignment operator: NOT IMPLEMENTED
    coTetin__bocoFile &operator=(const coTetin__bocoFile &)
    {
        return *this;
    };

    /// Default constructor: NOT IMPLEMENTED
    coTetin__bocoFile(){};

    // ===================== the command's data =====================

    // Flag value
    char *d_boco;

public:
    /// read from file
    coTetin__bocoFile(istream &str, int binary);

    /// read from memory
    coTetin__bocoFile(int *&intDat, float *&floatDat, char *&charDat);

    coTetin__bocoFile(char *boco_file);

    /// Destructor
    virtual ~coTetin__bocoFile();

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
