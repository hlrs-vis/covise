/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _CO_TETIN__MATERIAL_POINT_H_
#define _CO_TETIN__MATERIAL_POINT_H_

#include "iostream.h"
#include "coTetinCommand.h"

// 09.06.99

/**
 * Class coTetin__matPoint implements Tetin file "material_point" command
 *
 */
class coTetin__matPoint : public coTetinCommand
{

private:
    /// Copy-Constructor: NOT IMPLEMENTED
    coTetin__matPoint(const coTetin__matPoint &){};

    /// Assignment operator: NOT IMPLEMENTED
    coTetin__matPoint &operator=(const coTetin__matPoint &)
    {
        return *this;
    };

    /// Default constructor: NOT IMPLEMENTED
    coTetin__matPoint(){};

    // ===================== the command's data =====================

    float coord[3];
    char *d_family, *d_name;

public:
    /// read from file
    coTetin__matPoint(istream &str, int binary);

    /// read from memory
    coTetin__matPoint(int *&, float *&, char *&);

    /// Destructor
    virtual ~coTetin__matPoint();

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
