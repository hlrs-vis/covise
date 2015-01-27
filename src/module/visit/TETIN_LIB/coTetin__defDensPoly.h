/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _CO_TETIN__DEFINE_DENSITY_POLY_H_
#define _CO_TETIN__DEFINE_DENSITY_POLY_H_

#include "iostream.h"
#include "coTetinCommand.h"

// 06.06.99

/**
 * Class coTetin__defDensPoly implements Tetin file "define_density_poly" command
 *
 */
class coTetin__defDensPoly : public coTetinCommand
{

private:
    /// Copy-Constructor: NOT IMPLEMENTED
    coTetin__defDensPoly(const coTetin__defDensPoly &){};

    /// Assignment operator: NOT IMPLEMENTED
    coTetin__defDensPoly &operator=(const coTetin__defDensPoly &)
    {
        return *this;
    };

    /// Default constructor: NOT IMPLEMENTED
    coTetin__defDensPoly(){};

    // ===================== the command's data =====================

    int d_npnts; // number of points: required
    float d_size; // size of polygon: required
    char *d_name; // name of the Object: optional
    float *d_coord; // polygon coordinates

public:
    /// read from file
    coTetin__defDensPoly(istream &str, int binary);

    /// read from memory
    coTetin__defDensPoly(int *&, float *&, char *&);

    /// Destructor
    virtual ~coTetin__defDensPoly();

    /// whether object is valid
    virtual int isValid() const;

    /// count size required in fields
    virtual void addSizes(int &numInt, int &numFloat, int &numChar) const;

    /// put my data to a given set of pointers
    virtual void getBinary(int *&intDat, float *&floatDat, char *&charDat) const;

    /// print to a stream in Tetin format
    virtual void print(ostream &str) const;

    // ===================== command-specific functions =====================

    /// construct from given data
    coTetin__defDensPoly(int npts, float size, float *coord, char *name = NULL);

    /// number of points
    int getNumPoints()
    {
        return d_npnts;
    }

    /// number of points
    float getSize()
    {
        return d_size;
    }

    /// number of points
    const float *getCoord()
    {
        return d_coord;
    }

    /// name of the object
    const char *getName()
    {
        return d_name;
    }
};
#endif
