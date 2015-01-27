/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef __CO_SIO_READ_POINTS_H
#define __CO_SIO_READ_POINTS_H

#include <api/coModule.h>
using namespace covise;

class coPoint /// a point as part of a linked list, with a color
{
public:
    float _xyz[3]; ///< 3D point coordinate
    float _size; ///< point size (diameter)
    char *_name; ///< text related to point

    coPoint()
    {
        _xyz[0] = _xyz[1] = _xyz[2] = 0.0f;
        _size = 0.0f;
        _name = NULL;
    }

    // Copy constructor
    coPoint(coPoint *p)
    {
        int i;
        for (i = 0; i < 3; ++i)
        {
            _xyz[i] = p->_xyz[i];
        }
        _size = p->_size;
        _name = NULL;
        setName(p->_name);
    }

    coPoint(float x, float y, float z, float s = 0.0f, const char *name = NULL)
    {
        _xyz[0] = x;
        _xyz[1] = y;
        _xyz[2] = z;
        _size = s;
        _name = NULL;
        setName(name);
    }

    void setName(const char *name)
    {
        delete[] _name;
        if (name != NULL)
        {
            _name = new char[strlen(name) + 1];
            strcpy(_name, name);
        }
    }

    void print()
    {
        cerr << "x=" << _xyz[0] << ", y=" << _xyz[1] << ", z=" << _xyz[2] << ", size=" << _size << ", name=" << _name << endl;
    }
};

class coReadPoints : public coModule
{
public:
    typedef enum
    {
        TOWNS,
        STATIONS,
        EARTHQUAKES,
        UNKNOWN
    } FileType;
    coReadPoints();
    virtual ~coReadPoints();

private:
    coFileBrowserParam *_pbrFilename; ///< file name of points file
    coFloatParam *_pfsXOffset; ///< value to offset x with
    coFloatParam *_pfsYOffset; ///< value to offset y with
    coFloatParam *_pfsZFactor; ///< factor to multiply z coordinate with
    coFloatParam *_pfsSizeFactor; ///< factor to multiply size with
    coOutputPort *_poPoints; ///< output port: point coordinates
    coOutputPort *_poSizes; ///< output port: point sizes (=diameters)
    std::list<coPoint *> _pointList; ///< list of read points
    FileType _fileType; ///< type of data stored in file

    // Inherited methods:
    int compute();
    void quit();

    // New methods:
    bool parseLine(FILE *, coPoint *);
    bool readFile(FILE *);
    void makeOutputData();
    void freeMemory();
    FileType guessFileType(const char *);
};

#endif
