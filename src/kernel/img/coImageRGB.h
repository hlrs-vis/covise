/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _CO_IMAGERGB_H_
#define _CO_IMAGERGB_H_
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// CLASS coImageRGB
//
// Initial version: 2004-04-27 [we]
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// (C) 2001 by VirCinity IT Consulting
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// Changes:
//

#include "coImageImpl.h"
extern "C" {
#define CM_NORMAL 0
/* file contains rows of values which
    * are either RGB values (zsize == 3)
    * or greyramp values (zsize == 1) */
typedef struct
{
    unsigned short imagic; /* stuff saved on disk . . */
    unsigned short type;
    unsigned short dim;
    unsigned short xsize;
    unsigned short ysize;
    unsigned short zsize;
    unsigned long min;
    unsigned long max;
    unsigned long wastebytes;
    char name[80];
    unsigned long colormap;
} IMAGE;
extern IMAGE *iopen(const char *, const char *);
extern void getrow(IMAGE *, short *, int, int);
extern void iclose(IMAGE *);
#if !defined(__hpux) && !defined(__linux__)
extern void i_seterror(void (*func)(char *));
#endif
};

/**
 * Class Template for new image formats
 *
 */
namespace covise
{

class coImageRGB : public coImageImpl
{
public:
    // +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    // ++ Constructors / Destructor
    // +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    /// Construct from given filename
    coImageRGB(const char *filename);

    /// Destructor : virtual in case we derive objects
    virtual ~coImageRGB();

    // +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    // ++ Operations
    // +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    /// get pointer to internal data
    virtual unsigned char *getBitmap(int frameno = 0);

    // +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    // ++ Attribute request/set functions
    // +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

protected:
    // +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    // ++  Attributes
    // +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    int dimension_;
    int bytesPerPixel_;
    unsigned char *pixmap_;

private:
    // +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    // ++  Internally used functions
    // +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    void readBW(IMAGE *image);
    void readRGB(IMAGE *image);
    void readRGBA(IMAGE *image);
    // +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    // ++  prevent auto-generated bit copy routines by default
    // +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    /// Copy-Constructor: NOT IMPLEMENTED, checked by assert
    coImageRGB(const coImageRGB &);

    /// Assignment operator: NOT IMPLEMENTED, checked by assert
    coImageRGB &operator=(const coImageRGB &);

    /// Default constructor: NOT IMPLEMENTED, checked by assert
    coImageRGB();
};
#endif
