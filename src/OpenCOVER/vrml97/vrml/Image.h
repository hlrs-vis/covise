/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

//
//  Vrml 97 library
//  Copyright (C) 1998 Chris Morley
//

#ifndef _IMAGE_
#define _IMAGE_
//
//  Image document class
//

namespace vrml
{

class Doc;

class VRMLEXPORT Image
{

public:
    Image(const char *url = 0, Doc *relative = 0);
    ~Image();

    bool setURL(const char *url, Doc *relative = 0);

    bool tryURLs(int nUrls, char **urls, Doc *relative = 0);

    const char *url();

    int w()
    {
        return d_w;
    }
    int h()
    {
        return d_h;
    }
    int nc()
    {
        return d_nc;
    }
    int nFrames()
    {
        return d_nFrames;
    }
    unsigned char *pixels()
    {
        return d_pixels;
    }
    unsigned char *pixels(int frame);

    void setSize(int w, int h)
    {
        d_w = w;
        d_h = h;
    }

protected:
    Doc *d_url;
    int d_w, d_h, d_nc, d_nFrames;
    unsigned char *d_pixels;
    unsigned char **d_frame;
    bool newMovies;
};
}
#endif // _IMAGE_
