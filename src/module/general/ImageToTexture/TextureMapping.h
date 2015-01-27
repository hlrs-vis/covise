/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

// ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
//  CLASS TextureMapping
//
//  Mapping from the nodes of a polygons object to the texture coordinate space
//
//  Initial version: 2001-04-011 Sergio Leseduarte
// ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
//  (C) 2002 by VirCinity IT Consulting
// ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// Changes:

#ifndef _TEXTURE_MAPPING_
#define _TEXTURE_MAPPING_

class TextureMapping
{
public:
    enum Orientation
    {
        FREE = 0,
        PORTRAIT = 1,
        LANDSCAPE = 2
    };
    enum Fit
    {
        FIT_MANUAL = 0,
        FIT = 1,
        USE_IMAGE = 2
    };
    enum SizeControl
    {
        NONE,
        AUTOMATIC,
        MANUAL
    };

    // if I include LimitMapping:
    //   wo, ho, Width_, Height_
    TextureMapping(const float *x_c, const float *y_c, int no_points,
                   Orientation orientation_hint,
                   Fit fit_hint,
                   SizeControl mode_hint,
                   float Width, float Height, const float *bbox, int mirror);

    TextureMapping(const TextureMapping &tm, const int *indices, int no_points);

    virtual ~TextureMapping();

    void getMapping(const float **mx, const float **my) const;
    void getMapping(const float **mx, const float **my,
                    float minx, float maxx, float miny, float maxy) const;

    Orientation getOrientation() const;
    Fit getFit() const;
    SizeControl getSizeControl() const;

protected:
private:
    Orientation orientation_;
    Fit fit_;
    SizeControl mode_;

    void LimitMapping();
    float limitX_, limitY_; // limits in texture space

    float minX_, maxX_, minY_, maxY_;
    float Width_, Height_; // image size

    int no_points_;

    float *txCoord_[2];
};
#endif
