/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// CLASS coImage
//
// This class @@@
//
// Initial version: 2004-04-27 [we]
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// (C) 2001 by VirCinity IT Consulting
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// Changes:
//

#include "coImage.h"
#include "coBadImage.h"
#include "coBinImage.h"
#include <assert.h>

using namespace covise;

// list of virtual c'Tors
map<string, coImage::vCtor *> *coImage::vCtorList_ = NULL;

// Factory method
coImage::coImage(const char *filename)
{
    // find ending of file name
    const char *lastDot = strrchr(filename, '.');

    // we need one...
    if (!lastDot)
    {
        image_ = new coBadImage("No Suffix found");
    }
    else
    {
        string ending(lastDot + 1);

        map<string, vCtor *>::iterator iter = vCtorList_->find(ending);
        if (iter != vCtorList_->end())
            image_ = ((*iter).second)(filename);
        else
            image_ = new coBadImage("Unknown Image format");
    }
}

coImage::coImage(coBinImage *binImage)
{
    image_ = binImage;
}

coImage::~coImage()
{
    delete image_;
}

// register my type at the dfactory
bool
coImage::registerImageType(const char *suffixes[], coImage::vCtor *cTor)
{
    if (!vCtorList_)
        vCtorList_ = new map<string, vCtor *>;

    const char *const *suffix = suffixes;
    while (*suffix)
    {
        (*vCtorList_)[*suffix] = cTor;
        suffix++;
    }
    return true;
}

/// width of image
int
coImage::getWidth()
{
    return image_->getWidth();
}

/// height of image;
int
coImage::getHeight()
{
    return image_->getHeight();
};

/// get number of Color Channels
int coImage::getNumChannels()
{
    return image_->getNumChannels();
}

/// get number of frames
int coImage::getNumFrames()
{
    return image_->getNumFrames();
}

/// get pointer to internal data
const unsigned char *coImage::getBitmap(int frameno)
{
    return image_->getBitmap(frameno);
}

/// Did we have errors on reading
bool coImage::isBad()
{
    return image_->isBad();
}

/// Error message if problems occurred: empty if ok
const char *coImage::errorMessage()
{
    return image_->errorMessage();
}

void coImage::scaleExp2()
{
    int width = image_->getWidth();
    int height = image_->getHeight();

    int newWidth = 1;
    while (newWidth <= width)
        newWidth <<= 1;
    newWidth >>= 1;

    int newHeight = 1;
    while (newHeight <= height)
        newHeight <<= 1;
    newHeight >>= 1;

    if (newWidth > 1 && newHeight > 1)
        scale(newWidth, newHeight);
}

////////////////////////////////////////////////////////////
// Bi-Linar interpolation between old and new image
// Line weights calculated in loop, column weights
// pre-calculated.
////////////////////////////////////////////////////////////
/// To scale an image, we create a new coBinImage instead of the
/// current image and attach it to our image_ variable
////////////////////////////////////////////////////////////
///

// for debug
#include <sys/types.h>
#ifndef WIN32
#include <unistd.h>
#endif

void COIMAGEEXPORT coImage::scale(int newWidth, int newHeight)
{
    int numChan = image_->getNumChannels();
    int numFrames = image_->getNumFrames();

    // create a new binary image
    coImageImpl *newImg = new coBinImage(newWidth, newHeight, numChan, numFrames);

    int *colIdx0 = new int[newWidth];
    int *colIdx1 = new int[newWidth];
    float *colFct0 = new float[newWidth];
    float *colFct1 = new float[newWidth];
    int oldW = image_->getWidth();
    int oldH = image_->getHeight();

    int frameNo;
    for (frameNo = 0; frameNo < image_->getNumFrames(); frameNo++)
    {
        ////////////////////////////////////////////////////////////
        // Pre-calculate interpolation index and ratios for columns
        // Row Interpolation is between Pixels srcColIdx[i] and
        // srcColIdx[i+1] with factors (1-srcColFct[i]) and srcColFct[i]

        float srcW = (float)oldW;
        float dstW = (float)newWidth;
        int col;
        for (col = 0; col < newWidth; col++)
        {
            // float "index" of row in src image
            float srcColFl = (col + 0.5f) * srcW / dstW - 0.5f;
            int idx = (int)floor(srcColFl);

            colIdx0[col] = idx;
            colIdx1[col] = idx + 1;

            if (colIdx0[col] < 0)
                colIdx0[col] = 0;
            if (colIdx1[col] >= oldW - 1)
                colIdx1[col] = oldW - 1;

            colFct1[col] = srcColFl - idx;
            colFct0[col] = 1.0f - colFct1[col];
        }

        // buffer in new image
        unsigned char *newpix = newImg->getBitmap(frameNo);
        unsigned char *pixarr = image_->getBitmap(frameNo);

        /// Line-by-line calculations of result image
        float srcH = (float)oldH;
        float dstH = (float)newHeight;
        int row;
        for (row = 0; row < newHeight; row++)
        {

            // calculate per-line invariants
            // 'line index' in src img
            double srcRowFl = (row + 0.5) * srcH / dstH - 0.5;

            // line index for interpolation
            int rowIdx0 = (int)floor(srcRowFl);
            int rowIdx1 = rowIdx0 + 1;

            float rowFct1 = (float)srcRowFl - rowIdx0; // interpolation factors for both lines
            float rowFct0 = 1.0f - rowFct1;

            if (rowIdx0 < 0)
                rowIdx0 = 0;
            if (rowIdx1 >= oldH - 1)
                rowIdx1 = oldH - 1;

            unsigned char *srcRow0 = pixarr + (numChan * oldW * rowIdx0);
            unsigned char *srcRow1 = pixarr + (numChan * oldW * rowIdx1);
            unsigned char *dstLine = newpix + (numChan * newWidth * row);

            int col;
            for (col = 0; col < newWidth; col++)
            {
                // calculate per-pixel invariants
                unsigned char *pix00 = srcRow0 + numChan * colIdx0[col];
                unsigned char *pix01 = srcRow0 + numChan * colIdx0[col];
                unsigned char *pix10 = srcRow1 + numChan * colIdx1[col];
                unsigned char *pix11 = srcRow1 + numChan * colIdx1[col];

                // loop for all colors
                int i;
                for (i = 0; i < numChan; i++)
                {
                    *dstLine = (unsigned char)(*pix00 * rowFct0 * colFct0[col]
                                               + *pix01 * rowFct0 * colFct1[col]
                                               + *pix10 * rowFct1 * colFct0[col]
                                               + *pix11 * rowFct1 * colFct1[col]);
                    ++dstLine;
                    ++pix00;
                    ++pix01;
                    ++pix10;
                    ++pix11;
                }
            }
        }
    }

    // free intermediate storage
    delete[] colIdx0;
    delete[] colIdx1;
    delete[] colFct0;
    delete[] colFct1;

    // now substiture in our internal state
    delete image_;
    image_ = newImg;
}
