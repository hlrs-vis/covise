/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

// +++++++++++++++++++++++++++++++++++++++++
// TESTPROG coTestImg
//
// Read an image file via the coImgLib and
// print essential data to stdout
//
// Initial version: 2004-04-28 we
// +++++++++++++++++++++++++++++++++++++++++
// (C) 2004 by VirCinity IT Consulting
// +++++++++++++++++++++++++++++++++++++++++
// Changes:

#include "coImage.h"
#include <covise/covise.h>

int main(int argc, char *argv[])
{
    if (argc != 2)
    {
        cerr << "Call: " << argv[0] << " <filename>" << endl;
        exit(-2);
    }

    coImage *img = new coImage(argv[1]);
    if (!img)
    {
        cerr << "Could not find appropriate image type" << endl;
    }
    else if (img->isBad())
    {
        cerr << "Error reading " << argv[1] << ": "
             << img->errorMessage() << endl;
        return -1;
    }
    else
    {
        int numFrames = img->getNumFrames();
        cout << argv[1] << " : "
             << img->getWidth() << " x " << img->getHeight()
             << " " << img->getNumChannels() << " Channels, "
             << numFrames
             << ((numFrames) ? " Frame" : " Frames") << endl;

        img->scaleExp2();

        cout << argv[1] << " : "
             << img->getWidth() << " x " << img->getHeight()
             << " " << img->getNumChannels() << " Channels, "
             << numFrames
             << ((numFrames) ? " Frame" : " Frames") << endl;

        img->scale(2 * img->getWidth(), 2 * img->getHeight());

        cout << argv[1] << " : "
             << img->getWidth() << " x " << img->getHeight()
             << " " << img->getNumChannels() << " Channels, "
             << numFrames
             << ((numFrames) ? " Frame" : " Frames") << endl;

        return 0;
    }

    delete img;
}
