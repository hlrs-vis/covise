/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef HFT_OSG_READTEXTUREIMAGE_H_
#define HFT_OSG_READTEXTUREIMAGE_H_

// Klasse zum Einlesen der von Images für Texturen
#include <string>
#include <osg/Image>

class HfT_osg_ReadTextureImage
{
public:
    //Konstruktor
    HfT_osg_ReadTextureImage(std::string dirpath, unsigned int i);

    virtual ~HfT_osg_ReadTextureImage();
    osg::Image *readImage(unsigned int i);
    osg::Image *getImage();
    osg::Image *m_image;
    std::string m_imagedir;
};
#endif /* HFT_OSG_READTEXTURIMAGE_H_ */
