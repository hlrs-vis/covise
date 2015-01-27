/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/*ReadTextureImage.cpp
 *
 *  Created on: Jan 6, 2011
 *      Author: F-J S
 */

// class to read images for textures

#include <osgDB/ReadFile>
#include "HfT_osg_ReadTextureImage.h"
#include "HfT_string.h"

using namespace std;
using namespace osg;

//default constructor
HfT_osg_ReadTextureImage::HfT_osg_ReadTextureImage(std::string dirpath, unsigned int i)
{
    m_imagedir = dirpath;
    m_image = readImage(i);
}

HfT_osg_ReadTextureImage::~HfT_osg_ReadTextureImage()
{
}
osg::Image *HfT_osg_ReadTextureImage::readImage(unsigned int i)
{
    std::string istr = HfT_int_to_string(i);
    string imagepath = m_imagedir + "image" + istr + ".png";
    m_image = osgDB::readImageFile(imagepath);

    return m_image;
}
osg::Image *HfT_osg_ReadTextureImage::getImage()
{
    return m_image;
}
