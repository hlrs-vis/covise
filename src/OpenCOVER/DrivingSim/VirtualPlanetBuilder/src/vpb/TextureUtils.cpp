/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include <vpb/TextureUtils>
#include <vpb/BuildLog>
#include <iostream>

void vpb::compress(osg::State &state, osg::Texture &texture, osg::Texture::InternalFormatMode compressedFormat, bool generateMipMap, bool resizeToPowerOfTwo, vpb::BuildOptions::CompressionMethod method, vpb::BuildOptions::CompressionQuality quality)
{
    if (method != vpb::BuildOptions::GL_DRIVER)
    {
        osgDB::ImageProcessor *processor = osgDB::Registry::instance()->getImageProcessor();
        if (processor)
        {

            osgDB::ImageProcessor::CompressionMethod cm = (method == vpb::BuildOptions::NVTT) ? osgDB::ImageProcessor::USE_GPU : osgDB::ImageProcessor::USE_CPU;
            osgDB::ImageProcessor::CompressionQuality cq = osgDB::ImageProcessor::NORMAL;
            switch (quality)
            {
            case (vpb::BuildOptions::FASTEST):
                cq = osgDB::ImageProcessor::FASTEST;
            case (vpb::BuildOptions::NORMAL):
                cq = osgDB::ImageProcessor::NORMAL;
            case (vpb::BuildOptions::PRODUCTION):
                cq = osgDB::ImageProcessor::PRODUCTION;
            case (vpb::BuildOptions::HIGHEST):
                cq = osgDB::ImageProcessor::HIGHEST;
            }

            processor->compress(*texture.getImage(0), compressedFormat, generateMipMap, resizeToPowerOfTwo, cm, cq);

            texture.setInternalFormatMode(osg::Texture::USE_IMAGE_DATA_FORMAT);
            texture.setResizeNonPowerOfTwoHint(resizeToPowerOfTwo);

            return;
        }
        else
        {
            log(osg::WARN, "NVTT selected for texture processing but it is not available.");
        }
    }

    texture.setInternalFormatMode(compressedFormat);

    // force the mip mapping off temporay if we intend the graphics hardware to do the mipmapping.
    osg::Texture::FilterMode filterMin = texture.getFilter(osg::Texture::MIN_FILTER);
    if (!generateMipMap)
    {
        log(osg::INFO, "   switching off MIP_MAPPING for compile");
        texture.setFilter(osg::Texture::MIN_FILTER, osg::Texture::LINEAR);
    }

    // make sure the OSG doesn't rescale images if it doesn't need to.
    texture.setResizeNonPowerOfTwoHint(resizeToPowerOfTwo);

    // get OpenGL driver to create texture from image.
    texture.apply(state);

    texture.getImage(0)->readImageFromCurrentTexture(0, true);

    // restore the mip mapping mode.
    if (!generateMipMap)
    {
        texture.setFilter(osg::Texture::MIN_FILTER, filterMin);
    }
    texture.dirtyTextureObject();
    texture.setInternalFormatMode(osg::Texture::USE_IMAGE_DATA_FORMAT);
}

void vpb::generateMipMap(osg::State &state, osg::Texture &texture, bool resizeToPowerOfTwo, vpb::BuildOptions::CompressionMethod method)
{
    if (method != vpb::BuildOptions::GL_DRIVER)
    {
        osgDB::ImageProcessor *processor = osgDB::Registry::instance()->getImageProcessor();
        if (processor)
        {
            osgDB::ImageProcessor::CompressionMethod cm = (method == vpb::BuildOptions::NVTT) ? osgDB::ImageProcessor::USE_GPU : osgDB::ImageProcessor::USE_CPU;
            processor->generateMipMap(*texture.getImage(0), resizeToPowerOfTwo, cm);

            texture.setInternalFormatMode(osg::Texture::USE_IMAGE_DATA_FORMAT);
            texture.setResizeNonPowerOfTwoHint(resizeToPowerOfTwo);

            return;
        }
        else
        {
            log(osg::WARN, "NVTT selected for texture processing but it is not available.");
        }
    }

    // make sure the OSG doesn't rescale images if it doesn't need to.
    texture.setResizeNonPowerOfTwoHint(resizeToPowerOfTwo);

    // get OpenGL driver to create texture from image.
    texture.apply(state);

    texture.getImage(0)->readImageFromCurrentTexture(0, true);

    texture.setInternalFormatMode(osg::Texture::USE_IMAGE_DATA_FORMAT);

    texture.dirtyTextureObject();
}
