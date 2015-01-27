/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _TEXTURE_POPUP_H
#define _TEXTURE_POPUP_H

#include <osg/Camera>
#include <osg/StateSet>
#include <osg/Texture2D>

class TexturePopup
{
public:
    TexturePopup(double x, double y, double width, double height);
    virtual ~TexturePopup();

    void setImageFile(std::string fileName);

    void show();
    void hide();

private:
    std::map<const std::string, osg::ref_ptr<osg::Texture2D> > _popupTextures;

    bool _isShowing;

    osg::ref_ptr<osg::Camera> _camera;
    osg::StateSet *_state;
};

#endif
