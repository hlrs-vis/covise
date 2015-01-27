/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef CO_DIALOG_H
#define CO_DIALOG_H

#include <OpenVRUI/coUpdateManager.h>
#include <string>

namespace vrui
{

class coRowContainer;
class coMenuItem;
class coButton;
class coTexturedBackground;
class coPopupHandle;
class coFrame;
class coUIElement;
class vruiTransformNode;
class vruiMatrix;

class OPENVRUIEXPORT coDialog : public coUpdateable
{
protected:
    coFrame *itemsFrame;
    coPopupHandle *handle;

public:
    coDialog(const std::string &name);
    virtual ~coDialog();

    void addElement(coUIElement *);
    void showElement(coUIElement *);
    void removeElement(coUIElement *);
    virtual void childResized(bool shrink = true);
    virtual void resizeToParent(float, float, float, bool shrink = true);
    void setEnabled(bool on);
    void setHighlighted(bool on);
    void setVisible(bool on);
    vruiTransformNode *getDCS();
    void setTransformMatrix(vruiMatrix *matrix);
    void setTransformMatrix(vruiMatrix *matrix, float scale);
    void setScale(float scale);
    float getScale() const;
    float getWidth() const;
    float getHeight() const;
    float getDepth() const;
    float getXpos() const;
    float getYpos() const;
    float getZpos() const;
    void setPos(float x, float y, float z);
    void setSize(float width, float height, float depth);
    void setSize(float size);
    bool update();
    void show();
    void hide();
};
}
#endif
