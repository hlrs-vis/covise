/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef COPROGRESSBAR_H
#define COPROGRESSBAR_H

#include <OpenVRUI/coPanel.h>

namespace vrui
{

class coLabel;
class coTexturedBackground;

class OPENVRUIEXPORT coProgressBar : public coPanel
{
public:
    coProgressBar();
    virtual ~coProgressBar();

    enum Style
    {
        Default = 0xFF,
        Empty = 0x00,
        Integer = 0x01,
        Float = 0x02
    };

    void setProgress(float progress);
    void setProgress(int progress);

    float getProgress() const;

    void setStyle(Style style);
    Style getStyle() const;

    virtual void resizeToParent(float x, float y, float z, bool shrink);

    /// get the Element's classname
    virtual const char *getClassName() const;
    /// check if the Element or any ancestor is this classname
    virtual bool isOfClassName(const char *) const;

private:
    float progress;

    void setProgress(const std::string &progressString, float progress);

    Style styleCurrent;
    Style styleSet;

    coLabel *label;
    coLabel *dummyLabel;

    float px, py;

    coTexturedBackground *doneBackground;
};
}

#endif // COPROGRESSBAR_H
