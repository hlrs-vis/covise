/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef COPROGRESSBARMENUITEM_H
#define COPROGRESSBARMENUITEM_H

#include <OpenVRUI/coRowMenuItem.h>

#include <string>

namespace vrui
{

class coProgressBar;

class OPENVRUIEXPORT coProgressBarMenuItem : public coRowMenuItem
{
public:
    coProgressBarMenuItem(const std::string &name);
    virtual ~coProgressBarMenuItem();

    void setProgress(float progress);
    void setProgress(int progress);

    float getProgress() const;

    /// get the Element's classname
    virtual const char *getClassName() const;
    /// check if the Element or any ancestor is this classname
    virtual bool isOfClassName(const char *) const;

private:
    coProgressBar *progressBar;
};
}
#endif
