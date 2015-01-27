/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "coProgressBarMenuItem.h"

#include <OpenVRUI/coProgressBarMenuItem.h>

#include <OpenVRUI/coProgressBar.h>
#include <OpenVRUI/coMenuContainer.h>
#include <OpenVRUI/coColoredBackground.h>

using std::string;

namespace vrui
{

/** Constructor.
  @param name label string
*/
coProgressBarMenuItem::coProgressBarMenuItem(const std::string &name)
    : coRowMenuItem(name)
{
    this->progressBar = new coProgressBar();
    container->addElement(progressBar);
}

/// Destructor
coProgressBarMenuItem::~coProgressBarMenuItem()
{
}

void coProgressBarMenuItem::setProgress(float progress)
{
    this->progressBar->setProgress(progress);
}

void coProgressBarMenuItem::setProgress(int progress)
{
    this->progressBar->setProgress(progress);
}

float coProgressBarMenuItem::getProgress() const
{
    return this->progressBar->getProgress();
}

const char *coProgressBarMenuItem::getClassName() const
{
    return "coProgressBarMenuItem";
}

bool coProgressBarMenuItem::isOfClassName(const char *classname) const
{
    // paranoia makes us mistrust the string library and check for NULL.
    if (classname && getClassName())
    {
        // check for identity
        if (!strcmp(classname, getClassName()))
        { // we are the one
            return true;
        }
        else
        { // we are not the wanted one. Branch up to parent class
            return coRowMenuItem::isOfClassName(classname);
        }
    }

    // nobody is NULL
    return false;
}
}
