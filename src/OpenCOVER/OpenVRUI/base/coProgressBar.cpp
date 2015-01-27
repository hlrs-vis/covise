/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include <OpenVRUI/coProgressBar.h>
#include <OpenVRUI/coLabel.h>
#include <OpenVRUI/coFlatPanelGeometry.h>
#include <OpenVRUI/coTexturedBackground.h>

#include <util/unixcompat.h>

namespace vrui
{

coProgressBar::coProgressBar()
    : coPanel(new coFlatPanelGeometry(coUIElement::GREY))
    , progress(75.0f)
{

    createGeometry();

    this->doneBackground = new coTexturedBackground("UI/progressbar", "UI/progressbar", "UI/progressbar");
    this->label = new coLabel();
    this->dummyLabel = new coLabel(" ");

    this->doneBackground->setPos(0, 0, 0.5f);
    this->label->setPos(0, 0, 1);
    this->dummyLabel->setPos(400.0f, 60.0f, 0.0f);
    this->dummyLabel->setFontSize(80.0f);

    addElement(this->doneBackground);
    addElement(this->label);
    addElement(this->dummyLabel);
}

coProgressBar::~coProgressBar()
{
    delete doneBackground;
    delete label;
}

void coProgressBar::setStyle(coProgressBar::Style style)
{
    this->styleCurrent = style;
    this->styleSet = style;
}

coProgressBar::Style coProgressBar::getStyle() const
{
    return this->styleCurrent;
}

void coProgressBar::setProgress(float progress)
{

    if (this->styleSet == coProgressBar::Default)
        this->styleCurrent = coProgressBar::Float;

    this->progress = progress;
    char progressString[100];
    snprintf(progressString, 100, "%.2f%%", progress);
    setProgress(std::string(progressString), progress);
}

void coProgressBar::setProgress(int progress)
{
    if (this->styleSet == coProgressBar::Default)
        this->styleCurrent = coProgressBar::Integer;

    this->progress = (float)progress;
    char progressString[100];
    snprintf(progressString, 100, "%d%%", progress);
    setProgress(std::string(progressString), (float)progress);
}

void coProgressBar::setProgress(const std::string &progressString, float progress)
{
    this->label->setString(progressString);
    this->label->setFontSize(this->contentHeight * 3.0f / 5.0f);
    this->label->setPos((this->contentWidth - this->label->getWidth()) / 2.0f, (this->contentHeight - this->label->getHeight()) / 2.0f, 1.0f);
    this->doneBackground->setSize(this->contentWidth * progress / 100, this->contentHeight, 1.0f);
    this->dummyLabel->setPos(this->contentWidth - this->dummyLabel->getWidth(), 0, 2);
}

float coProgressBar::getProgress() const
{
    return this->progress;
}

const char *coProgressBar::getClassName() const
{
    return "coProgressBar";
}

void coProgressBar::resizeToParent(float x, float y, float z, bool shrink)
{
    //std::cerr << "coProgressBar::parentResized info: called " << x << " " << y << " " << z << " " << shrink << std::endl;

    //   y = fmin(y, x / 10.f);
    //   y = fmin(this->getMaxH(), y);
    //   x = fmax(this->getMaxW(), x);
    //
    //   this->px = x;
    //   this->py = y;

    coPanel::resizeToParent(x, y, z, shrink);
}

bool coProgressBar::isOfClassName(const char *classname) const
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
            return coUIElement::isOfClassName(classname);
        }
    }

    // nobody is NULL
    return false;
}
}
