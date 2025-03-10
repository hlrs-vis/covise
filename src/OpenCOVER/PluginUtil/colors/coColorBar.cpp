/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// class coColorBar
//
// derived from coMenuItem
// colorbar is a window containing a texture and labels
//
// Initial version: 2002-01, dr
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// (C) 2002 by Vircinity IT Consulting
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

#include "coColorBar.h"
#include <util/unixcompat.h>
#include <OpenVRUI/coLabel.h>
#include <OpenVRUI/coRowContainer.h>
#include <OpenVRUI/coTexturedBackground.h>
#include <OpenVRUI/coColoredBackground.h>
#include <OpenVRUI/coMenu.h>
#include <OpenVRUI/coSubMenuItem.h>
#include <config/CoviseConfig.h>
#include <cmath>
#include <cassert>
#include <algorithm>

using namespace opencover;

static const float LabelHeight = 60.;
static const float Height = (MAX_LABELS - 1) * LabelHeight;

// this should be in libm, but is missing unter linux
inline int xx_round(double x)
{
    return int(x + 0.5);
}

// mask wird neuerdings irgnoriert und statt dessen %g genommen
static void calcFormat(float &minIO, float &maxIO, char * /*mask*/, int &iSteps)
{
    double min = minIO;
    double max = maxIO;
    bool swapped = false;
    if (min > max) 
    {
        swapped = true;
        std::swap(min, max);
    }

    double basis[3] = { 1.0, 2.0, 5.0 };
    double tens = 1.0;

    int i = 0;
    ///int digits=0;
    double step = tens * basis[i];
    double numSteps = (max - min) / step;

    while (numSteps > 13 || numSteps < -13)
    {
        ++i;
        if (i > 2)
        {
            i = 0;
            tens *= 10.0;
            //--digits;
        }
        step = tens * basis[i];
        numSteps = (max - min) / step;
    }

    while (numSteps < 5 && numSteps > -5)
    {
        --i;
        if (i < 0)
        {
            i = 2;
            tens /= 10.0;
            //++digits;
        }
        step = tens * basis[i];
        numSteps = (max - min) / step;
    }

    //adjust all
    // add little offsets against
    min = step * floor(min / (double)step + 0.000001);
    // roundoff errors
    max = step * ceil(max / (double)step - 0.000001);
    iSteps = xx_round((max - min) / step) + 1;

    /*
   if (digits>0)
   {
      //sprintf(mask,"%%.%df",digits);
      strcpy(mask,"%g");
   }
   else
   {
 
      /// negative digits - may require exponetials
      double maxAbs = fabs(max);
      double minAbs = fabs(min);
      if (minAbs>maxAbs)
         maxAbs=minAbs;
      double maxExp=1.0;
      while (maxExp<maxAbs)
      {
         maxExp*=10.0;
         digits++;
      }
      digits--;
      if (digits>5)
         sprintf(mask,"%%.%de",digits);
      else
         sprintf(mask,"%%.0f");
   }
   */
    if (swapped)
    {
        std::swap(min, max);
    }

    minIO = (float)min;
    maxIO = (float)max;
}

coColorBar::coColorBar(const std::string &name, const ColorMap &map, bool inMenu)
    : coMenuItem(name.c_str())
    , name_(name)
{

    image_.clear();

    auto bg = inMenu ? vrui::coUIElement::ITEM_BACKGROUND_NORMAL : vrui::coUIElement::BLACK;

    // create the
    makeLabelValues(map);
    makeTickImage();
    for (int i = 0; i < MAX_LABELS; i++)
    {
        labels_[i] = new vrui::coLabel();
        labels_[i]->setUniqueName("label" + std::to_string(i));
        hspaces_[i] = new vrui::coTexturedBackground((const uint *)tickImage_.data(), (const uint *)tickImage_.data(), (const uint *)tickImage_.data(), 4, 32, 32, 1);
        hspaces_[i]->setUniqueName("hspaces" + std::to_string(i));
        hspaces_[i]->setMinWidth(60);
        hspaces_[i]->setMinHeight(LabelHeight);
        labelAndHspaces_[i] = new vrui::coRowContainer(vrui::coRowContainer::HORIZONTAL);
        labelAndHspaces_[i]->setUniqueName("labelAndSpaces" + std::to_string(i));
        labelAndHspaces_[i]->setVgap(0.0);
        labelAndHspaces_[i]->setHgap(0.0);
        labelAndHspaces_[i]->addElement(hspaces_[i]);
        labelAndHspaces_[i]->addElement(labels_[i]);
        vspaces_[i] = new vrui::coColoredBackground(bg, vrui::coUIElement::ITEM_BACKGROUND_HIGHLIGHTED, vrui::coUIElement::ITEM_BACKGROUND_DISABLED);
        vspaces_[i]->setUniqueName("vspaces" + std::to_string(i));
        vspaces_[i]->setMinWidth(60);
        vspaces_[i]->setMinHeight(0);
    }
    allLabels_ = new vrui::coRowContainer(vrui::coRowContainer::VERTICAL);
    allLabels_->setUniqueName("allLabels");
    allLabels_->setVgap(0.0);
    allLabels_->setHgap(0.0);

    speciesLabel_ = new vrui::coLabel();
    speciesLabel_->setUniqueName("speciesLabel");

    std::string precision = covise::coCoviseConfig::getEntry("COVER.Plugin.ColorBars.Precision");
    if (!precision.empty())
    {
        sprintf(format_str_, "%%.%sf", precision.c_str());
    }
    else if ((fabs(map.max()) > 1e-4 || fabs(map.min()) > 1e-4) && fabs(map.min()) < 1e+6 && fabs(map.max()) < 1e+6)
    {
        int ndig = 5;
        int prec = std::max(0, ndig-(int)std::log10(std::max(fabs(map.min()), fabs(map.max()))));
        std::string sign = "";
        if (map.min() < 0 || map.max() < 0)
            sign = "+";
        sprintf(format_str_, "%%%s%d.%df", sign.c_str(), ndig-prec, prec);
    }
    else if (map.min() < 0)
    {
        sprintf(format_str_, "%%+g");
    }
    else
    {
        sprintf(format_str_, "%%g");
    }

    // create texture
    image_.resize(4 * 256 * 2); // 4 componenten, 256*2 gross
    makeImage(map, map.min() > map.max());
    texture_ = new vrui::coTexturedBackground((const uint *)image_.data(), (const uint *)image_.data(), (const uint *)image_.data(), 4, 2, 256, 1);
    texture_->setUniqueName("texture");
    texture_->setMinHeight(LabelHeight); // entspricht einer color
    texture_->setMinWidth(100);
    texture_->setHeight(Height);
    vspace_ = new vrui::coColoredBackground(bg, vrui::coUIElement::ITEM_BACKGROUND_HIGHLIGHTED, vrui::coUIElement::ITEM_BACKGROUND_DISABLED);
    vspace_->setUniqueName("vspace");
    vspace_->setMinWidth(2);
    vspace_->setMinHeight(30);
    textureAndVspace_ = new vrui::coRowContainer(vrui::coRowContainer::VERTICAL);
    textureAndVspace_->setUniqueName("textureAndVspace");
    textureAndVspace_->setVgap(0.0);
    textureAndVspace_->setHgap(0.0);

    textureAndVspace_->addElement(vspace_);
    textureAndVspace_->addElement(texture_);

    // create container for the two containers
    textureAndLabels_ = new vrui::coRowContainer(vrui::coRowContainer::HORIZONTAL);
    textureAndLabels_->setUniqueName("textureAndLabels");
    textureAndLabels_->setVgap(0.0);
    textureAndLabels_->setHgap(0.0);

    textureAndLabels_->addElement(textureAndVspace_);
    textureAndLabels_->addElement(allLabels_);

    // create background and add top container
    background_ = new vrui::coColoredBackground(bg, vrui::coUIElement::ITEM_BACKGROUND_HIGHLIGHTED, vrui::coUIElement::ITEM_BACKGROUND_DISABLED);
    background_->setUniqueName("background");

    if (inMenu)
    {
        background_->addElement(textureAndLabels_);
    }
    else
    {
        everything_ = new vrui::coRowContainer(vrui::coRowContainer::VERTICAL);
        everything_->setUniqueName("everything");
        everything_->setVgap(10.0);
        everything_->setHgap(10.0);
        everything_->addElement(textureAndLabels_);
        everything_->addElement(speciesLabel_);
        background_->addElement(everything_);
    }

    update(map);
}

/// return the actual UI Element that represents this menu.
vrui::coUIElement *coColorBar::getUIElement()
{
    return background_;
}

coColorBar::~coColorBar()
{
    int i;
    //fprintf(stderr,"delete coColorBar [%s] with\n", name_);

    for (i = 0; i < MAX_LABELS; i++)
    {
        delete labels_[i];
        delete hspaces_[i];
        delete labelAndHspaces_[i];
    }
    delete allLabels_;

    delete texture_;
    delete vspace_;
    delete textureAndVspace_;

    delete textureAndLabels_;
    delete speciesLabel_;
    delete background_;
}

void
coColorBar::update(const ColorMap &map)
{
    char str[100];
    // use - and + symbyls with same width
    //const char minus[] = "\u2212"; // minus
    //const char minus[] = "\uff0d"; // full-width hypen minus
    const char minus[] = "\u2013"; // en-dash
    //const char plus[] = "\uff0b"; // full-width plus
    const char plus[] = "+";
    //const char space[] = "\u2002"; // en-space
    const char space[] = " ";
    const size_t off = std::max(sizeof space, std::max(sizeof minus,sizeof plus)-2); // reuse one char, don't count terminating 0

    // remove old labels
    for (size_t i = 0; i < MAX_LABELS; i++)
    {
        allLabels_->removeElement(labelAndHspaces_[i]);
        allLabels_->removeElement(vspaces_[i]);
    }
    assert(allLabels_->getSize() == 0);

    // update labels
    makeLabelValues(map);
    float vgap = (Height - (numLabels_-1)*LabelHeight)/(numLabels_-1);

    for (size_t i = 0; i < numLabels_; i++)
    {
        snprintf(str+off, sizeof(str)-off, format_str_, labelValues_[i]);
        if (str[off] == '-')
        {
            for (size_t j=0; j<sizeof minus-1; ++j)
                str[off+2-sizeof(minus)+j] = minus[j];
            labels_[i]->setString(str+off+2-sizeof minus);
        }
        else if (str[off] == '+')
        {
            for (size_t j=0; j<sizeof plus-1; ++j)
                str[off+2-sizeof(plus)+j] = plus[j];
            labels_[i]->setString(str+off+2-sizeof plus);
        }
        else if (str[off] == ' ')
        {
            for (size_t j=0; j<sizeof space-1; ++j)
                str[off+2-sizeof(space)] = space[j];
            labels_[i]->setString(str+off+2-sizeof space);
        }
        else
        {
            labels_[i]->setString(str+off);
        }
        allLabels_->addElement(labelAndHspaces_[i]);
        if (i < numLabels_-1)
        {
            vspaces_[i]->setMinHeight(vgap);
            allLabels_->addElement(vspaces_[i]);
        }
    }

    // update image
    makeImage(map, map.min() > map.max());
    texture_->setImage((const uint *)image_.data(), (const uint *)image_.data(), (const uint *)image_.data(), 4, 2, 256, 1);
    texture_->setHeight(Height);
    auto species = map.species();
    if(!map.unit().empty())
        species += " (" + map.unit() + ")";
    speciesLabel_->setString(species);
}

const char *coColorBar::getName() const
{
    return name_.c_str();
}

void
coColorBar::makeImage(const ColorMap &m, bool swapped)
{
    unsigned char *cur;
    int x, y, idx;

    cur = image_.data();
    for (y = 0; y < 256; y++)
    {
        for (x = 0; x < 2; x++)
        {
            idx = (int)(((float)y / 256.0) * m.steps());
            if (swapped)
                idx = m.steps() - idx - 1;
            auto c = m.getColorPerStep(idx);
            for (size_t i = 0; i < 4; i++)
            {
                *cur = (unsigned char)(255.0 * c[i]);
                cur++;
            }
        }
    }
}

void
coColorBar::makeTickImage()
{
    unsigned char *cur;
    int x, y;

    tickImage_.resize(4*32*32);
    cur = tickImage_.data();

    for (y = 0; y < 32; y++) // white
    {
        for (x = 0; x < 32; x++)
        {
            if (y >= 15 && y <= 17 && x < 16)
            {
                *cur = (unsigned char)(255.0 * 1.0);
                cur++;
                *cur = (unsigned char)(255.0 * 1.0);
                cur++;
                *cur = (unsigned char)(255.0 * 1.0);
                cur++;
                *cur = (unsigned char)(255.0 * 1.0);
                cur++;
            }
            else
            {

                *cur = (unsigned char)(255.0 * 1.0);
                cur++;
                *cur = (unsigned char)(255.0 * 1.0);
                cur++;
                *cur = (unsigned char)(255.0 * 1.0);
                cur++;
                *cur = (unsigned char)(255.0 * 0.0);
                cur++;
            }
        }
    }
}

void
coColorBar::makeLabelValues(const ColorMap &map)
{
    // create labels
    numLabels_ = map.steps() + 1;
    if (map.steps() < 256)
    {
        // label every n-th step
        for (int i=1; i<(256/MAX_LABELS)+1; ++i)
        {
            if (map.steps() % i != 0)
                continue;
            if (map.steps()/i+1 > MAX_LABELS)
                continue;
            numLabels_ = map.steps()/i + 1;
            break;
        }
    }
    auto min = map.min();
    auto max = map.max();
    // adapt the min/max values to more readible values
    if (numLabels_ >= MAX_LABELS)
    {
        calcFormat(min, max, format_str_, numLabels_);
        //strcpy(format_str_,"%f");
        //numLabels_=MAX_LABELS;
    }
    else
    {
        float dummyMin = min; // if we have a stepped map, we only use the formats
        float dummyMax = max;
        int dummyNum = numLabels_;
        calcFormat(dummyMin, dummyMax, format_str_, dummyNum);
    }
    assert(numLabels_ <= MAX_LABELS);

    float step = (max - min) / (numLabels_ - 1);

    if (step < 0)
    {
        labelValues_[0] = min;
        for (int i = 1; i < numLabels_ - 1; i++) {
            labelValues_[i] = min + i * step;
        }
        for (int i = numLabels_ > 0 ? numLabels_ - 1 : 0; i < MAX_LABELS; i++) {
            labelValues_[i] = max;
        }
    }
    else
    {
        labelValues_[0] = max;
        for (int i = 1; i < numLabels_ - 1; i++) {
            labelValues_[i] = max - i * step;
        }
        for (int i = numLabels_ > 0 ? numLabels_ - 1 : 0; i < MAX_LABELS; i++) {
            labelValues_[i] = min;
        }
    }
}

const char *coColorBar::getClassName() const
{
    return "coColorBar";
}

bool coColorBar::isOfClassName(const char *classname) const
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
            return vrui::coMenuItem::isOfClassName(classname);
        }
    }

    // nobody is NULL
    return false;
}
