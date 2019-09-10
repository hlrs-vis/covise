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
    minIO = (float)min;
    maxIO = (float)max;
}

coColorBar::coColorBar(const std::string &name, const std::string &species, float mi, float ma, int nc, const float *r, const float *g, const float *b, const float *a, bool inMenu)
    : coMenuItem(name.c_str())
    , name_(name)
    , species_(species)
{
    int i;

    numColors_ = nc;
    min_ = mi;
    max_ = ma;

    //fprintf(stderr,"new coColorBar [%s] with %d colors\n", name_.c_str(), numColors_);
    image_.clear();

    auto bg = inMenu ? vrui::coUIElement::ITEM_BACKGROUND_NORMAL : vrui::coUIElement::BLACK;

    // create the
    makeLabelValues();
    makeTickImage();
    for (i = 0; i < MAX_LABELS; i++)
    {
        labels_[i] = new vrui::coLabel();
        hspaces_[i] = new vrui::coTexturedBackground((const uint *)tickImage_.data(), (const uint *)tickImage_.data(), (const uint *)tickImage_.data(), 4, 32, 32, 1);
        hspaces_[i]->setMinWidth(60);
        hspaces_[i]->setMinHeight(LabelHeight);
        labelAndHspaces_[i] = new vrui::coRowContainer(vrui::coRowContainer::HORIZONTAL);
        labelAndHspaces_[i]->setVgap(0.0);
        labelAndHspaces_[i]->setHgap(0.0);
        labelAndHspaces_[i]->addElement(hspaces_[i]);
        labelAndHspaces_[i]->addElement(labels_[i]);
        vspaces_[i] = new vrui::coColoredBackground(bg, vrui::coUIElement::ITEM_BACKGROUND_HIGHLIGHTED, vrui::coUIElement::ITEM_BACKGROUND_DISABLED);
        vspaces_[i]->setMinWidth(60);
        vspaces_[i]->setMinHeight(0);
    }
    allLabels_ = new vrui::coRowContainer(vrui::coRowContainer::VERTICAL);
    allLabels_->setVgap(0.0);
    allLabels_->setHgap(0.0);

    speciesLabel_ = new vrui::coLabel();

    std::string precision = covise::coCoviseConfig::getEntry("ColorsPlugin.Precision");
    if (!precision.empty())
    {
        sprintf(format_str_, "%%.%sf", precision.c_str());
    }
    else if ((fabs(ma) > 1e-4 || fabs(mi) > 1e-4) && fabs(mi) < 1e+6 && fabs(ma) < 1e+6)
    {
        int ndig = 5;
        int prec = std::max(0, ndig-(int)std::log10(std::max(fabs(mi), fabs(ma))));
        std::string sign = "";
        if (mi < 0 || ma < 0)
            sign = "+";
        sprintf(format_str_, "%%%s%d.%df", sign.c_str(), ndig-prec, prec);
    }
    else
    {
        sprintf(format_str_, "%%g");
    }

    // create texture
    image_.resize(4 * 256 * 2); // 4 componenten, 256*2 gross
    makeImage(numColors_, r, g, b, a);
    texture_ = new vrui::coTexturedBackground((const uint *)image_.data(), (const uint *)image_.data(), (const uint *)image_.data(), 4, 2, 256, 1);
    texture_->setMinHeight(LabelHeight); // entspricht einer color
    texture_->setMinWidth(100);
    texture_->setHeight(Height);
    vspace_ = new vrui::coColoredBackground(bg, vrui::coUIElement::ITEM_BACKGROUND_HIGHLIGHTED, vrui::coUIElement::ITEM_BACKGROUND_DISABLED);
    vspace_->setMinWidth(2);
    vspace_->setMinHeight(30);
    textureAndVspace_ = new vrui::coRowContainer(vrui::coRowContainer::VERTICAL);
    textureAndVspace_->setVgap(0.0);
    textureAndVspace_->setHgap(0.0);

    textureAndVspace_->addElement(vspace_);
    textureAndVspace_->addElement(texture_);

    // create container for the two containers
    textureAndLabels_ = new vrui::coRowContainer(vrui::coRowContainer::HORIZONTAL);
    textureAndLabels_->setVgap(0.0);
    textureAndLabels_->setHgap(0.0);

    textureAndLabels_->addElement(textureAndVspace_);
    textureAndLabels_->addElement(allLabels_);

    // create background and add top container
    background_ = new vrui::coColoredBackground(bg, vrui::coUIElement::ITEM_BACKGROUND_HIGHLIGHTED, vrui::coUIElement::ITEM_BACKGROUND_DISABLED);

    if (inMenu)
    {
        background_->addElement(textureAndLabels_);
    }
    else
    {
        everything_ = new vrui::coRowContainer(vrui::coRowContainer::VERTICAL);
        everything_->setVgap(10.0);
        everything_->setHgap(10.0);

        speciesLabel_->setString(species_);
        everything_->addElement(textureAndLabels_);
        everything_->addElement(speciesLabel_);
        background_->addElement(everything_);
    }

    update(min_, max_, numColors_, r, g, b, a);
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
coColorBar::update(float mi, float ma, int nc, const float *r, const float *g, const float *b, const float *a)
{
    int i;
    char str[100];

    numColors_ = nc;
    min_ = mi;
    max_ = ma;

    // remove old labels
    for (i = 0; i < MAX_LABELS; i++)
    {
        allLabels_->removeElement(labelAndHspaces_[i]);
        allLabels_->removeElement(vspaces_[i]);
    }

    // update labels
    makeLabelValues();
    float vgap = (Height - (numLabels_-1)*LabelHeight)/(numLabels_-1);

    for (i = 0; i < numLabels_; i++)
    {
        snprintf(str, sizeof(str), format_str_, labelValues_[i]);
        labels_[i]->setString(str);
        allLabels_->addElement(labelAndHspaces_[i]);
        if (i < numLabels_-1)
        {
            vspaces_[i]->setMinHeight(vgap);
            allLabels_->addElement(vspaces_[i]);
        }
    }

    // update image
    makeImage(numColors_, r, g, b, a);
    texture_->setImage((const uint *)image_.data(), (const uint *)image_.data(), (const uint *)image_.data(), 4, 2, 256, 1);
    texture_->setHeight(Height);
}

const char *coColorBar::getName() const
{
    return name_.c_str();
}

void
coColorBar::makeImage(int numColors, const float *r, const float *g, const float *b, const float *a)
{
    unsigned char *cur;
    int x, y, idx;

    cur = image_.data();
    for (y = 0; y < 256; y++)
    {
        for (x = 0; x < 2; x++)
        {
            idx = (int)(((float)y / 256.0) * numColors);
            *cur = (unsigned char)(255.0 * r[idx]);
            cur++;

            *cur = (unsigned char)(255.0 * g[idx]);
            cur++;

            *cur = (unsigned char)(255.0 * b[idx]);
            cur++;

            *cur = (unsigned char)(255.0 * a[idx]);
            cur++;
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
coColorBar::makeLabelValues()
{
    float step;
    int i;

    // create labels
    numLabels_ = numColors_ + 1;
    if (numColors_ < 256)
    {
        for (int i=1; i<(256/MAX_LABELS)+1; ++i)
        {
            if (numColors_ % i != 0)
                continue;
            if (numColors_/i+1 > MAX_LABELS)
                continue;
            numLabels_ = numColors_/i + 1;
            break;
        }
    }

    // adapt the min/max values to more readible values
    if (numLabels_ >= MAX_LABELS)
    {
        calcFormat(min_, max_, format_str_, numLabels_);
        //strcpy(format_str_,"%f");
        //numLabels_=MAX_LABELS;
    }
    else
    {
        float dummyMin = min_; // if we have a stepped map, we only use the formats
        float dummyMax = max_;
        int dummyNum = numLabels_;
        calcFormat(dummyMin, dummyMax, format_str_, dummyNum);
    }
    assert(numLabels_ <= MAX_LABELS);

    step = (max_ - min_) / (numLabels_ - 1);

    labelValues_[0] = max_;
    if (numLabels_ > 0)
        labelValues_[numLabels_ - 1] = min_;
    for (i = 1; i < numLabels_ - 1; i++)
    {
        labelValues_[i] = max_ - i * step;
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
