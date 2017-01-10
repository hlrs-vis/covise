/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// ++                                                  (C)2001 VirCinity  ++
// ++ Description:  Implementation of class TextField                     ++
// ++                                                                     ++
// ++                                                                     ++
// ++ Author:  Ralf Mikulla (rm@vircinity.com)                            ++
// ++                                                                     ++
// ++               VirCinity GmbH                                        ++
// ++               Nobelstrasse 15                                       ++
// ++               70569 Stuttgart                                       ++
// ++                                                                     ++
// ++ Date: 26.10.2001                                                    ++
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#include <cstdlib>
#include <math.h>
#include <cstring>
#ifdef CO_LINUX
#include <mm.h>
#endif

#include <ctype.h>

#include "TextField.h"
#include <iostream>

int
min(int x, int y)
{
    return (x < y) ? x : y;
}

//
// Constructor
//
CoTextField::CoTextField()
    : unit_(CoTextField::LETTERS)
    , width_(20)
    , numLines_(0)
{
}

CoTextField::CoTextField(const int &width, const int &unit)
    : unit_(unit)
    , width_(width)
    , numLines_(0)
{
}

//
// Destructor
//
CoTextField::~CoTextField()
{
}

void
CoTextField::append(const std::string &text)
{
    // append the text to the internal text field
    text_ = text_ + text;

    substitueUmlaut();
    // format it
    format();
}

void
CoTextField::backSpace(const int &n)
{
    int len = text_.size();
    char *chText = new char[len + 1];
    len -= n;
    if (len > 0)
    {
        strncpy(chText, text_.c_str(), len);
        chText[len] = '\0';
    }
    text_ = std::string(chText);
    delete[] chText;

    //    cerr << "CoTextField::backSpace(..) text is now: " << text_ << endl;

    // format it
    format();
}

//
// here the text is formatted
//
void
CoTextField::format()
{
    if (unit_ == CoTextField::LETTERS)
    {
        std::string curText(text_); // we have to copy the mem area of text_

        // array of space positions
        // static for now
        const int iAllocInc = 10;
        int estSpaceArrSize = 100;
        int *spaceArr = new int[estSpaceArrSize];

        int idx = curText.find_first_of(" ");
        int i = 0;
        while (idx != std::string::npos)
        {
            spaceArr[i] = idx;
            idx = curText.find_first_of(" ", idx + 1);
            i++;
            if (i >= estSpaceArrSize)
            {
                // realloc memory for  space array if necessary
                int *iTmp = new int[estSpaceArrSize + iAllocInc];
                for (int ii = 0; ii < estSpaceArrSize; ii++)
                    iTmp[ii] = spaceArr[ii];
                delete[] spaceArr;
                spaceArr = iTmp;
                estSpaceArrSize += iAllocInc;
            }
        }
        spaceArr[i] = curText.size();
        i++;
        int spaceArrLen = i;

        int estNrLines = 1 + curText.size() / width_;
        lines_ = new std::string[estNrLines];

        const int allocInc = 5;

        int workIdx = 0;
        int currTextWidth = width_;
        int lineCnt = 0;

        while (currTextWidth < curText.size() + width_)
        {
            int minDist = 100000000;
            int start = workIdx;
            for (int ii = 0; ii < spaceArrLen; ++ii)
            {
                int dist = std::abs(spaceArr[ii] - currTextWidth);
                if (dist <= minDist)
                {
                    minDist = dist;
                    workIdx = 1 + spaceArr[ii];
                }
            }

            currTextWidth = workIdx + width_;

            lines_[lineCnt] = curText.substr(start, workIdx - start);
            lineCnt++;
            if (lineCnt >= estNrLines)
            {
                // realloc memory
                std::string *tmp = new std::string[estNrLines + allocInc];
                for (int i = 0; i < estNrLines; ++i)
                    tmp[i] = lines_[i];
                delete[] lines_;
                lines_ = tmp;
                estNrLines += allocInc;
            }
        }
        numLines_ = lineCnt;
    }
    else
    {
        std::cerr << "CoTextField::format() Formatting of text can only be done in units of letteres" << std::endl;
        std::cerr << "CoTextField::format() Set units to LETTERS and use default width (20 letters)" << std::endl;
        unit_ = CoTextField::LETTERS;
        width_ = 20;
        format();
    }
}

//
// return a line of formatted text
//
std::string
CoTextField::getLine(const int &index) const
{
    if (index > numLines_)
    {
        std::string empty;
        return empty;
    }

    return lines_[index];
}

void
CoTextField::clear()
{
    if (numLines_)
        delete[] lines_;
    numLines_ = 0;
    // set the internal text representaion to ""
    text_ = std::string("");
}

std::string
CoTextField::getText() const
{
    return text_;
}

std::string
CoTextField::getTaggedText(const std::string &tag) const
{
    std::string ret;

    int idx(text_.find_first_of(" "));
    int i(-1);

    while (idx != std::string::npos)
    {
        ret = ret + text_.substr(i + 1, idx - i - 1) + tag;
        i = idx;
        idx = text_.find_first_of(" ", idx + 1);
    }

    int len(text_.size());
    ret = ret + text_.substr(i + 1, len - i);

    return ret;
}

void
CoTextField::untag(const std::string &tag)
{
    std::string tmp;
    std::string spc(" ");
    int offset = tag.size();

    int idx = text_.find_first_of(tag.c_str());
    int i = -offset;

    while (idx != std::string::npos)
    {
        tmp = tmp + text_.substr(i + offset, idx - i - offset) + spc;
        i = idx;
        idx = text_.find_first_of(tag.c_str(), idx + 1);
    }

    int len = text_.size();
    tmp = tmp + text_.substr(i + offset, len - i);

    text_ = tmp;

    format();
}

//
// replace Umlaut chars with "clean" (7b) SCII chars
// ä->ae; ü->ue; ö->oe; ß->ss
//
void
CoTextField::substitueUmlaut()
{
    std::string tmp;
    int i(0);
    std::string tag("\xe4");
    int idx(text_.find_first_of(tag.c_str()));

    // ä->ae
    std::string spc("ae");
    while (idx != std::string::npos)
    {
        tmp = tmp + text_.substr(i, idx - i) + spc;
        i = idx + 1;
        idx = text_.find_first_of(tag.c_str(), idx + 1);
    }
    tmp = tmp + text_.substr(i, text_.size() - i);
    text_ = tmp;

    i = 0;
    tmp = std::string();
    tag = std::string("\xf6");
    idx = text_.find_first_of(tag.c_str());
    // ö->oe
    spc = std::string("oe");
    while (idx != std::string::npos)
    {
        tmp = tmp + text_.substr(i, idx - i) + spc;
        i = idx + 1;
        idx = text_.find_first_of(tag.c_str(), idx + 1);
    }
    tmp = tmp + text_.substr(i, text_.size() - i);
    text_ = tmp;

    i = 0;
    tmp = std::string();
    tag = std::string("\xfc");
    idx = text_.find_first_of(tag.c_str());
    // ä->ae
    spc = std::string("ue");
    while (idx != std::string::npos)
    {
        tmp = tmp + text_.substr(i, idx - i) + spc;
        i = idx + 1;
        idx = text_.find_first_of(tag.c_str(), idx + 1);
    }
    tmp = tmp + text_.substr(i, text_.size() - i);
    text_ = tmp;

    i = 0;
    tmp = std::string();
    tag = std::string("\xdf");
    idx = text_.find_first_of(tag.c_str());
    // ä->ae
    spc = std::string("ss");
    while (idx != std::string::npos)
    {
        tmp = tmp + text_.substr(i, idx - i) + spc;
        i = idx + 1;
        idx = text_.find_first_of(tag.c_str(), idx + 1);
    }
    tmp = tmp + text_.substr(i, text_.size() - i);
    text_ = tmp;
}
