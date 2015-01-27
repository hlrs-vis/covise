/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// ++                                                  (C)2001 VirCinity  ++
// ++ Description: Class to format arbitrary texts to block formatted     ++
// ++              texts                                                  ++
// ++                                                                     ++
// ++ Author:  Ralf Mikulla (rm@vircinity.com)                            ++
// ++                                                                     ++
// ++               VirCinity GmbH                                        ++
// ++               Nobelstrasse 15                                       ++
// ++               70569 Stuttgart                                       ++
// ++                                                                     ++
// ++ Date: 26.10.2001                                                    ++
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#ifndef TEXTFIELD_H
#define TEXTFIELD_H

#include <string>

class CoTextField
{
public:
    // unit to set width
    enum
    {
        LETTERS,
        POINTS
    };

    CoTextField();
    CoTextField(const int &width, const int &unit);

    // set units for text width
    void setUnit(const int &unit);
    // set preferred text width
    void setWidth(const int &width);

    // clear the text field
    void clear();

    // add text to the TextField
    void append(const std::string &text);

    // remove the last n chars from the text field
    void backSpace(const int &n);

    // get # of lines of the formatted text
    int getNumLines()
    {
        return numLines_;
    };
    // get formatted text by line
    std::string getLine(const int &index) const;

    // get unformatted text
    std::string getText() const;

    // get unformatted text and substitute " " (blanks) by a tag
    std::string getTaggedText(const std::string &tag = std::string("<b>")) const;

    // remove all tags out of text_ and reformat it
    void untag(const std::string &tag = std::string("<b>"));

    virtual ~CoTextField();

private:
    int unit_;
    int width_;
    std::string text_;
    int numLines_;
    std::string *lines_;

    // should be implemented later and set public
    CoTextField(const CoTextField &tf);
    void format();
    void substitueUmlaut();
};
#endif
