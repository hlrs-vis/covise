/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// ++                                                  (C)2001 VirCinity  ++
// ++ Description: Craetes an annotation flag for the COVISE INVENTOR     ++
// ++ renderer                                                            ++
// ++                                                                     ++
// ++ Author:  Ralf Mikulla (rm@vircinity.com)                            ++
// ++                                                                     ++
// ++               VirCinity GmbH                                        ++
// ++               Nobelstrasse 15                                       ++
// ++               70569 Stuttgart                                       ++
// ++                                                                     ++
// ++ Date: 22.10.2001                                                    ++
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#ifndef INVANNOFLAG_H
#define INVANNOFLAG_H

#include "InvActiveNode.h"

// include Inventor NODES
#include <Inventor/nodes/SoSeparator.h>
#include <Inventor/nodes/SoSelection.h>
#include <Inventor/nodes/SoDrawStyle.h>
#include <Inventor/nodes/SoIndexedFaceSet.h>
#include <Inventor/nodes/SoRotation.h>
#include <Inventor/nodes/SoTranslation.h>
#include <Inventor/nodes/SoText2.h>

#include <Inventor/Xt/viewers/SoXtExaminerViewer.h>

#include <Inventor/SoPickedPoint.h>

#include "TextField.h"

typedef std::string SerializeString;

class InvAnnoFlag : public InvActiveNode
{
public:
    /// default constructor
    InvAnnoFlag();

    /// constructor
    //
    /// @param instance number of the flag to be constructed
    InvAnnoFlag(const int &num);

    /// just like java: create an object out of a string representation
    //  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    /// !!! The method isAlive() must be called after an instance of !!!
    //  !!! InvAnnoFlag was created using this constructor           !!!
    //  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    /// recommanded candidate for throwing of exceptions
    //
    /// @param string representation of an object
    InvAnnoFlag(const SerializeString &str);

    /// are we fully constructed ?
    /// method to avoid throwing of exceptions out of InvAnnoFlag(const SerializeString &str)
    //
    /// @return  true if obj was completely build up false else
    bool isAlive();

    /// just like java: make a string representation of this
    //
    /// @return string representation of this
    SerializeString serialize() const;

    /// check obj against a string representation
    //
    /// @param   str  string representation of an object
    //
    /// @return  true if obj is identical to the string representation str false else
    bool operator==(const SerializeString &str) const;

    /// default destructor
    virtual ~InvAnnoFlag();

    /// set point at which the flag is generated
    //
    /// @param  pick  point at which the flag should be set
    void setPickedPoint(const SoPickedPoint *pick, const SbVec3f &camPos);

    /// overloaded selection callback from InvActiveNode
    //
    /// @param me the current instance
    //
    /// @param p path selected by the pointer
    static void selectionCB(void *me, SoPath *p);

    /// overloaded deselection callback from InvActiveNode
    //
    /// @param me the current instance
    //
    /// @param p path selected by the pointer
    static void deSelectionCB(void *me, SoPath *p);

    /// get instance number
    //
    /// @return  instance number
    int getInstance();

    /// scale the reference point
    //
    /// @param scale factor
    void reScale(const float &s);

    /// set text methods
    void setText();
    void setText(const char *cStr);
    void setTextLine(const std::string &str);
    void setBackSpace(const int &n);
    void clearText();

private:
    std::string parseToken(const std::string &str, const char *tok) const;

    SoGroup *makeArrow();
    SoText2 *annoText_;
    SoTranslation *transl_;
    SoRotation *rota_;
    SbVec3f iNorm_;
    CoTextField textField_;
    bool alive_;
    int num_;
    std::string hostname_;
    float scale_;
    SoScale *reScale_;
};
#endif
