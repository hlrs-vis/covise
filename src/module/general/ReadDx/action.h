/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef __ACTION_DEFINED
#define __ACTION_DEFINED

#include "DxObject.h"
#include "MultiGridMember.h"

using namespace covise;

struct ltstr
{
    bool operator()(const char *s1, const char *s2) const
    {
        return strcmp(s1, s2) < 0;
    }
};

typedef std::map<const char *, DxObject *, ltstr> DxObjectMap;

class actionClass
{
private:
    int defaultByteOrder_;

public:
    DxObjectMap arrays_;
    DxObjectMap fields_;
    MemberList multiGridMembers_;
    DxObject *currObject_;
    DxObject *getCurrObject()
    {
        return currObject_;
    }
    DxObjectMap &getArrays()
    {
        return arrays_;
    }
    DxObjectMap &getFields()
    {
        return fields_;
    }
    MemberList &getMembers()
    {
        return multiGridMembers_;
    }
    void addMember(const char *name, const char *fieldName);
    void addMember(int number, const char *fieldName);
    void setCurrFileName(const char *filename);
    void setCurrFileName(const char *dirname, const char *filename);
    void setCurrElementType(const char *elementType);
    void setCurrRef(const char *ref);
    void setCurrName(const char *name);
    void setCurrAttributeDep(const char *attributeRef);
    void setCurrAttributeName(const char *attributeName);
    void setCurrData(const char *data);
    void setCurrConnections(const char *connections);
    void setCurrPositions(const char *positions);
    void setCurrData(int number);
    void setCurrConnections(int number);
    void setCurrPositions(int number);
    void setCurrName(int number);
    void setCurrFollows(bool follows)
    {
        currObject_->setFollows(follows);
    }
    void setCurrRank(int rank)
    {
        currObject_->setRank(rank);
    }
    void setCurrItems(int items)
    {
        currObject_->setItems(items);
    }
    void setCurrType(int type)
    {
        currObject_->setType(type);
    }
    void setCurrShape(int shape)
    {
        currObject_->setShape(shape);
    }
    void setCurrDataFormat(int format)
    {
        currObject_->setDataFormat(format);
    }
    void setCurrObjectClass(int type)
    {
        currObject_->setObjectClass(type);
    }
    void setCurrDataOffset(int offset)
    {
        currObject_->setDataOffset(offset);
    }
    void setCurrByteOrder(int order)
    {
        currObject_->setByteOrder(order);
    }

    const char *getCurrRef()
    {
        return currObject_->getRef();
    }
    const char *getCurrAttributeName()
    {
        return currObject_->getAttributeName();
    }
    const char *getCurrAttributeDep()
    {
        return currObject_->getAttributeDep();
    }
    const char *getCurrName()
    {
        return currObject_->getName();
    }
    const char *getCurrData()
    {
        return currObject_->getData();
    }
    const char *getCurrConnections()
    {
        return currObject_->getConnections();
    }
    const char *getCurrPositions()
    {
        return currObject_->getPositions();
    }
    const char *getCurrElementType()
    {
        return currObject_->getElementType();
    }
    int getCurrRank()
    {
        return currObject_->getRank();
    }
    int getCurrItems()
    {
        return currObject_->getItems();
    }
    int getCurrType()
    {
        return currObject_->getType();
    }
    int getCurrShape()
    {
        return currObject_->getShape();
    }
    int getCurrByteOrder()
    {
        return currObject_->getByteOrder();
    }
    int getCurrDataOffset()
    {
        return currObject_->getDataOffset();
    }
    int getCurrObjectClass()
    {
        return currObject_->getObjectClass();
    }
    int getCurrDataFormat()
    {
        return currObject_->getDataFormat();
    }
    void show();
    void newCurrent()
    {
        currObject_ = new DxObject(defaultByteOrder_);
    }
    actionClass(int defaultByteOrder);
};
#endif
