/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/**********************************************************************
 *<
    FILE: inlist.h
 
    DESCRIPTION:  Defines a List of INodes
 
    CREATED BY: Scott Morrison
 
    HISTORY: created January 1996
 
 *> Copyright (c) 1996, All Rights Reserved.
 **********************************************************************/

#ifndef _INLIST_H_
#define _INLIST_H_

// List of inodes.

class INodeList
{
public:
    INodeList(INode *nd)
    {
        node = nd;
        next = NULL;
    }
    ~INodeList()
    {
        delete next;
    }
    INodeList *AddNode(INode *nd)
    {
        INodeList *n = new INodeList(nd);
        n->next = this;
        return n;
    }
    BOOL NodeInList(INode *nd)
    {
        for (INodeList *l = this; l; l = l->next)
            if (l->node == nd)
                return TRUE;
        return FALSE;
    }
    INode *GetNode()
    {
        return node;
    }
    INodeList *GetNext()
    {
        return next;
    }

private:
    INode *node;
    INodeList *next;
};

#endif
