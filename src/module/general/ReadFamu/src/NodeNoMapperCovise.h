/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/** @file NodeNoMapperCovise.h
 * a manager for node numbers for Covise *.cvr files.
 */

// #include "NodeNoMapperCovise.h"  // a manager for node numbers for Covise *.cvr files.

#ifndef __NodeNoMapperCovise_h__
#define __NodeNoMapperCovise_h__

#include "NodeNoMapper.h" // a manager for node numbers (maps from internal to mesh file node numbers)

/**
 * a manager for node numbers for Covise *.cvr files.
 */
class NodeNoMapperCovise : public NodeNoMapper
{
public:
    NodeNoMapperCovise(ObjectInputStream *archive,
                       OutputHandler *outputHandler);
    NodeNoMapperCovise(const NodeNoMapper *);

    virtual void deleteInstance(void)
    {
        delete this;
    };

protected:
    virtual ~NodeNoMapperCovise(){};
    virtual void readFromArchive(ObjectInputStream *archive);
};

#endif
