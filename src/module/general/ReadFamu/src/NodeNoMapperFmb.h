/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/** @file NodeNoMapperFmb.h
 * a manager for node numbers for Famu *.fmb files.
 */

// #include "NodeNoMapperFmb.h"  // a manager for node numbers for Famu *.fmb files.

#ifndef __NodeNoMapperFmb_h__
#define __NodeNoMapperFmb_h__

#include "NodeNoMapper.h" // a manager for node numbers (maps from internal to mesh file node numbers)

/**
 * a manager for node numbers for Famu *.fmb files.
 * singleton instance
 */
class NodeNoMapperFmb : public NodeNoMapper
{
public:
    static NodeNoMapperFmb *getInstance();
    static NodeNoMapperFmb *createInstance(ObjectInputStream *archive,
                                           OutputHandler *outputHandler);
    virtual void deleteInstance(void);
    static void delInstance(void); // same as deleteInstance() but static

protected:
    virtual void readFromArchive(ObjectInputStream *archive);

    static OutputHandler *_outputHandler;
    static NodeNoMapperFmb *_instance;
    NodeNoMapperFmb(ObjectInputStream *archive,
                    OutputHandler *outputHandler);
    virtual ~NodeNoMapperFmb();
};

#endif
