/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _CO_OBJID_H_
#define _CO_OBJID_H_

// 25.09.98

#include <iostream>
#include "coTypes.h"
#ifdef WIN32
#include "unixcompat.h"
#endif
#include <stdio.h>
#include <cstring>
#include <string>

#ifndef INLINE
#define INLINE inline
#endif

namespace covise
{

class coDistributedObject;

/// Object-ID structure
struct coObjID
{
    coObjID()
        : portID(0)
        , modID(0)
        , seqNo(0)
        , id(NULL)
    {
    }
    coObjID(const char *name)
        : portID(0)
        , modID(0)
        , seqNo(0)
        , id(NULL)
    {
        if (name)
        {
            id = new char[strlen(name) + 1];
            strcpy(id, name);
        }
    }
    coObjID(const std::string &name)
        : portID(0)
        , modID(0)
        , seqNo(0)
        , id(NULL)
    {
        if (name.length() > 0)
        {
            id = new char[name.length() + 1];
            strcpy(id, name.c_str());
        }
    }
    coObjID(const coObjID &src)
        : portID(src.portID)
        , modID(src.modID)
        , seqNo(src.seqNo)
        , id(NULL)
    {
        if (src.id)
        {
            id = new char[strlen(src.id) + 1];
            strcpy(id, src.id);
        }
    }
    coObjID &operator=(const coObjID &src)
    {
        portID = src.portID;
        modID = src.modID;
        seqNo = src.seqNo;
        id = NULL;
        if (src.id)
        {
            id = new char[strlen(src.id) + 1];
            strcpy(id, src.id);
        }
        return *this;
    }
    // WARNING: if anything else than integers here: major changes in send/recv
    // for any changes: change shm/coDoHdr.h
    int portID;
    int modID;
    int seqNo;
    char *id;

    int getModID() const
    {
        return modID;
    }
    int getPortID() const
    {
        return portID;
        ;
    }
    int getHash() const
    {
        return (((portID & 0x00000007)) // 3 bit Mod-ID
                | ((modID & 0x00000007) << 3) // 3 bit Port-ID
                | ((seqNo & 0x00000007) << 6));
    } // 3 bit sequence

    int getSeqNo() const
    {
        return seqNo;
    }
    void setInvalid()
    {
        seqNo = portID = modID = -1;
    }
    int isValid() const
    {
        return (seqNo > 0) && (portID > 0) && (modID > 0);
    }
    const char *getString() const
    {
        static char name[500];
        snprintf(name, sizeof(name), "coObjID(m=%d,p=%d,seq%d)", modID, portID, seqNo);
        name[sizeof(name) - 1] = '\0';
        return name;
    }

    UTILEXPORT static int compare(const coObjID &a, const coObjID &b);
};

UTILEXPORT std::ostream &operator<<(std::ostream &, const coObjID &);

// enough for all hash entries...
enum
{
    CO_OBJECT_ID_HASH_SIZE = 512
};

UTILEXPORT int compare(const coObjID &a, const coObjID &b);

INLINE bool operator<(const coObjID &a, const coObjID &b)
{
    return (coObjID::compare(a, b) < 0);
}

INLINE bool operator>(const coObjID &a, const coObjID &b)
{
    return (coObjID::compare(a, b) > 0);
}

INLINE bool operator<=(const coObjID &a, const coObjID &b)
{
    return (coObjID::compare(a, b) <= 0);
}

INLINE bool operator>=(const coObjID &a, const coObjID &b)
{
    return (coObjID::compare(a, b) >= 0);
}

INLINE bool operator==(const coObjID &a, const coObjID &b)
{
    return (coObjID::compare(a, b) == 0);
}

INLINE bool operator!=(const coObjID &a, const coObjID &b)
{
    return (coObjID::compare(a, b) != 0);
}

/**
 * Class to create Object-IDs
 * 
 */
class coObjIDMaker
{

private:
    // Copy-Constructor: NOT  IMPLEMENTED
    coObjIDMaker(const coObjIDMaker &);

    // Assignment operator: NOT  IMPLEMENTED
    coObjIDMaker &operator=(const coObjIDMaker &);

    // Default constructor: NOT IMPLEMENTED
    coObjIDMaker();

    // My last ObjID
    coObjID d_objID;

    // sequence number increment
    int d_seqInc;

public:
    /// Constructor: set Module-ID,  Port-ID, sequence increment and start value
    coObjIDMaker(int32_t modID, int portID, int seqInc = 1, int firstSeqNo = 1)
    {
        d_objID.modID = modID;
        d_objID.portID = portID;
        d_objID.seqNo = firstSeqNo;
        d_seqInc = seqInc;
    }

    /// get a new Sequence Number
    coObjID getNewID()
    {
        d_objID.seqNo += d_seqInc;
        return d_objID;
    }

    // get a new Sequence Number
    coObjID getLastID()
    {
        return d_objID;
    }

    /// Destructor
    ~coObjIDMaker(){};
};

/**
 * Structure to describe all necessary data to request a new object
 */

struct UTILEXPORT coObjInfo // if changing this, change in co{Send,Recv}Buffer, too
{

    static int sequence;
    static char *baseName;
    coObjInfo()
    {
        blockNo = numBlocks = timeStep = numTimeSteps = -1;
        time = -1.0f;
        reqModID = 0;
    }
    coObjInfo(const char *name)
    {
        blockNo = numBlocks = timeStep = numTimeSteps = -1;
        time = -1.0f;
        reqModID = 0;
        if (baseName)
        {
            if (strlen(name) == 0)
            {
                id = baseName;
            }
            else
            {
                char *n;
                if (baseName)
                {
                    size_t l = strlen(baseName) + strlen(name) + 50;
                    n = new char[l];
                    snprintf(n, l, "%s_%d_%s", name, sequence, baseName);
                    n[l - 1] = '\0';
                }
                else
                {
                    size_t l = strlen(name) + 50;
                    n = new char[l];
                    snprintf(n, l, "%s_%d", name, sequence);
                    n[l - 1] = '\0';
                }
                sequence++;
                id = n;
                delete[] n;
            }
        }
        else
            id = name;
    }
    coObjInfo(const std::string &name)
    {
        blockNo = numBlocks = timeStep = numTimeSteps = -1;
        time = -1.0f;
        reqModID = 0;
        id = name;
    }
    static void setBaseName(const char *bn)
    {
        delete[] baseName;
        sequence = 0;
        if (bn)
        {
            baseName = new char[strlen(bn) + 1];
            strcpy(baseName, bn);
        }
        else
            baseName = NULL;
    };
    enum
    {
        UNKNOWN = -2,
        TIMESTEP_ALL = -1,
        NUM_TIMESTEPS_INFINIT = -1
    };

    coObjID id;
    int blockNo, numBlocks, timeStep, numTimeSteps;
    float time;
    int32_t reqModID; // needed for Object request only: modID of asking module

    const char *getName() const
    {
        return id.id;
    }
};

UTILEXPORT std::ostream &operator<<(std::ostream &, const coObjInfo &);

INLINE int operator==(const coObjInfo &a, const coObjInfo &b)
{
    return ((coObjID::compare(a.id, b.id) == 0)
            && (a.blockNo == b.blockNo)
            && (a.numBlocks == b.numBlocks)
            && (a.timeStep == b.timeStep)
            && (a.numTimeSteps == b.numTimeSteps)
            && (a.time == b.time));
}
}

#endif
