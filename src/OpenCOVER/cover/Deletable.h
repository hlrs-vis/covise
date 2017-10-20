#ifndef COVER_DELETABLE_H
#define COVER_DELETABLE_H

#include <vector>
#include <deque>

class Deletable
{
public:
    virtual ~Deletable();
    void deleteLater();
};

class DeletionManager
{
    friend class Deletable;

    static DeletionManager *s_instance;

    DeletionManager();
    ~DeletionManager();

    void add(Deletable *obj);

    std::deque<std::vector<Deletable *>> m_destroyList;

public:
    static DeletionManager *the();
    static void destroy();

    bool run();
};
#endif
