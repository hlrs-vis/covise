#pragma once

#include <vector>
#include <deque>

class vvDeletable
{
public:
    virtual ~vvDeletable();
    void deleteLater();
};

class DeletionManager
{
    friend class vvDeletable;

    static DeletionManager *s_instance;

    DeletionManager();
    ~DeletionManager();

    void add(vvDeletable *obj);

    std::deque<std::vector<vvDeletable *>> m_destroyList;

public:
    static DeletionManager *the();
    static void destroy();

    bool run();
};
