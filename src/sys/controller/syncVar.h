/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */
#ifndef CONTROLLER_SYNC_VAR_H
#define CONTROLLER_SYNC_VAR_H

#include <mutex>
#include <condition_variable>

namespace covise{
namespace controller{

template<typename T>
struct SyncVar{

    void setValue(const T &value)
    {
        {
            std::lock_guard<std::mutex> g{m_m};
            m_value = value;
            m_updated = true;
        }
        m_conVar.notify_one();
    }
    const T &waitForValue()
    {
        if (!m_updated)
        {
          std::unique_lock<std::mutex> lk(m_m);
          m_conVar.wait(lk);
        }
        std::lock_guard<std::mutex>g(m_m);
        m_updated = false;
        return m_value;
    }

private:
    std::mutex m_m;
    std::condition_variable m_conVar;
    bool m_updated = false;
    T m_value;
};

}
}

#endif // !CONTROLLER_SYNC_VAR_H