#ifndef _THREADWORKER_H
#define _THREADWORKER_H

#include <future>
#include <vector>

namespace utils {

/**
 * @brief A utility class for managing a pool of threads that perform asynchronous tasks.
 *
 * The ThreadWorker class provides a convenient way to manage a pool of threads that execute
 * tasks asynchronously. It allows adding threads to the pool, checking their status, and clearing
 * the pool when all threads have finished their tasks.
 *
 * @tparam T The type of the result returned by the threads.
 */
template<typename T>
struct ThreadWorker {
    ThreadWorker() = default;

    /**
     * @brief Checks the status of all threads in the pool.
     *
     * This function checks the status of all threads in the pool and returns true if all threads
     * have finished their tasks, or false otherwise.
     *
     * @return True if all threads have finished their tasks, false otherwise.
     */
    bool checkStatus()
    {
        for (auto &t: threads) {
            // thread finished
            if (!t.valid())
                continue;
            auto status = t.wait_for(std::chrono::milliseconds(1));
            if (status != std::future_status::ready)
                return false;
        }
        return true;
    }

    /**
     * @brief Clears the thread pool.
     *
     * This function clears the thread pool if all threads have finished their tasks. If any thread
     * is still running, the pool is not cleared.
     */
    void clear()
    {
        if (checkStatus())
            threads.clear();
    }

    /**
     * @brief Adds a thread to the pool.
     *
     * This function adds a thread to the pool. The thread is moved into the pool using std::move.
     *
     * @param t The thread to be added to the pool.
     */
    void addThread(std::future<T> &&t) { threads.push_back(std::move(t)); }

    /**
     * @brief Accesses a thread in the pool.
     *
     * This function allows accessing a thread in the pool by index.
     *
     * @param i The index of the thread to be accessed.
     * @return A reference to the thread at the specified index.
     */
    const auto &operator[](size_t i) { return threads[i]; }

    /**
     * @brief Returns the list of threads in the pool.
     *
     * This function returns a reference to the list of threads in the pool.
     *
     * @return A reference to the list of threads in the pool.
     */
    auto &threadsList() { return threads; }

    /**
     * @brief Returns the size of the thread pool.
     *
     * This function returns the number of threads in the pool.
     *
     * @return The size of the thread pool.
     */
    const size_t poolSize() const { return threads.size(); }

private:
    std::vector<std::future<T>> threads;
};
} // namespace utils
#endif