#ifndef _THREADWORKER_H
#define _THREADWORKER_H

#include <exception>
#include <future>
#include <vector>
#include <iostream>

namespace opencover {
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
    typedef std::unique_ptr<std::vector<T>> Result;
    ThreadWorker() = default;
    ~ThreadWorker() { stop(); }
    // delete copy constructor and assignment operator because std::future is not copyable
    ThreadWorker(const ThreadWorker &) = delete;
    ThreadWorker &operator=(const ThreadWorker &) = delete;

    /**
     * @brief Checks the status of all threads in the pool.
     *
     * This function checks the status of all threads in the pool and returns true if all threads
     * have finished their tasks, or false otherwise.
     *
     * @return True if all threads have finished their tasks, false otherwise.
     */
    bool checkStatus() const
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

    bool isRunning() const { return threads.size() > 0; }

    /**
     * @brief Clears the thread pool.
     *
     * This function clears the thread pool if all threads have finished their tasks. If any thread
     * is still running, the pool is not cleared.
     */
    void clear()
    {
        if (checkStatus()) {
            threads.clear();
            threads.shrink_to_fit();
        }
    }

    /**
     * @brief Retrieves the result of the thread worker.
     * 
     * This function returns a unique pointer to a vector containing the result of the thread worker.
     * 
     * @return A unique pointer to a vector containing the result.
     */
    Result getResult()
    {
        if (checkStatus() && poolSize() > 0) {
            Result res = std::make_unique<std::vector<T>>();
            std::cout << "Worker finished" << "\n";
            for (auto &t: threads) {
                try {
                    res->push_back(t.get());
                } catch (const std::exception &e) {
                    std::cout << e.what() << "\n";
                }
            }
            clear();
            return res;
        }
        return nullptr;
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
    const auto &operator[](size_t i) const { return threads[i]; }

    /**
     * @brief Returns the list of threads in the pool.
     *
     * This function returns a reference to the list of threads in the pool.
     *
     * @return A reference to the list of threads in the pool.
     */
    const auto &threadsList() const { return threads; }

    /**
     * @brief Returns the size of the thread pool.
     *
     * This function returns the number of threads in the pool.
     *
     * @return The size of the thread pool.
     */
    const size_t poolSize() const { return threads.size(); }

private:
    void stop()
    {
        for (auto &t: threads) {
            try {
                t.get();
            } catch (const std::exception &e) {
            }
        }
        threads.clear();
    }
    // TODO: use std::queue instead of std::vector
    std::vector<std::future<T>> threads;
};
} // namespace utils
} // namespace opencover
#endif