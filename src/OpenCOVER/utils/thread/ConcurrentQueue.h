#pragma once
#include <condition_variable>
#include <mutex>
#include <optional>  // C++17 for std::optional
#include <queue>

template <typename T>
class ConcurrentQueue {
 public:
  void push(T value) {
    {
      std::lock_guard<std::mutex> lock(m_mutex);
      m_queue.push(std::move(value));
    }
    m_cv.notify_one();  // Notify one waiting consumer
  }

  // Pop an item. Returns std::nullopt if queue is closed and empty.
  std::optional<T> pop() {
    std::unique_lock<std::mutex> lock(m_mutex);
    m_cv.wait(lock, [this] { return !m_queue.empty() || m_closed; });

    if (m_queue.empty() && m_closed) {
      return std::nullopt;  // Queue closed and empty, no more items
    }

    T value = std::move(m_queue.front());
    m_queue.pop();
    return value;
  }

  // Signals that no more elements will be pushed to the queue
  void close() {
    {
      std::lock_guard<std::mutex> lock(m_mutex);
      m_closed = true;
    }
    m_cv.notify_all();  // Notify all waiting consumers that queue is closed
  }

  bool closed() const { return m_closed; }
  void reset() {
    std::lock_guard<std::mutex> lock(m_mutex);
    m_closed = false;
    std::queue<T> emptyQueue;
    std::swap(m_queue, emptyQueue);  // Clear the queue
  }

 private:
  std::queue<T> m_queue;
  std::mutex m_mutex;
  std::condition_variable m_cv;
  bool m_closed = false;
};
