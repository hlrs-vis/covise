#ifndef _RESET_ON_COPY_PTR_H
#define _RESET_ON_COPY_PTR_H

#include <utility>

namespace opencover::utils::pointer {

template <typename T>
class ResetOnCopyPtr {
public:
    friend class ResetOnCopyPtr;
    // Constructor
    explicit ResetOnCopyPtr(T* ptr = nullptr) : m_ptr(ptr) {}

    // Destructor
    ~ResetOnCopyPtr() {
        delete m_ptr;
    }

    // Copy constructor
    ResetOnCopyPtr(const ResetOnCopyPtr& other) : m_ptr(other.m_ptr) {
        other.m_ptr = nullptr;
    }

    // Copy assignment operator
    template<typename U>
    ResetOnCopyPtr& operator=(const ResetOnCopyPtr<U>& other) {
        if ((void*)this != (void*)&other) {
            delete m_ptr;
            m_ptr = other.m_ptr;
            other.m_ptr = nullptr;
        }
        return *this;
    }

    
    // Move constructor
    ResetOnCopyPtr(ResetOnCopyPtr&& other) noexcept : m_ptr(other.m_ptr) {
        other.m_ptr = nullptr;
    }

    // Move assignment operator
    ResetOnCopyPtr& operator=(ResetOnCopyPtr&& other) noexcept {
        if (this != &other) {
            delete m_ptr;
            m_ptr = other.m_ptr;
            other.m_ptr = nullptr;
        }
        return *this;
    }

    // Dereference operator
    T& operator*() const {
        return *m_ptr;
    }

    // Arrow operator
    T* operator->() const {
        return m_ptr;
    }

    // Get the raw pointer
    T* get() const {
        return m_ptr;
    }

    // Reset the pointer
    void reset(T* ptr = nullptr) {
        delete m_ptr;
        m_ptr = ptr;
    }

    bool operator==(const ResetOnCopyPtr& other) const {
        return m_ptr == other.m_ptr;
    }

    explicit operator bool() const {
        return m_ptr != nullptr;
    }
private:
    mutable T* m_ptr; // Mutable to allow modification in const copy constructor and assignment operator
};

template<typename T, typename...Args>
ResetOnCopyPtr<T> makeResetOnCopy(Args&&...args) {
    return ResetOnCopyPtr(new T(std::forward<Args>(args)...));
}

} //opencover::utils::pointer
#endif // _RESET_ON_COPY_PTR_H