#ifndef _RESET_ON_COPY_PTR_H
#define _RESET_ON_COPY_PTR_H

#include <utility>
#include <memory>
namespace opencover::utils::pointer {

//copies of this pointer will be null

template <typename T>
class NullCopyPtr {
public:
    template <typename U>
    friend class NullCopyPtr;
    // Constructor
    explicit NullCopyPtr(T* ptr = nullptr) : m_ptr(ptr) {}

    // Copy constructor
    NullCopyPtr(const NullCopyPtr& other)
    {
        m_ptr = nullptr;
    }

    // Copy assignment operator
    template<typename U>
    NullCopyPtr& operator=(const NullCopyPtr<U>& other) 
    {
        if ((void*)this != (void*)&other) {
            m_ptr = nullptr;
        }
        return *this;
    }

    // Move assignment operator
    template<typename U>
    NullCopyPtr& operator=(NullCopyPtr<U>&& other) 
    {
        m_ptr = std::move(other.m_ptr);
        return *this;
    }
    
    // Dereference operator
    T& operator*() const 
    {
        return **m_ptr;
    }

    // Arrow operator
    T* operator->() const 
    {
        return m_ptr.get();
    }

    // Get the raw pointer
    T* get() const 
    {
        return m_ptr.get();
    }

    // Reset the pointer
    void reset(T* ptr = nullptr) 
    {
        m_ptr = ptr;
    }

    template<typename U>
    bool operator==(const NullCopyPtr<U>& other) const 
    {
        return m_ptr == other.m_ptr;
    }

    explicit operator bool() const {
        return m_ptr != nullptr;
    }
private:
    std::unique_ptr<T> m_ptr; 
};

template<typename T, typename...Args>
NullCopyPtr<T> makeNullCopyPtr(Args&&...args) 
{
    return NullCopyPtr(new T(std::forward<Args>(args)...));
}

} //opencover::utils::pointer
#endif // _RESET_ON_COPY_PTR_H