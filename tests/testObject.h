#ifndef TEST_TEST_OBJECT_H
#define TEST_TEST_OBJECT_H
namespace test{

struct TestObject
{
    TestObject();
    TestObject(const TestObject &other);
    TestObject(TestObject &&other);
    TestObject &operator==(const TestObject &other);;
    TestObject &operator==(TestObject &&other);
    ~TestObject();
};
} // namespace test

#endif //!TEST_TEST_OBJECT_H