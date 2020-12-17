#include "testObject.h"
#include <iostream>

using namespace test;

TestObject::TestObject(){
std::cerr << "TestObjects was constructed" << std::endl;
}

TestObject::TestObject(const TestObject &other){
std::cerr << "TestObjects was copy constructed" << std::endl;
}

TestObject::TestObject(TestObject &&other){
std::cerr << "TestObjects was move  constructed"  << std::endl;
}

TestObject &TestObject::operator==(const TestObject &other){
std::cerr << "TestObjects was copied" << std::endl;
return *this;
}

TestObject &TestObject::operator==(TestObject &&other){
std::cerr << "TestObjects was moved"  << std::endl;
return *this;
}

TestObject::~TestObject(){
    std::cerr << "TestObjects was destroyed" << std::endl;
}