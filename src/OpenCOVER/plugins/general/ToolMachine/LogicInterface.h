#ifndef TOOLMACHINE_LOGIC_INTERFACE_H
#define TOOLMACHINE_LOGIC_INTERFACE_H

class LogicInterface {
public:
    virtual void update() = 0;
    virtual ~LogicInterface() = default;
};

#endif // TOOLMACHINE_LOGIC_INTERFACE_H