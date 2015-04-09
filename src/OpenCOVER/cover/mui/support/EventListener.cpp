#include "EventListener.h"
#include <iostream>

using namespace mui;

EventListener::EventListener()
{
}

EventListener::~EventListener()
{
}

void EventListener::muiEvent(Element *muiItem)
{
    std::cerr << "EventListener::muiEvent(): was called and does nothing" << std::endl;
}

void EventListener::muiPressEvent(Element *muiItem)
{
    std::cerr << "EventListener::muiPressEvent(): was called and does nothing" << std::endl;
}

void EventListener::muiValueChangeEvent(Element *muiItem)
{
    std::cerr << "EventListener::muiValueChangeEvent(): was called and does nothing" << std::endl;
}

void EventListener::muiClickEvent(Element *muiItem)
{
    std::cerr << "EventListener::muiClickEvent(): was called and does nothing" << std::endl;
}

void EventListener::muiReleaseEvent(Element *muiItem)
{
    std::cerr << "EventListener::muiReleaseEvent(): was called and does nothing" << std::endl;
}
