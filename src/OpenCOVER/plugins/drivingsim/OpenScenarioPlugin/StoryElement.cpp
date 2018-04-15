#include "StoryElement.h"

StoryElement::StoryElement():
    state(stopped)
{

}

void StoryElement::finish()
{
    state = finished;
}
void StoryElement::start()
{
    state = running;
}
void StoryElement::stop()
{
    state = stopped;
}
bool StoryElement::isRunning()
{
    if(state == running)
    {
        return true;
    }
    else
    {
        return false;
    }
}
bool StoryElement::isStopped()
{
    if(state == stopped)
    {
        return true;
    }
    else
    {
        return false;
    }
}
bool StoryElement::isFinished()
{
    if(state == finished)
    {
        return true;
    }
    else
    {
        return false;
    }
}
