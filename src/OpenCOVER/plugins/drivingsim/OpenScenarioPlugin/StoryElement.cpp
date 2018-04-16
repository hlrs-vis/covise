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
	return state == running;
}
bool StoryElement::isStopped()
{
	return state == stopped;
}
bool StoryElement::isFinished()
{
	return state == finished;
}
