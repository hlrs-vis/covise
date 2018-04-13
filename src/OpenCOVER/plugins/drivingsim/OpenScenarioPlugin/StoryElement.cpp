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
