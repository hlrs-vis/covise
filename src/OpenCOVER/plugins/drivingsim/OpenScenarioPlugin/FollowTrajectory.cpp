#include "FollowTrajectory.h"

FollowTrajectory::FollowTrajectory():
oscFollowTrajectory()
{}
FollowTrajectory::~FollowTrajectory(){}

void FollowTrajectory::finishedParsing()
{
	name = oscFollowTrajectory::CatalogReference->entryName.getValue();
}
