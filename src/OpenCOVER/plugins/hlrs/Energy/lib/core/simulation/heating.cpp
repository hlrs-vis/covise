#include "heating.h"

namespace core::simulation::heating {

void HeatingSimulation::computeParameters() {
    computeParameter(m_consumers);
    computeParameter(m_producers);
}
}  // namespace core::simulation::heating
