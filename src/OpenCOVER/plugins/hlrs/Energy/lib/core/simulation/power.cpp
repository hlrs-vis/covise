#include "power.h"

namespace core::simulation::power {

void PowerSimulation::computeParameters() {
    computeParameter(m_buses);
    computeParameter(m_generators);
    computeParameter(m_transformators);
}
}  // namespace core::simulation::power
