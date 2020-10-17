#include "GA.h"
#include "DataManager.h"
#include "Sensor.h"
#include "Profiling.h"
#include "SensorPlacement.h"
#include <numeric>


PropertiesMaxCoverage1 GA::s_PropsMaxCoverage1;
PropertiesMaxCoverage2 GA::s_PropsMaxCoverage2;
float GA::s_VisibiltyThreshold{0.2};
bool GA::s_UseVisibilityThrsholdInOrientationComparison{false};
bool GA::s_OnlyKeepOrientationWithMostPoints{true};


/*
1) we have 2 cameras and 2 Prio2 Zones and 1 Prio1 Zone:
    - One camera can only see 1 Prio2 Zone
    - The other must decide whether completly cover Prio2 Zone or just partially cover Prio1 zone
    - Max Coverage1: prefer a fully covered Prio2 zone
*/
bool GA:: maxCoverage1(const Solution& sensorNetwork, MiddleCost &c)
{
    SP_PROFILE_FUNCTION();
    
    std::vector<int> sensorsPerPoint(m_NumberOfObservationPoints,0);
    std::vector<float> sumVisMat(m_NumberOfObservationPoints,0.0f);

    for(const auto& sensor : sensorNetwork.sensors)
    {
        std::transform(sensor->getVisibilityMatrix().begin(), sensor->getVisibilityMatrix().end(), sensorsPerPoint.begin(), sensorsPerPoint.begin(),[](float i, int j ) {return (i == 0.0f ? j : j+1);});  // count nbrOf sensors 
                                                                                              
        std::transform(sensor->getVisibilityMatrix().begin(), sensor->getVisibilityMatrix().end(), sumVisMat.begin(), sumVisMat.begin(), std::plus<float>());                                              // add coefficients 
    }
    
    std::vector<float> coveredPoints;
    coveredPoints.reserve(m_NumberOfObservationPoints);

    int penaltyCounterPRIO1{0};
    int penaltyCounterPRIO2{0};
    int sumCoveredPrio1Points{0};
    int sumCoveredPrio2Points{0};
    // float sumCoefficientsOfCoveredPrio1Points{0.0f};
    // float sumCoefficientsOfCoveredPrio2Points{0.0f};


    auto ItsensorsPerPoint = sensorsPerPoint.begin();
    auto ItRequiredSensors = m_RequiredSensorsPerPoint.begin();
    while( ItsensorsPerPoint != sensorsPerPoint.end())
    {
        int distance = std::distance(sensorsPerPoint.begin(), ItsensorsPerPoint);
        int diff = *ItsensorsPerPoint - *ItRequiredSensors;         // difference between actual an required number of sensors
        if( diff >=0 && (sumVisMat.at(distance) / m_RequiredSensorsPerPoint.at(distance) >= s_VisibiltyThreshold) )
        {
            if( m_RequiredSensorsPerPoint.at(distance) == (int)SafetyZone::Priority::PRIO1)
               sumCoveredPrio1Points += diff+1; //hier werden jetzt punkte mehr wie einmal abgedeckt !
            else if(m_RequiredSensorsPerPoint.at(distance) == (int)SafetyZone::Priority::PRIO2)
                sumCoveredPrio2Points +=diff+1;
        }
        else
        {
            if( m_RequiredSensorsPerPoint.at(distance) == (int)SafetyZone::Priority::PRIO1)
                penaltyCounterPRIO1 += std::abs(diff);
            else if(m_RequiredSensorsPerPoint.at(distance) == (int)SafetyZone::Priority::PRIO2)
                penaltyCounterPRIO2 += std::abs(diff);
        }
        
        // increment iterators
        if( ItsensorsPerPoint != sensorsPerPoint.end())
        {
            ++ItsensorsPerPoint;
            ++ItRequiredSensors;
        }
    }
    
        
    c.objective = -(s_PropsMaxCoverage1.weightingFactorPRIO1 * (sumCoveredPrio1Points - penaltyCounterPRIO1 * s_PropsMaxCoverage1.Penalty) + sumCoveredPrio2Points - (penaltyCounterPRIO2 * s_PropsMaxCoverage1.Penalty));
    calcCoverageProcentage(c, sumCoveredPrio1Points, sumCoveredPrio2Points);
    
    return true;
}

bool GA::maxCoverage2(const Solution &sensorNetwork, MiddleCost &c)
{
    SP_PROFILE_FUNCTION();
    std::vector<int> sensorsPerPoint(m_NumberOfObservationPoints,0);
    std::vector<float> sumVisMat(m_NumberOfObservationPoints,0.0f);

    for(const auto& sensor : sensorNetwork.sensors)
    {
        std::transform(sensor->getVisibilityMatrix().begin(), sensor->getVisibilityMatrix().end(), sensorsPerPoint.begin(), sensorsPerPoint.begin(),[](float i, int j ) {return (i == 0.0f ? j : j+1);});  // count nbrOf sensors 
                                                                                              
        std::transform(sensor->getVisibilityMatrix().begin(), sensor->getVisibilityMatrix().end(), sumVisMat.begin(), sumVisMat.begin(), std::plus<float>());                                              // add coefficients 
    }

    int sumCoveredPrio1 = sumOfCoveredPrio1Points(sensorsPerPoint, m_RequiredSensorsPerPoint, sumVisMat);
    int penalty{0};
    if(sumCoveredPrio1 > 0 && sumCoveredPrio1 < s_PropsMaxCoverage2.m_RequiredCoverage * m_NumberOfPrio1Points)
        penalty = s_PropsMaxCoverage2.m_PenaltyConstant * m_NumberOfPrio1Points / sumCoveredPrio1;
    else if(sumCoveredPrio1 == 0)
        penalty = s_PropsMaxCoverage2.m_PenaltyConstant * m_NumberOfPrio1Points;
    else if(sumCoveredPrio1 > s_PropsMaxCoverage2.m_RequiredCoverage * m_NumberOfPrio1Points)
        penalty = 0;
    
    int covered = coverEachPointWithMin1Sensor(sensorsPerPoint, sumVisMat);
   // std::cout<<"c_i: "<<covered <<"..."<<std::endl;
   // std::cout <<"penalty" <<penalty<<"..."<<std::endl;
    c.objective = - ( covered - penalty);

    //calcCoverageProcentage(c, sumCoveredPrio1, sumCoveredPrio2Points);

    return true; 
}

void GA::calcCoverageProcentage(MiddleCost &c, int sumCoveredPrio1Points, int sumCoveredPrio2Points)const
{
    c.coverage.prio1 = (float)sumCoveredPrio1Points / (float)m_NumberOfPrio1Points  * 100.0;
    c.coverage.prio2 = (float)sumCoveredPrio2Points / (float)(m_NumberOfObservationPoints - m_NumberOfPrio1Points) * 100.0;
    c.coverage.total = (float)(sumCoveredPrio1Points + sumCoveredPrio2Points) / (float)m_NumberOfObservationPoints * 100.0;
}


int GA::coverEachPointWithMin1Sensor(std::vector<int>& nbrOfSensors, std::vector<float> &sumVisMat)
{
    int counter{0};
    int c{0};
    for(const auto& nbr : nbrOfSensors)
    {
        if(nbr - 1 >= 0 && (sumVisMat.at(counter) / 1 >= s_VisibiltyThreshold) ) //geteilt durch n_min <--- checken! 
            c += nbr - 1;
        else
            c+=0;

        counter ++;
    }
    return c;
}

int GA::sumOfCoveredPrio1Points(const std::vector<int>& sensorsPerPoint, const std::vector<int>& requiredSensorsPerPoint, const std::vector<float>& sumVisMat) const
{
    auto ItsensorsPerPoint = sensorsPerPoint.begin();
    auto ItRequiredSensors = requiredSensorsPerPoint.begin();
    int sumCoveredPrio1{0};

    while( ItsensorsPerPoint != sensorsPerPoint.end())
    {
        int distance = std::distance(sensorsPerPoint.begin(), ItsensorsPerPoint);
        int diff = *ItsensorsPerPoint - *ItRequiredSensors;         // difference between actual an required number of sensors
        if( diff >=0 && (sumVisMat.at(distance) /  requiredSensorsPerPoint.at(distance) >= s_VisibiltyThreshold) )
        {
            if( requiredSensorsPerPoint.at(distance) == (int)SafetyZone::Priority::PRIO1)
                sumCoveredPrio1 += diff+1;
        }
        // increment iterators
        if( ItsensorsPerPoint != sensorsPerPoint.end())
        {
            ++ItsensorsPerPoint;
            ++ItRequiredSensors;
        }
    }
    return sumCoveredPrio1;
}

// float GA::sumOfCoveredPrio1Points(const std::vector<int>& sensorsPerPoint, const std::vector<int>& requiredSensorsPerPoint, const std::vector<float>& sumVisMat) const
// {
//     auto ItsensorsPerPoint = sensorsPerPoint.begin();
//     auto ItRequiredSensors = requiredSensorsPerPoint.begin();
//     float sumCoveredPrio1{0.0};

//     while( ItsensorsPerPoint != sensorsPerPoint.end())
//     {
//         int distance = std::distance(sensorsPerPoint.begin(), ItsensorsPerPoint);
//         int diff = *ItsensorsPerPoint - *ItRequiredSensors;         // difference between actual an required number of sensors
//         if( diff >=0 && (sumVisMat.at(distance) /  requiredSensorsPerPoint.at(distance) >= m_PropsMaxCoverage1.thresholdVisibility) )
//         {
//             if( requiredSensorsPerPoint.at(distance) == (int)SafetyZone::Priority::PRIO1)
//                 sumCoveredPrio1 += diff;
//         }
//         // increment iterators
//         if( ItsensorsPerPoint != sensorsPerPoint.end())
//         {
//             ++ItsensorsPerPoint;
//             ++ItRequiredSensors;
//         }
//     }
// }



GA::GA(FitnessFunctionType fitness)
{
    m_NumberOfSensors = calcNumberOfSensors();
    m_RequiredSensorsPerPoint = calcRequiredSensorsPerPoint();
    m_NumberOfObservationPoints = m_RequiredSensorsPerPoint.size();
    m_NumberOfPrio1Points = std::accumulate(m_RequiredSensorsPerPoint.begin(), m_RequiredSensorsPerPoint.end(), 0, [](int accValue, int currValue)
                                                                                                                    { return (currValue == (int)SafetyZone::Priority::PRIO1 ? accValue +=1 : accValue );});

    std::cout <<"nbr of prio1 points: " << m_NumberOfPrio1Points << std::endl;                                                                                                            

    SP_PROFILE_BEGIN_SESSION("SensorPlacement-Optimization","SensorPlacement-Optimization.json");

	EA::Chronometer timer;
	timer.tic();	
	using std::bind;
	using std::placeholders::_1;
	using std::placeholders::_2;
	using std::placeholders::_3;

    if(fitness == FitnessFunctionType::MaxCoverage1) // implement this here directly more down there->optimization Strategy not necessary
        m_FitnessFunction = std::bind(&GA::maxCoverage1,this,_1,_2);
    else if(fitness == FitnessFunctionType::MaxCoverage2)
        m_FitnessFunction = std::bind(&GA::maxCoverage2,this,_1,_2);

	ga_obj.problem_mode = EA::GA_MODE::SOGA;
    ga_obj.multi_threading = true;
    ga_obj.idle_delay_us = 1; // switch between threads quickly
    ga_obj.dynamic_threading = m_DynamicThreading;
    ga_obj.verbose = false;
    ga_obj.population = m_PopulationSize;
    ga_obj.generation_max = m_MaxGeneration;
    ga_obj.calculate_SO_total_fitness = std::bind( &GA::calculate_SO_total_fitness, this, _1);
    ga_obj.init_genes = std::bind( &GA::init_genes, this, _1,_2);
    ga_obj.eval_solution = std::bind( &GA::optimizationStrategy, this, _1,_2 );
    ga_obj.mutate = std::bind( &GA::mutate, this, _1,_2,_3 );
    ga_obj.crossover = std::bind( &GA::crossover, this, _1,_2,_3 );
    ga_obj.SO_report_generation = std::bind( &GA::SO_report_generation, this, _1,_2,_3 );
    ga_obj.crossover_fraction = m_CrossoverRate;
    ga_obj.mutation_rate = m_MutationRate;
    ga_obj.best_stall_max = 10;
    ga_obj.elite_count = ga_obj.population / 100 * 6; //6% of population size;
    ga_obj.solve();
	
	std::cout<<"The problem is optimized in "<<timer.toc()<<" seconds."<<std::endl;
    m_OptimizationTime = timer.toc();
    SP_PROFILE_END_SESSION();
}

bool GA::optimizationStrategy(const Solution& p, MiddleCost &c)
{
	return m_FitnessFunction(p,c);
}

// reject sensors which have same position!
bool rejectSensorPosition(const Solution& sensorNetwork, Orientation* newPos)
{
    auto itFound = std::find_if(sensorNetwork.sensors.begin(),sensorNetwork.sensors.end(),[&newPos](const Orientation* orientation){return (orientation->getMatrix().getTrans() == newPos->getMatrix().getTrans() ? true : false); }); 
    
    return (itFound == sensorNetwork.sensors.end() ? false : true);
}

void GA::init_genes(Solution& sensorNetwork,const std::function<double(void)> &rnd01)
{
    SP_PROFILE_FUNCTION();

    int count{0};
    while(count < m_NumberOfSensors )
    {
        Orientation* orient = m_SensorPool.getRandomOrientation(count, rnd01);
        if(!rejectSensorPosition(sensorNetwork, orient))
        {
            sensorNetwork.sensors.push_back(orient);
            count ++;
        }
    }
    // for(int i{0}; i < m_NumberOfSensors; ++i)
        // sensorNetwork.sensors.push_back(m_SensorPool.getRandomOrientation(i, rnd01));
    
}

Solution GA::mutate(const Solution& X_base,const std::function<double(void)> &rnd01, double shrink_scale)
{
    SP_PROFILE_FUNCTION();

    Solution X_new = X_base;
    int count = std::ceil(shrink_scale * m_NumberOfSensors);

    while(count != 0)
    {
        int randomPos = std::roundl(rnd01()*(m_NumberOfSensors-1));
        Orientation* randomOrient = m_SensorPool.getRandomOrientation(randomPos, rnd01);
        
        
        if(!rejectSensorPosition(X_new, randomOrient))
        {
            X_new.sensors.at(randomPos) = randomOrient;
            count--;
        }
    }

    return X_new;
}

Solution GA::crossover(const Solution& X1, const Solution& X2,const std::function<double(void)> &rnd01)
{
    SP_PROFILE_FUNCTION();

    // std::vector<int> possibleCutPoints;
    // for(int i{0}; i < m_SensorPool.getNbrOfSingleSensors(); i++)
    // {
        // if(possibleCutPoints.empty())
            // possibleCutPoints.push_back(1);
        // else
            // possibleCutPoints.push_back(possibleCutPoints.back() + 1);
    // }
    // auto sensorsPerZone = m_SensorPool.getNbrOfSensorsPerZone();
    // for(const auto& spz : sensorsPerZone)
    // {
        // if(possibleCutPoints.empty())
            // possibleCutPoints.push_back(spz);
        // else
            // possibleCutPoints.push_back(possibleCutPoints.back() + spz);
    // }    
    // int random = roundl(rnd01() * (possibleCutPoints.size()-1));
    // int cutPoint = possibleCutPoints.at(random);
    // std::cout << "random "<< random<< "CutPoint: "<<cutPoint <<std::endl
    
    int cutPoint = roundl(rnd01() * (m_NumberOfSensors-1));

    Solution X_new;
    X_new.sensors.clear(); // brauch man das ? 
    X_new.sensors.insert(X_new.sensors.begin(), X1.sensors.begin(), X1.sensors.begin() + cutPoint);
    X_new.sensors.insert(X_new.sensors.end(), X2.sensors.begin() + cutPoint,X2.sensors.end());

    return X1;
}

double GA::calculate_SO_total_fitness(const GA_Type::thisChromosomeType &X)
{
    SP_PROFILE_FUNCTION();

    // finalize the cost
    // obtain the final cost from the middle cost
    double final_cost{0};
    final_cost += X.middle_costs.objective;
    
    return final_cost;
}

void GA::SO_report_generation(int generation_number,const EA::GenerationType<Solution,MiddleCost> &last_generation,const Solution& best_genes)
{
    std::cout
        <<"Generation ["<<generation_number<<"], "
        <<"Best="<<last_generation.best_total_cost<<", "
        <<"Average="<<last_generation.average_cost<<", "
        //<<"Best genes(cameras)=("<<best_genes.to_string()<<")"<<", "
        <<"Best Coverage[%]="<<last_generation.chromosomes.at(last_generation.best_chromosome_index).middle_costs.coverage.to_string()<<", "
        <<"Exe_time="<<last_generation.exe_time
    <<std::endl;

    m_TotalCoverage = last_generation.chromosomes.at(last_generation.best_chromosome_index).middle_costs.coverage.total;
    m_Prio1Coverage = last_generation.chromosomes.at(last_generation.best_chromosome_index).middle_costs.coverage.prio1;
    m_Prio2Coverage = last_generation.chromosomes.at(last_generation.best_chromosome_index).middle_costs.coverage.prio2;
    m_FinalFitness =  last_generation.chromosomes.at(last_generation.best_chromosome_index).middle_costs.objective;
}

std::vector<Orientation> GA::getFinalOrientations() const
{
    std::vector<Orientation*> p_finalOrientations = ga_obj.last_generation.chromosomes.at(ga_obj.last_generation.best_chromosome_index).genes.sensors;
    std::vector<Orientation> finalOrientations;
    std::transform(std::begin(p_finalOrientations), std::end(p_finalOrientations),
                   std::back_inserter(finalOrientations),[](Orientation* orient){return *orient;});
    
    return finalOrientations; 
}

SensorPool::SensorPool()
{
    for(const auto& sensor : DataManager::GetSensors())
        m_Sensors.push_back(sensor.get());

    for(const auto& zone : DataManager::GetSensorZones())
        m_SensorZones.push_back(zone.get());
}


Orientation* SensorPool::getRandomOrientation(int pos, const std::function<double(void)> &rnd01) const
{
    Orientation* result{nullptr};

    // better Solution: create Class Random Generator and use it in openGA and in Sensors Class to return random Sensor
    if( pos < m_Sensors.size() )
    {
        int numberOfOrientations = m_Sensors.at(pos)->getNbrOfOrientations();
        int random = std::roundl(rnd01() * (numberOfOrientations - 1));

        result = m_Sensors.at(pos)->getSpecificOrientation(random);
    }
    else if(pos >= m_Sensors.size() && pos <= calcNumberOfSensors())
    {
        //pos = pos - m_Sensors.size();

        int sensorZonePos = getSensorInSensorZone(pos);

        int numberOfSensors = m_SensorZones.at(sensorZonePos)->getNumberOfPoints();
        int randomSensor = std::roundl(rnd01() * (numberOfSensors - 1));
        int numberOfOrientations = m_SensorZones.at(sensorZonePos)->getSpecificSensor(randomSensor)->getNbrOfOrientations();
        int randomOrientation = std::roundl(rnd01() * (numberOfOrientations - 1));

        result = m_SensorZones.at(sensorZonePos)->getSpecificSensor(randomSensor)->getSpecificOrientation(randomOrientation);
        //std::cout <<"Sensor Pos:"<<pos <<"Orientations Pos: "<<randomOrientation <<std::endl;
    }
    else
    {
        //std::cout << "No sensor at this position available" << std::endl;
    }

    return result;
}

std::vector<int> SensorPool::getNbrOfSensorsPerZone()const
{
    std::vector<int> sensors;
    for(const auto& zone : m_SensorZones)
        sensors.push_back(zone->getTargetNumberOfSensors());

    return sensors;
}
