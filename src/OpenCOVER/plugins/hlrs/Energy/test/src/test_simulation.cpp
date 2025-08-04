#include <gtest/gtest.h>
#include <lib/core/simulation/object.h>
#include <lib/core/simulation/object_factory.h>
#include <lib/core/simulation/heating.h>
#include <lib/core/simulation/power.h>
#include <lib/core/simulation/simulation.h>

using namespace core::simulation;
using namespace core::simulation::heating;
using namespace core::simulation::power;

namespace {

TEST(Simulation, ObjectConstructionWithoutData) {
    Object obj("testObj");
    EXPECT_EQ(obj.getName(), "testObj");
    EXPECT_TRUE(obj.getData().empty());
}

TEST(Simulation, ObjectConstructionWithData) {
    Object obj("testObj", {{"species", {1.0, 2.0, 3.0}}});
    EXPECT_EQ(obj.getName(), "testObj");
    EXPECT_FALSE(obj.getData().empty());
    EXPECT_DOUBLE_EQ(obj.getData().at("species")[0], 1.0);
    EXPECT_DOUBLE_EQ(obj.getData().at("species")[1], 2.0);
    EXPECT_DOUBLE_EQ(obj.getData().at("species")[2], 3.0);
}

TEST(Simulation, ObjectDataManipulation) {
    Object obj("testObj");
    obj.addData("species", 42.0);
    EXPECT_EQ(obj.getData().at("species").size(), 1);
    EXPECT_DOUBLE_EQ(obj.getData().at("species")[0], 42.0);

    obj.addData("species", std::vector<double>{1.0, 2.0});
    EXPECT_EQ(obj.getData().at("species").size(), 2);
    EXPECT_DOUBLE_EQ(obj.getData().at("species")[0], 1.0);
    EXPECT_DOUBLE_EQ(obj.getData().at("species")[1], 2.0);

    obj.emplace_back("emplace", 3.14);
    EXPECT_DOUBLE_EQ(obj.getData().at("emplace")[0], 3.14);
}

TEST(Simulation, CheckObjectFactoryForValidTypes) {
    auto objBus = createObject(ObjectType::Bus, "bus1", {{"voltage", {230.0}}});
    EXPECT_NE(objBus, nullptr);
    EXPECT_EQ(objBus->getName(), "bus1");
    EXPECT_FALSE(objBus->getData().empty());
    EXPECT_DOUBLE_EQ(objBus->getData().at("voltage")[0], 230.0);

    auto objGenerator = createObject(ObjectType::Generator, "gen1", {{"power", {500.0}}});
    EXPECT_NE(objGenerator, nullptr);
    EXPECT_EQ(objGenerator->getName(), "gen1");
    EXPECT_FALSE(objGenerator->getData().empty());
    EXPECT_DOUBLE_EQ(objGenerator->getData().at("power")[0], 500.0);

    auto objBuilding = createObject(ObjectType::Building, "build1", {{"area", {100.0}}});
    EXPECT_NE(objBuilding, nullptr);
    EXPECT_EQ(objBuilding->getName(), "build1");
    EXPECT_FALSE(objBuilding->getData().empty());
    EXPECT_DOUBLE_EQ(objBuilding->getData().at("area")[0], 100.0);

    auto objCable = createObject(ObjectType::Cable, "cable1", {{"length", {50.0}}});
    EXPECT_NE(objCable, nullptr);
    EXPECT_EQ(objCable->getName(), "cable1");
    EXPECT_FALSE(objCable->getData().empty());
    EXPECT_DOUBLE_EQ(objCable->getData().at("length")[0], 50.0);

    auto objTransformator = createObject(ObjectType::Transformator, "trans1", {{"efficiency", {0.95}}});
    EXPECT_NE(objTransformator, nullptr);
    EXPECT_EQ(objTransformator->getName(), "trans1");
    EXPECT_FALSE(objTransformator->getData().empty());
    EXPECT_DOUBLE_EQ(objTransformator->getData().at("efficiency")[0], 0.95);

    auto objConsumer = createObject(ObjectType::Consumer, "cons1", {{"demand", {100.0}}});
    EXPECT_NE(objConsumer, nullptr);
    EXPECT_EQ(objConsumer->getName(), "cons1");
    EXPECT_FALSE(objConsumer->getData().empty());
    EXPECT_DOUBLE_EQ(objConsumer->getData().at("demand")[0], 100.0);

    auto objProducer = createObject(ObjectType::Producer, "prod1", {{"output", {200.0}}});
    EXPECT_NE(objProducer, nullptr);
    EXPECT_EQ(objProducer->getName(), "prod1");
    EXPECT_FALSE(objProducer->getData().empty());
    EXPECT_DOUBLE_EQ(objProducer->getData().at("output")[0], 200.0);
}

TEST(Simulation, CheckObjectFactoryForInvalidType) {
    EXPECT_THROW(createObject(ObjectType::Unknown, "unknown", {}), std::runtime_error);
}

TEST(Simulation, HeatingSimulation) {
    HeatingSimulation sim;
    auto &consumers = sim.Consumers();
    auto &producers = sim.Producers();

    consumers.emplace("consumer1", std::make_unique<Object>("consumer1"));
    producers.emplace("producer1", std::make_unique<Object>("producer1"));

    consumers.at("consumer1")->addData("heat_demand", 100.0);
    producers.at("producer1")->addData("heat_output", 200.0);

    EXPECT_EQ(consumers.size(), 1);
    EXPECT_EQ(producers.size(), 1);
    EXPECT_EQ(consumers.at("consumer1")->getData().at("heat_demand")[0], 100.0);
    EXPECT_EQ(producers.at("producer1")->getData().at("heat_output")[0], 200.0);
}

}
