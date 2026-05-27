// tests/ngtaq/test_soa_graph.cpp
#include "NGT/NGTAQ/SoAGraph.h"
#include <cassert>
#include <iostream>
#include <sstream>
#include <vector>

static int failures = 0;
#define EXPECT_EQ(a,b) do { if((a)!=(b)){std::cerr<<__FILE__<<":"<<__LINE__<<" FAIL "#a"=="#b" got "<<(a)<<" vs "<<(b)<<"\n";++failures;}}while(0)
#define EXPECT_TRUE(c) do { if(!(c)){std::cerr<<__FILE__<<":"<<__LINE__<<" FAIL "#c"\n";++failures;}}while(0)

void testAddAndAccess() {
    // Add 2 nodes, verify BQ planes and neighbor access
    NGTAQ::SoAGraph graph(4 /* words per node = 256/64 */);

    std::vector<uint64_t> sign0 = {0xAAAA,0xBBBB,0xCCCC,0xDDDD};
    std::vector<uint64_t> mag0  = {0x1111,0x2222,0x3333,0x4444};
    uint32_t id0 = graph.addNode(sign0.data(), mag0.data());
    EXPECT_EQ(id0, 0u);

    std::vector<uint64_t> sign1 = {0xFFFF,0xEEEE,0xDDDD,0xCCCC};
    std::vector<uint64_t> mag1  = {0x9999,0x8888,0x7777,0x6666};
    uint32_t id1 = graph.addNode(sign1.data(), mag1.data());
    EXPECT_EQ(id1, 1u);

    graph.finalizeCSR();

    // Verify BQ plane retrieval
    EXPECT_EQ(graph.getSignPlane(id0)[0], 0xAAAAu);
    EXPECT_EQ(graph.getMagPlane(id0)[1],  0x2222u);
    EXPECT_EQ(graph.getSignPlane(id1)[0], 0xFFFFu);

    // Set neighbors for id0 → [id1]
    graph.setNeighbors(id0, {id1});
    auto neighbors = graph.getNeighbors(id0);
    EXPECT_EQ(neighbors.size(), 1u);
    EXPECT_EQ(neighbors[0], id1);
}

void testTombstone() {
    NGTAQ::SoAGraph graph(2);
    std::vector<uint64_t> s(2, 0), m(2, 0);
    uint32_t id0 = graph.addNode(s.data(), m.data());
    uint32_t id1 = graph.addNode(s.data(), m.data());
    graph.finalizeCSR();
    EXPECT_EQ(graph.activeCount(), 2u);

    graph.removeNode(id0);
    EXPECT_EQ(graph.activeCount(), 1u);
    EXPECT_TRUE(graph.isTombstone(id0));
    EXPECT_TRUE(!graph.isTombstone(id1));
}

void testRebuildRemovesTombstones() {
    NGTAQ::SoAGraph graph(2);
    std::vector<uint64_t> s(2, 0xAAAA), m(2, 0xBBBB);
    uint32_t id0 = graph.addNode(s.data(), m.data());
    uint32_t id1 = graph.addNode(s.data(), m.data());
    uint32_t id2 = graph.addNode(s.data(), m.data());
    graph.finalizeCSR();

    graph.setNeighbors(id0, {id1, id2});
    graph.setNeighbors(id1, {id0});
    graph.setNeighbors(id2, {id0});

    graph.removeNode(id1);
    graph.rebuild();  // Should compact out id1

    // After rebuild: 2 active nodes remain
    EXPECT_EQ(graph.activeCount(), 2u);
    // Tombstone nodes should no longer appear in neighbor lists after rebuild
    for (uint32_t id = 0; id < graph.size(); id++) {
        if (graph.isTombstone(id)) continue;
        auto nbrs = graph.getNeighbors(id);
        for (uint32_t nbr : nbrs) {
            EXPECT_TRUE(!graph.isTombstone(nbr));
        }
    }
}

void testSerializeDeserialize() {
    NGTAQ::SoAGraph graph(2);
    std::vector<uint64_t> s = {0xDEADBEEF, 0xCAFEBABE};
    std::vector<uint64_t> m = {0x12345678, 0xABCDEF01};
    uint32_t id0 = graph.addNode(s.data(), m.data());
    graph.finalizeCSR();
    graph.setNeighbors(id0, {});

    std::ostringstream oss;
    graph.serialize(oss);
    std::string data = oss.str();

    NGTAQ::SoAGraph graph2(2);
    std::istringstream iss(data);
    graph2.deserialize(iss);

    EXPECT_EQ(graph2.getSignPlane(id0)[0], 0xDEADBEEFu);
    EXPECT_EQ(graph2.getSignPlane(id0)[1], 0xCAFEBABEu);
}

int main() {
    testAddAndAccess();
    testTombstone();
    testRebuildRemovesTombstones();
    testSerializeDeserialize();
    if (failures > 0) { std::cerr << failures << " test(s) FAILED\n"; return 1; }
    std::cout << "All SoAGraph tests PASSED\n";
    return 0;
}
