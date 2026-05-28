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
    // Add 2 nodes with interleaved bq buffers, verify getNodeBQ and neighbors
    // words=4: bq_buf = [s0, m0, s1, m1, s2, m2, s3, m3]
    NGTAQ::SoAGraph graph(4 /* words per node = 256/64 */);

    // Node 0: sign=[0xAAAA,0xBBBB,0xCCCC,0xDDDD], mag=[0x1111,0x2222,0x3333,0x4444]
    std::vector<uint64_t> bq0 = {
        0xAAAA, 0x1111,  // s0, m0
        0xBBBB, 0x2222,  // s1, m1
        0xCCCC, 0x3333,  // s2, m2
        0xDDDD, 0x4444   // s3, m3
    };
    uint32_t id0 = graph.addNode(bq0.data());
    EXPECT_EQ(id0, 0u);

    // Node 1: sign=[0xFFFF,0xEEEE,0xDDDD,0xCCCC], mag=[0x9999,0x8888,0x7777,0x6666]
    std::vector<uint64_t> bq1 = {
        0xFFFF, 0x9999,
        0xEEEE, 0x8888,
        0xDDDD, 0x7777,
        0xCCCC, 0x6666
    };
    uint32_t id1 = graph.addNode(bq1.data());
    EXPECT_EQ(id1, 1u);

    graph.finalizeCSR();

    // Verify BQ plane retrieval via getNodeBQ
    // sign word 0 of id0 = bq[0] = 0xAAAA
    EXPECT_EQ(graph.getNodeBQ(id0)[0], 0xAAAAu);
    // mag word 1 of id0 = bq[1*2+1] = bq[3] = 0x2222
    EXPECT_EQ(graph.getNodeBQ(id0)[3], 0x2222u);
    // sign word 0 of id1 = bq[0] = 0xFFFF
    EXPECT_EQ(graph.getNodeBQ(id1)[0], 0xFFFFu);

    // Set neighbors for id0 → [id1]
    graph.setNeighbors(id0, {id1});
    auto neighbors = graph.getNeighbors(id0);
    EXPECT_EQ(neighbors.size(), 1u);
    EXPECT_EQ(neighbors[0], id1);
}

void testTombstone() {
    NGTAQ::SoAGraph graph(2);
    std::vector<uint64_t> bq(4, 0);  // 2 words * 2 = 4 entries, all zero
    uint32_t id0 = graph.addNode(bq.data());
    uint32_t id1 = graph.addNode(bq.data());
    graph.finalizeCSR();
    EXPECT_EQ(graph.activeCount(), 2u);

    graph.removeNode(id0);
    EXPECT_EQ(graph.activeCount(), 1u);
    EXPECT_TRUE(graph.isTombstone(id0));
    EXPECT_TRUE(!graph.isTombstone(id1));
}

void testRebuildRemovesTombstones() {
    NGTAQ::SoAGraph graph(2);
    // words=2: bq_buf = [s0, m0, s1, m1]
    std::vector<uint64_t> bq = {0xAAAA, 0xBBBB, 0xAAAA, 0xBBBB};
    uint32_t id0 = graph.addNode(bq.data());
    uint32_t id1 = graph.addNode(bq.data());
    uint32_t id2 = graph.addNode(bq.data());
    graph.finalizeCSR();

    graph.setNeighbors(id0, {id1, id2});
    graph.setNeighbors(id1, {id0});
    graph.setNeighbors(id2, {id0});

    graph.removeNode(id1);
    graph.rebuild();  // Should compact out id1

    EXPECT_EQ(graph.activeCount(), 2u);
    EXPECT_EQ(graph.size(), 2u);
    for (uint32_t id = 0; id < graph.size(); id++) {
        if (graph.isTombstone(id)) continue;
        auto nbrs = graph.getNeighbors(id);
        for (uint32_t nbr : nbrs) {
            EXPECT_TRUE(nbr < graph.size());
            EXPECT_TRUE(!graph.isTombstone(nbr));
        }
    }
    bool found_correct_remap = false;
    for (uint32_t id = 0; id < graph.size(); id++) {
        auto nbrs = graph.getNeighbors(id);
        for (uint32_t nbr : nbrs) {
            EXPECT_TRUE(nbr < graph.size());
            EXPECT_TRUE(!graph.isTombstone(nbr));
        }
        if (nbrs.size() > 0) found_correct_remap = true;
    }
    EXPECT_TRUE(found_correct_remap);
}

void testSerializeDeserialize() {
    NGTAQ::SoAGraph graph(2);
    // words=2: bq_buf[0]=s0=0xDEADBEEF, bq_buf[1]=m0=0x12345678,
    //          bq_buf[2]=s1=0xCAFEBABE, bq_buf[3]=m1=0xABCDEF01
    std::vector<uint64_t> bq0 = {0xDEADBEEF, 0x12345678, 0xCAFEBABE, 0xABCDEF01};
    uint32_t id0 = graph.addNode(bq0.data());

    std::vector<uint64_t> bq1 = {0x11223344, 0xAABBCCDD, 0x55667788, 0xEEFF0011};
    uint32_t id1 = graph.addNode(bq1.data());
    graph.finalizeCSR();
    graph.setNeighbors(id0, {id1});
    graph.setNeighbors(id1, {id0});

    std::ostringstream oss;
    graph.serialize(oss);
    std::string data = oss.str();

    NGTAQ::SoAGraph graph2(2);
    std::istringstream iss(data);
    graph2.deserialize(iss);

    // Sign word 0 of id0 = getNodeBQ(id0)[0] = 0xDEADBEEF
    EXPECT_EQ(graph2.getNodeBQ(id0)[0], 0xDEADBEEFu);
    // Sign word 1 of id0 = getNodeBQ(id0)[2] = 0xCAFEBABE
    EXPECT_EQ(graph2.getNodeBQ(id0)[2], 0xCAFEBABEu);
    // Mag word 0 of id0 = getNodeBQ(id0)[1] = 0x12345678
    EXPECT_EQ(graph2.getNodeBQ(id0)[1], 0x12345678u);
    // Mag word 1 of id0 = getNodeBQ(id0)[3] = 0xABCDEF01
    EXPECT_EQ(graph2.getNodeBQ(id0)[3], 0xABCDEF01u);
    EXPECT_EQ(graph2.activeCount(), 2u);
    auto nbrs = graph2.getNeighbors(id0);
    EXPECT_EQ(nbrs.size(), 1u);
    EXPECT_EQ(nbrs[0], id1);
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
