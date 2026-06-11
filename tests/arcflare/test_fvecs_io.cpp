// tests/arcflare/test_fvecs_io.cpp
#include "fvecs_io.h"
#include <cstdio>
#include <fstream>
#include <iostream>

static int failures = 0;
#define EXPECT_EQ(a,b) do { if((a)!=(b)){std::cerr<<__FILE__<<":"<<__LINE__\
  <<" FAIL "#a"!="#b" got "<<(a)<<" vs "<<(b)<<"\n";++failures;}}while(0)
#define EXPECT_NEAR(a,b,eps) do { if(std::abs((double)(a)-(double)(b))>(eps)){\
  std::cerr<<__FILE__<<":"<<__LINE__<<" FAIL "#a"≈"#b"\n";++failures;}}while(0)

static void writeFvecs(const std::string& p) {
    std::ofstream f(p, std::ios::binary);
    int32_t dim = 4;
    float rows[3][4] = {{1,2,3,4},{5,6,7,8},{9,10,11,12}};
    for (int i = 0; i < 3; ++i) {
        f.write(reinterpret_cast<char*>(&dim), sizeof(dim));
        f.write(reinterpret_cast<char*>(rows[i]), static_cast<std::streamsize>(static_cast<size_t>(dim) * sizeof(float)));
    }
}
static void writeIvecs(const std::string& p) {
    std::ofstream f(p, std::ios::binary);
    int32_t k = 3;
    int32_t ids[2][3] = {{0,1,2},{3,4,5}};
    for (int i = 0; i < 2; ++i) {
        f.write(reinterpret_cast<char*>(&k), sizeof(k));
        f.write(reinterpret_cast<char*>(ids[i]), static_cast<std::streamsize>(static_cast<size_t>(k) * sizeof(int32_t)));
    }
}

void testFvecs() {
    writeFvecs("/tmp/test_fvecs.fvecs");
    auto v = ArcFlare::loadFvecs("/tmp/test_fvecs.fvecs");
    EXPECT_EQ((int)v.size(), 3);
    EXPECT_EQ((int)v[0].size(), 4);
    EXPECT_NEAR(v[0][0], 1.0f, 1e-6f);
    EXPECT_NEAR(v[2][3], 12.0f, 1e-6f);
    std::remove("/tmp/test_fvecs.fvecs");
}

void testIvecs() {
    writeIvecs("/tmp/test_ivecs.ivecs");
    auto v = ArcFlare::loadIvecs("/tmp/test_ivecs.ivecs");
    EXPECT_EQ((int)v.size(), 2);
    EXPECT_EQ((int)v[0].size(), 3);
    EXPECT_EQ(v[0][0], 0);
    EXPECT_EQ(v[1][2], 5);
    std::remove("/tmp/test_ivecs.ivecs");
}

void testMissing() {
    bool threw = false;
    try { ArcFlare::loadFvecs("/tmp/does_not_exist_xyz.fvecs"); }
    catch (const std::exception&) { threw = true; }
    EXPECT_EQ(threw, true);

    threw = false;
    try { ArcFlare::loadIvecs("/tmp/does_not_exist_xyz.ivecs"); }
    catch (const std::exception&) { threw = true; }
    EXPECT_EQ(threw, true);
}

int main() {
    testFvecs();
    testIvecs();
    testMissing();
    if (failures > 0) { std::cerr << failures << " test(s) FAILED\n"; return 1; }
    std::cout << "All fvecs_io tests PASSED\n";
    return 0;
}
