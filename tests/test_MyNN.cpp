#include <cassert>
#include <iostream>
#include <cmath>

#include "DenseLayer.hpp"
#include "Tensor.hpp"

using namespace myNN;

void test_ConstructorAndShape()
{
    Tensor t({2, 3});
    assert(t.getData()[0] == 2);
    assert(t.getData()[1] == 3);
    assert(t.size() == 6);
}

void test_fill()
{
    Tensor t({2, 3});
    t.fill(5);
    for (int i = 0; i < t.size(); i++)
        assert(t.getData()[i] == 5);
}

void test_apply()
{
    Tensor t({2, 2});
    t.fill(3);

    t.apply([](float x)
            { return x * x; });

    for (int i = 0; i < t.size(); i++)
        assert(t.getData()[i] == 9);
}

void test_transpose()
{
    Tensor t({2, 3});
    // Fill t with a simple pattern
    for (int i = 0; i < 6; i++)
        t.getData()[i] = i + 1;
    // t =
    // 1 2 3
    // 4 5 6

    Tensor T = t.transpose();
    // T =
    // 1 4
    // 2 5
    // 3 6

    assert(T.getData()[0] == 3);
    assert(T.getData()[1] == 2);

    assert(T(0, 0) == 1);
    assert(T(0, 1) == 4);
    assert(T(1, 0) == 2);
    assert(T(1, 1) == 5);
    assert(T(2, 0) == 3);
    assert(T(2, 1) == 6);
}

void test_matMul()
{
    Tensor A({2, 3});
    Tensor B({3, 2});

    // A =
    // 1 2 3
    // 4 5 6
    int k = 1;
    for (int i = 0; i < A.size(); i++)
        A.getData()[i] = k++;

    // B =
    // 7  8
    // 9  10
    // 11 12
    k = 7;
    for (int i = 0; i < B.size(); i++)
        B.getData()[i] = k++;

    Tensor C = A.matMul(B);

    // Expected:
    // [58  64]
    // [139 154]

    assert(C.getData()[0] == 2);
    assert(C.getData()[1] == 2);

    assert(C(0, 0) == 58);
    assert(C(0, 1) == 64);
    assert(C(1, 0) == 139);
    assert(C(1, 1) == 154);
}

void test_addBroadcast()
{
    Tensor A({2, 3});
    // A = row-major: 1 2 3 | 4 5 6
    for (int i = 0; i < A.size(); i++)
        A.getData()[i] = i + 1;

    // Test scalar broadcast
    Tensor s({1, 1});
    s.getData()[0] = 10;
    Tensor S = A.addBroadcast(s);
    for (int i = 0; i < S.size(); i++)
        assert(S.getData()[i] == A.getData()[i] + 10);

    // Test row broadcast
    Tensor r({1, 3});
    r.getData()[0] = 1;
    r.getData()[1] = 2;
    r.getData()[2] = 3;

    Tensor R = A.addBroadcast(r);
    // each row added by [1 2 3]
    assert(R(0, 0) == 2);
    assert(R(0, 1) == 4);
    assert(R(0, 2) == 6);
    assert(R(1, 0) == 5);
    assert(R(1, 1) == 7);
    assert(R(1, 2) == 9);

    // Test col broadcast
    Tensor c({2, 1});
    c.getData()[0] = 10;
    c.getData()[1] = 20;

    Tensor C = A.addBroadcast(c);
    // rows:
    // A row 0 + 10
    // A row 1 + 20
    assert(C(0, 0) == 11);
    assert(C(0, 1) == 12);
    assert(C(0, 2) == 13);
    assert(C(1, 0) == 24);
    assert(C(1, 1) == 25);
    assert(C(1, 2) == 26);
}

void test_denseLayerForward()
{
    DenseLayer layer(3, 2);

    layer.getWeights().print();
    layer.getBias().print();

    // manually set weights and bias for predictable output
    layer.getWeights()(0, 0) = 1;
    layer.getWeights()(0, 1) = 2;
    layer.getWeights()(1, 0) = 3;
    layer.getWeights()(1, 1) = 4;
    layer.getWeights()(2, 0) = 5;
    layer.getWeights()(2, 1) = 6;

    layer.getBias()(0, 0) = 10;
    layer.getBias()(0, 1) = 20;

    layer.getWeights().print();
    layer.getBias().print();

    Tensor input({2, 3});
    input.print();
    input(0, 0) = 1;
    input(0, 1) = 2;
    input(0, 2) = 3;
    input(1, 0) = 4;
    input(1, 1) = 5;
    input(1, 2) = 6;
    input.print();

    Tensor output = layer.forward(input);
    output.print();

    // expected calculations:
    // row0: [1*1+2*3+3*5 +10 , 1*2+2*4+3*6 +20] = [32, 48]
    // row1: [4*1+5*3+6*5 +10 , 4*2+5*4+6*6 +20] = [59, 84]
    float epsilon = 1e-5;
    std::cout << "0, 0: " << std::fabs(output(0, 0) - 32.0) << "\n";
    std::cout << "0, 1: " << std::fabs(output(0, 1) - 48.0) << "\n";
    std::cout << "1, 0: " << std::fabs(output(1, 0) - 59.0) << "\n";
    std::cout << "1, 1: " << std::fabs(output(1, 1) - 84.0) << "\n";

    assert(std::fabs(output(0, 0) - 32.0f) < epsilon);
    assert(std::fabs(output(0, 1) - 48.0f) < epsilon);
    assert(std::fabs(output(1, 0) - 59.0f) < epsilon);
    assert(std::fabs(output(1, 1) - 84.0f) < epsilon);
}

int main()
{
    std::cout << "Running Tensor tests...\n";

    test_ConstructorAndShape();
    test_fill();
    test_apply();
    test_transpose();
    test_matMul();
    test_addBroadcast();
    test_denseLayerForward();

    std::cout << "All tests passed successfully.\n";

    return 0;
}
