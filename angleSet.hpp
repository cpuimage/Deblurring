#ifndef CALC_ANGLES_H
#define CALC_ANGLES_H

#include <vector>

struct angle_t {
    int x, y;
    double angle;
    double dist;
    bool valid;
};

void computeProjectionAngleSet(std::vector<angle_t>& angles, int kernelSize);

#endif

