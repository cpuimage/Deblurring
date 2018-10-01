#include <cstdlib>
#include <cmath>
#include <algorithm>

#include "image.hpp"
#include "angleSet.hpp"

// greatest common factor
static int gcd(int a, int b)
{
    return b ? gcd(b, a % b) : a;
}

/// compute the angle set that allows to reach each pixels in a square
/// of size kernelSize*kernelSize and starting at position 0,0
static void computeProjectionHalfAngleSet(std::vector<angle_t>& angles, int kernelSize)
{
    for (int x = 0; x < kernelSize+1; x++) {
        for (int y = 0; y < kernelSize+1; y++) {
            // if gcd(x,y) is not one, this means that we could find (x',y')
            // such that (x,y) = (i*x',i*y'), thus we would have the same angle twice
            if (gcd(x, y) != 1)
                continue;

            angle_t angle;
            angle.angle = std::atan2(y, x);
            angle.x = x;
            angle.y = y;
            angles.push_back(angle);
        }
    }

    struct angle_sort_by_angle {
        bool operator() (const angle_t& a, const angle_t& b) {
            return a.angle > b.angle;
        }
    };
    std::sort(angles.begin(), angles.end(), angle_sort_by_angle());
}

/// same as computeProjectionHalfAngleSet but with additional mirrored angles
void computeProjectionAngleSet(std::vector<angle_t>& angles, int kernelSize)
{
    // get angles from pi/2 to 0
    computeProjectionHalfAngleSet(angles, kernelSize);

    // copy the angles in reverse order (so that the final set goes from pi/2 to -pi/2)
    int s = angles.size();

    // don't copy the first one and the last one (theta=pi/2 and theta=0)
    angles.resize(s * 2 - 2);

    for (int i = 0; i < s-2; i++) {
        angle_t ref = angles[s-1 - i - 1];
        angles[s + i] = ref;
        angles[s + i].angle = - ref.angle;
        angles[s + i].y = - ref.y;
    }
}

