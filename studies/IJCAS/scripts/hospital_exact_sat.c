/* Exact finite-cell rectangle SAT, matching HospitalEnvelopeChecker equations.
 * No quantization, reduced envelope, point-robot check or fast-math flags.
 * Inputs are contiguous numpy bool / int64 arrays and double local poses.
 */
#include <stdint.h>
#include <math.h>

int hospital_sat_batch(const uint8_t *occupied, const int64_t *integral,
                      int rows, int cols, double ox, double oy, double resolution,
                      double sx, double sy, double syaw, double clearance,
                      const double *poses, int count, uint64_t *stats) {
    stats[0] = stats[1] = stats[2] = 0;
    const double rc = cos(syaw), rs = sin(syaw);
    const double a = 1.630 / 2.0 + clearance, b = .965 / 2.0 + clearance;
    const double half = resolution / 2.0;
    const double xmax = ox + cols * resolution, ymax = oy + rows * resolution;
    for (int i = 0; i < count; ++i) {
        const double lx = poses[3*i], ly = poses[3*i+1], lyaw = poses[3*i+2];
        if (!isfinite(lx) || !isfinite(ly) || !isfinite(lyaw)) return -1;
        const double x = sx + rc*lx - rs*ly, y = sy + rs*lx + rc*ly;
        const double yaw = syaw + lyaw, c = cos(yaw), s = sin(yaw);
        const double ex = fabs(c)*a + fabs(s)*b, ey = fabs(s)*a + fabs(c)*b;
        stats[0]++;
        if (x-ex < ox || y-ey < oy || x+ex > xmax || y+ey > ymax) return 1;
        int x0 = (int)floor((x-ex-ox)/resolution - 1e-10);
        int y0 = (int)floor((y-ey-oy)/resolution - 1e-10);
        int x1 = (int)floor((x+ex-ox)/resolution + 1e-10);
        int y1 = (int)floor((y+ey-oy)/resolution + 1e-10);
        if (x0 < 0) x0 = 0;
        if (y0 < 0) y0 = 0;
        if (x1 >= cols) x1 = cols-1;
        if (y1 >= rows) y1 = rows-1;
        const int stride = cols+1;
        int64_t n = integral[(y1+1)*stride+x1+1] - integral[y0*stride+x1+1]
                  - integral[(y1+1)*stride+x0] + integral[y0*stride+x0];
        if (n == 0) { stats[1]++; continue; }
        stats[2]++;
        const double support = half * (fabs(c)+fabs(s));
        for (int yy = y0; yy <= y1; ++yy) {
            for (int xx = x0; xx <= x1; ++xx) {
                if (!occupied[yy*cols+xx]) continue;
                const double dx = ox + (xx+.5)*resolution - x;
                const double dy = oy + (yy+.5)*resolution - y;
                if (fabs(dx) <= ex+half+1e-12 && fabs(dy) <= ey+half+1e-12 &&
                    fabs(c*dx+s*dy) <= a+support+1e-12 &&
                    fabs(-s*dx+c*dy) <= b+support+1e-12) return 1;
            }
        }
    }
    return 0;
}
