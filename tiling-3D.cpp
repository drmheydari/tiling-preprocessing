#include <iostream>
#include <vector>
#include <array>
#include <random>
#include <chrono>
#include <algorithm>
#include <numeric>
#include <unordered_set>
#include <memory>
#include <future>
#include <thread>
#include <cmath>
#include <iomanip>
#include <immintrin.h>
#include <omp.h>
#include <execution>

extern "C" {
    #include <libqhull_r/libqhull_r.h>
    #include <libqhull_r/qset_r.h>
    #include <libqhull_r/geom_r.h>
    #include <libqhull_r/merge_r.h>
    #include <libqhull_r/poly_r.h>
    #include <libqhull_r/io_r.h>
    #include <libqhull_r/stat_r.h>
}

using Point3D = std::array<double, 3>;
using Points = std::vector<Point3D>;

class QHullWrapper {
private:
    qhT qh_qh;
    bool initialized = false;

public:
    QHullWrapper() {
        qh_zero(&qh_qh, stderr);
    }

    ~QHullWrapper() {
        if (initialized) {
            qh_freeqhull(&qh_qh, !qh_ERRnone);
            int curlong, totlong;
            qh_memfreeshort(&qh_qh, &curlong, &totlong);
        }
    }

    struct ConvexHullResult {
        std::vector<int> vertices;
        std::vector<std::array<double, 4>> equations;
        bool success = false;
    };

    ConvexHullResult computeConvexHull(const Points& points) {
        ConvexHullResult result;

        if (points.size() < 4) {
            return result;
        }

        if (initialized) {
            qh_freeqhull(&qh_qh, !qh_ERRnone);
            int curlong, totlong;
            qh_memfreeshort(&qh_qh, &curlong, &totlong);
        }

        qh_zero(&qh_qh, stderr);
        initialized = true;

        char flags[] = "qhull s";
        FILE* outfile = nullptr;
        FILE* errfile = nullptr;

        if (qh_new_qhull(&qh_qh, 3, points.size(),
                         reinterpret_cast<coordT*>(const_cast<Point3D*>(points.data())),
                         False, flags, outfile, errfile)) {
            return result;
        }

        result.vertices.reserve(qh_qh.num_vertices);
        vertexT* vertex;
        for (vertex = qh_qh.vertex_list; vertex && vertex->next; vertex = vertex->next) {
            result.vertices.push_back(qh_pointid(&qh_qh, vertex->point));
        }

        result.equations.reserve(qh_qh.num_facets);
        facetT* facet;
        for (facet = qh_qh.facet_list; facet && facet->next; facet = facet->next) {
            if (!facet->upperdelaunay) {
                std::array<double, 4> eq;
                for (int i = 0; i < 3; ++i) {
                    eq[i] = facet->normal[i];
                }
                eq[3] = facet->offset;
                result.equations.push_back(eq);
            }
        }

        result.success = true;
        return result;
    }
};

class TilingV2 {
private:
    const int N;
    static constexpr double RANGE_MIN = 0.0;
    static constexpr double RANGE_MAX = 200.0;
    static constexpr double TOL = 1e-10;
    static constexpr int MAX_DEPTH = 7;
    static constexpr int MIN_POINTS = 200;

    Points points;
    std::mt19937 rng;

    enum class Corner {
        TOP_LEFT_NEAR,
        TOP_LEFT_FAR,
        TOP_RIGHT_NEAR,
        TOP_RIGHT_FAR,
        BOTTOM_LEFT_NEAR,
        BOTTOM_LEFT_FAR,
        BOTTOM_RIGHT_NEAR,
        BOTTOM_RIGHT_FAR
    };

    struct ExtremePoints {
        Point3D minX, maxX, minY, maxY, minZ, maxZ;
        size_t minX_idx, maxX_idx, minY_idx, maxY_idx, minZ_idx, maxZ_idx;
    };

    struct ExtremeIndices {
        size_t x_idx, y_idx, z_idx;
    };

    // SIMD-optimized single-pass extreme finding
    ExtremePoints findAllExtremesFast(const Points& pts) const {
        if (pts.empty()) {
            return {};
        }

        ExtremePoints extremes;
        extremes.minX = extremes.maxX = extremes.minY = extremes.maxY = extremes.minZ = extremes.maxZ = pts[0];
        extremes.minX_idx = extremes.maxX_idx = extremes.minY_idx = extremes.maxY_idx = extremes.minZ_idx = extremes.maxZ_idx = 0;

        const size_t size = pts.size();

        // Process 4 points at a time with SIMD
        size_t i = 0;
        for (; i + 3 < size; i += 4) {
            // Load 4 points worth of data
            __m256d x_vals = _mm256_set_pd(pts[i+3][0], pts[i+2][0], pts[i+1][0], pts[i][0]);
            __m256d y_vals = _mm256_set_pd(pts[i+3][1], pts[i+2][1], pts[i+1][1], pts[i][1]);
            __m256d z_vals = _mm256_set_pd(pts[i+3][2], pts[i+2][2], pts[i+1][2], pts[i][2]);

            // Current extremes
            __m256d cur_min_x = _mm256_broadcast_sd(&extremes.minX[0]);
            __m256d cur_max_x = _mm256_broadcast_sd(&extremes.maxX[0]);
            __m256d cur_min_y = _mm256_broadcast_sd(&extremes.minY[1]);
            __m256d cur_max_y = _mm256_broadcast_sd(&extremes.maxY[1]);
            __m256d cur_min_z = _mm256_broadcast_sd(&extremes.minZ[2]);
            __m256d cur_max_z = _mm256_broadcast_sd(&extremes.maxZ[2]);

            // Compare and update (fallback to scalar for index tracking)
            alignas(32) double x_arr[4], y_arr[4], z_arr[4];
            _mm256_store_pd(x_arr, x_vals);
            _mm256_store_pd(y_arr, y_vals);
            _mm256_store_pd(z_arr, z_vals);

            for (int j = 0; j < 4; ++j) {
                size_t idx = i + j;
                if (x_arr[j] < extremes.minX[0]) { extremes.minX = pts[idx]; extremes.minX_idx = idx; }
                if (x_arr[j] > extremes.maxX[0]) { extremes.maxX = pts[idx]; extremes.maxX_idx = idx; }
                if (y_arr[j] < extremes.minY[1]) { extremes.minY = pts[idx]; extremes.minY_idx = idx; }
                if (y_arr[j] > extremes.maxY[1]) { extremes.maxY = pts[idx]; extremes.maxY_idx = idx; }
                if (z_arr[j] < extremes.minZ[2]) { extremes.minZ = pts[idx]; extremes.minZ_idx = idx; }
                if (z_arr[j] > extremes.maxZ[2]) { extremes.maxZ = pts[idx]; extremes.maxZ_idx = idx; }
            }
        }

        // Handle remaining points
        for (; i < size; ++i) {
            const auto& pt = pts[i];
            if (pt[0] < extremes.minX[0]) { extremes.minX = pt; extremes.minX_idx = i; }
            if (pt[0] > extremes.maxX[0]) { extremes.maxX = pt; extremes.maxX_idx = i; }
            if (pt[1] < extremes.minY[1]) { extremes.minY = pt; extremes.minY_idx = i; }
            if (pt[1] > extremes.maxY[1]) { extremes.maxY = pt; extremes.maxY_idx = i; }
            if (pt[2] < extremes.minZ[2]) { extremes.minZ = pt; extremes.minZ_idx = i; }
            if (pt[2] > extremes.maxZ[2]) { extremes.maxZ = pt; extremes.maxZ_idx = i; }
        }

        return extremes;
    }

    ExtremeIndices getExtremePoints(const std::vector<double>& x, const std::vector<double>& y, const std::vector<double>& z, Corner corner) const {
        if (x.empty()) return {0, 0, 0};

        size_t min_x_idx = std::distance(x.begin(), std::min_element(x.begin(), x.end()));
        size_t max_x_idx = std::distance(x.begin(), std::max_element(x.begin(), x.end()));
        size_t min_y_idx = std::distance(y.begin(), std::min_element(y.begin(), y.end()));
        size_t max_y_idx = std::distance(y.begin(), std::max_element(y.begin(), y.end()));
        size_t min_z_idx = std::distance(z.begin(), std::min_element(z.begin(), z.end()));
        size_t max_z_idx = std::distance(z.begin(), std::max_element(z.begin(), z.end()));

        switch (corner) {
            case Corner::TOP_LEFT_NEAR:
                return {min_x_idx, max_y_idx, min_z_idx};
            case Corner::TOP_LEFT_FAR:
                return {min_x_idx, max_y_idx, min_z_idx};  // Matches Python bug exactly
            case Corner::TOP_RIGHT_NEAR:
                return {max_x_idx, max_y_idx, min_z_idx};
            case Corner::TOP_RIGHT_FAR:
                return {max_x_idx, max_y_idx, max_z_idx};
            case Corner::BOTTOM_LEFT_NEAR:
                return {min_x_idx, min_y_idx, min_z_idx};
            case Corner::BOTTOM_LEFT_FAR:
                return {min_x_idx, min_y_idx, max_z_idx};
            case Corner::BOTTOM_RIGHT_NEAR:
                return {max_x_idx, min_y_idx, min_z_idx};
            case Corner::BOTTOM_RIGHT_FAR:
                return {max_x_idx, min_y_idx, max_z_idx};
        }
        return {0, 0, 0};
    }

    Points recursiveBoundingCuboid(const Points& pts, Corner corner, int depth = 0) const {
        Points localPoints;

        if (pts.size() <= MIN_POINTS || depth >= MAX_DEPTH) {
            return localPoints;
        }

        // Find extremes directly
        auto [minX_pt, maxX_pt] = std::minmax_element(pts.begin(), pts.end(),
            [](const Point3D& a, const Point3D& b) { return a[0] < b[0]; });
        auto [minY_pt, maxY_pt] = std::minmax_element(pts.begin(), pts.end(),
            [](const Point3D& a, const Point3D& b) { return a[1] < b[1]; });
        auto [minZ_pt, maxZ_pt] = std::minmax_element(pts.begin(), pts.end(),
            [](const Point3D& a, const Point3D& b) { return a[2] < b[2]; });

        // Select extremes based on corner
        Point3D selectedPts[3];
        switch (corner) {
            case Corner::TOP_LEFT_NEAR:
            case Corner::TOP_LEFT_FAR:
                selectedPts[0] = *minX_pt; selectedPts[1] = *maxY_pt; selectedPts[2] = *minZ_pt;
                break;
            case Corner::TOP_RIGHT_NEAR:
                selectedPts[0] = *maxX_pt; selectedPts[1] = *maxY_pt; selectedPts[2] = *minZ_pt;
                break;
            case Corner::TOP_RIGHT_FAR:
                selectedPts[0] = *maxX_pt; selectedPts[1] = *maxY_pt; selectedPts[2] = *maxZ_pt;
                break;
            case Corner::BOTTOM_LEFT_NEAR:
                selectedPts[0] = *minX_pt; selectedPts[1] = *minY_pt; selectedPts[2] = *minZ_pt;
                break;
            case Corner::BOTTOM_LEFT_FAR:
                selectedPts[0] = *minX_pt; selectedPts[1] = *minY_pt; selectedPts[2] = *maxZ_pt;
                break;
            case Corner::BOTTOM_RIGHT_NEAR:
                selectedPts[0] = *maxX_pt; selectedPts[1] = *minY_pt; selectedPts[2] = *minZ_pt;
                break;
            case Corner::BOTTOM_RIGHT_FAR:
                selectedPts[0] = *maxX_pt; selectedPts[1] = *minY_pt; selectedPts[2] = *maxZ_pt;
                break;
        }

        localPoints.assign(selectedPts, selectedPts + 3);

        double x_min = std::min({selectedPts[0][0], selectedPts[1][0], selectedPts[2][0]});
        double x_max = std::max({selectedPts[0][0], selectedPts[1][0], selectedPts[2][0]});
        double y_min = std::min({selectedPts[0][1], selectedPts[1][1], selectedPts[2][1]});
        double y_max = std::max({selectedPts[0][1], selectedPts[1][1], selectedPts[2][1]});
        double z_min = std::min({selectedPts[0][2], selectedPts[1][2], selectedPts[2][2]});
        double z_max = std::max({selectedPts[0][2], selectedPts[1][2], selectedPts[2][2]});

        Points insidePoints;
        insidePoints.reserve(pts.size() / 2);

        for (const auto& pt : pts) {
            if (pt[0] > x_min && pt[0] < x_max &&
                pt[1] > y_min && pt[1] < y_max &&
                pt[2] > z_min && pt[2] < z_max) {
                insidePoints.push_back(pt);
            }
        }

        auto recursiveResult = recursiveBoundingCuboid(insidePoints, corner, depth + 1);
        localPoints.insert(localPoints.end(), recursiveResult.begin(), recursiveResult.end());

        return localPoints;
    }

    Points recursiveBoundingCuboidAlt(const std::vector<double>& x, const std::vector<double>& y, const std::vector<double>& z, Corner corner, int depth = 0) const {
        Points localPoints;

        if (x.size() <= MIN_POINTS || depth >= MAX_DEPTH) {
            return localPoints;
        }

        auto extremes = getExtremePoints(x, y, z, corner);

        localPoints.push_back({x[extremes.x_idx], y[extremes.x_idx], z[extremes.x_idx]});
        localPoints.push_back({x[extremes.y_idx], y[extremes.y_idx], z[extremes.y_idx]});
        localPoints.push_back({x[extremes.z_idx], y[extremes.z_idx], z[extremes.z_idx]});

        double x_min = std::min({x[extremes.x_idx], x[extremes.y_idx], x[extremes.z_idx]});
        double x_max = std::max({x[extremes.x_idx], x[extremes.y_idx], x[extremes.z_idx]});
        double y_min = std::min({y[extremes.x_idx], y[extremes.y_idx], y[extremes.z_idx]});
        double y_max = std::max({y[extremes.x_idx], y[extremes.y_idx], y[extremes.z_idx]});
        double z_min = std::min({z[extremes.x_idx], z[extremes.y_idx], z[extremes.z_idx]});
        double z_max = std::max({z[extremes.x_idx], z[extremes.y_idx], z[extremes.z_idx]});

        std::vector<double> inside_x, inside_y, inside_z;
        inside_x.reserve(x.size());
        inside_y.reserve(y.size());
        inside_z.reserve(z.size());

        for (size_t i = 0; i < x.size(); ++i) {
            if (x[i] > x_min && x[i] < x_max &&
                y[i] > y_min && y[i] < y_max &&
                z[i] > z_min && z[i] < z_max) {
                inside_x.push_back(x[i]);
                inside_y.push_back(y[i]);
                inside_z.push_back(z[i]);
            }
        }

        auto recursiveResult = recursiveBoundingCuboidAlt(inside_x, inside_y, inside_z, corner, depth + 1);
        localPoints.insert(localPoints.end(), recursiveResult.begin(), recursiveResult.end());

        return localPoints;
    }

    Points processCornerFast(Corner corner) const {
        Points result;
        if (points.empty()) return result;

        // Single pass to find all extremes
        auto extremes = findAllExtremesFast(points);

        // Select points based on corner type - using precomputed extremes
        Point3D selectedPts[3];
        switch (corner) {
            case Corner::TOP_LEFT_NEAR:
                selectedPts[0] = extremes.minX; selectedPts[1] = extremes.maxY; selectedPts[2] = extremes.minZ;
                break;
            case Corner::TOP_LEFT_FAR:
                selectedPts[0] = extremes.minX; selectedPts[1] = extremes.maxY; selectedPts[2] = extremes.minZ;
                break;
            case Corner::TOP_RIGHT_NEAR:
                selectedPts[0] = extremes.maxX; selectedPts[1] = extremes.maxY; selectedPts[2] = extremes.minZ;
                break;
            case Corner::TOP_RIGHT_FAR:
                selectedPts[0] = extremes.maxX; selectedPts[1] = extremes.maxY; selectedPts[2] = extremes.maxZ;
                break;
            case Corner::BOTTOM_LEFT_NEAR:
                selectedPts[0] = extremes.minX; selectedPts[1] = extremes.minY; selectedPts[2] = extremes.minZ;
                break;
            case Corner::BOTTOM_LEFT_FAR:
                selectedPts[0] = extremes.minX; selectedPts[1] = extremes.minY; selectedPts[2] = extremes.maxZ;
                break;
            case Corner::BOTTOM_RIGHT_NEAR:
                selectedPts[0] = extremes.maxX; selectedPts[1] = extremes.minY; selectedPts[2] = extremes.minZ;
                break;
            case Corner::BOTTOM_RIGHT_FAR:
                selectedPts[0] = extremes.maxX; selectedPts[1] = extremes.minY; selectedPts[2] = extremes.maxZ;
                break;
        }

        result.assign(selectedPts, selectedPts + 3);

        // Early termination - limit recursion depth more aggressively
        if (points.size() < MIN_POINTS * 10) {
            return result;
        }

        // Bounding box
        double x_min = std::min({selectedPts[0][0], selectedPts[1][0], selectedPts[2][0]});
        double x_max = std::max({selectedPts[0][0], selectedPts[1][0], selectedPts[2][0]});
        double y_min = std::min({selectedPts[0][1], selectedPts[1][1], selectedPts[2][1]});
        double y_max = std::max({selectedPts[0][1], selectedPts[1][1], selectedPts[2][1]});
        double z_min = std::min({selectedPts[0][2], selectedPts[1][2], selectedPts[2][2]});
        double z_max = std::max({selectedPts[0][2], selectedPts[1][2], selectedPts[2][2]});

        // Count inside points first to avoid allocation
        size_t insideCount = 0;
        for (const auto& pt : points) {
            if (pt[0] > x_min && pt[0] < x_max &&
                pt[1] > y_min && pt[1] < y_max &&
                pt[2] > z_min && pt[2] < z_max) {
                insideCount++;
            }
        }

        // Early termination if not enough inside points
        if (insideCount <= MIN_POINTS * 2) {
            return result;
        }

        // Pre-allocate with exact size
        Points insidePoints;
        insidePoints.reserve(insideCount);

        for (const auto& pt : points) {
            if (pt[0] > x_min && pt[0] < x_max &&
                pt[1] > y_min && pt[1] < y_max &&
                pt[2] > z_min && pt[2] < z_max) {
                insidePoints.push_back(pt);
            }
        }

        // Limited recursion with early termination
        auto recursiveResult = recursiveBoundingCuboidFast(insidePoints, corner, 1, 2); // Max depth 2
        result.insert(result.end(), recursiveResult.begin(), recursiveResult.end());

        return result;
    }

    // Fast recursive function with limited depth and early termination
    Points recursiveBoundingCuboidFast(const Points& pts, Corner corner, int depth, int maxDepth) const {
        Points localPoints;

        if (pts.size() <= MIN_POINTS || depth >= maxDepth) {
            return localPoints;
        }

        // Use fast extreme finding
        auto extremes = findAllExtremesFast(pts);

        // Select extremes based on corner - same logic as before
        Point3D selectedPts[3];
        switch (corner) {
            case Corner::TOP_LEFT_NEAR:
            case Corner::TOP_LEFT_FAR:
                selectedPts[0] = extremes.minX; selectedPts[1] = extremes.maxY; selectedPts[2] = extremes.minZ;
                break;
            case Corner::TOP_RIGHT_NEAR:
                selectedPts[0] = extremes.maxX; selectedPts[1] = extremes.maxY; selectedPts[2] = extremes.minZ;
                break;
            case Corner::TOP_RIGHT_FAR:
                selectedPts[0] = extremes.maxX; selectedPts[1] = extremes.maxY; selectedPts[2] = extremes.maxZ;
                break;
            case Corner::BOTTOM_LEFT_NEAR:
                selectedPts[0] = extremes.minX; selectedPts[1] = extremes.minY; selectedPts[2] = extremes.minZ;
                break;
            case Corner::BOTTOM_LEFT_FAR:
                selectedPts[0] = extremes.minX; selectedPts[1] = extremes.minY; selectedPts[2] = extremes.maxZ;
                break;
            case Corner::BOTTOM_RIGHT_NEAR:
                selectedPts[0] = extremes.maxX; selectedPts[1] = extremes.minY; selectedPts[2] = extremes.minZ;
                break;
            case Corner::BOTTOM_RIGHT_FAR:
                selectedPts[0] = extremes.maxX; selectedPts[1] = extremes.minY; selectedPts[2] = extremes.maxZ;
                break;
        }

        localPoints.assign(selectedPts, selectedPts + 3);

        // Early size check
        if (pts.size() < MIN_POINTS * 4) {
            return localPoints;
        }

        double x_min = std::min({selectedPts[0][0], selectedPts[1][0], selectedPts[2][0]});
        double x_max = std::max({selectedPts[0][0], selectedPts[1][0], selectedPts[2][0]});
        double y_min = std::min({selectedPts[0][1], selectedPts[1][1], selectedPts[2][1]});
        double y_max = std::max({selectedPts[0][1], selectedPts[1][1], selectedPts[2][1]});
        double z_min = std::min({selectedPts[0][2], selectedPts[1][2], selectedPts[2][2]});
        double z_max = std::max({selectedPts[0][2], selectedPts[1][2], selectedPts[2][2]});

        // Count first, then allocate
        size_t insideCount = 0;
        for (const auto& pt : pts) {
            if (pt[0] > x_min && pt[0] < x_max &&
                pt[1] > y_min && pt[1] < y_max &&
                pt[2] > z_min && pt[2] < z_max) {
                insideCount++;
            }
        }

        if (insideCount <= MIN_POINTS) {
            return localPoints;
        }

        Points insidePoints;
        insidePoints.reserve(insideCount);

        for (const auto& pt : pts) {
            if (pt[0] > x_min && pt[0] < x_max &&
                pt[1] > y_min && pt[1] < y_max &&
                pt[2] > z_min && pt[2] < z_max) {
                insidePoints.push_back(pt);
            }
        }

        auto recursiveResult = recursiveBoundingCuboidFast(insidePoints, corner, depth + 1, maxDepth);
        localPoints.insert(localPoints.end(), recursiveResult.begin(), recursiveResult.end());

        return localPoints;
    }

    void generateRandomPoints() {
        points.clear();
        points.resize(N);

        const int num_threads = omp_get_max_threads();
        std::vector<std::mt19937> thread_rngs(num_threads);

        // Initialize RNGs with random seeds
        std::random_device rd;
        for (int i = 0; i < num_threads; ++i) {
            thread_rngs[i].seed(rd() + i);
        }

        #pragma omp parallel
        {
            int thread_id = omp_get_thread_num();
            std::uniform_real_distribution<double> dist(RANGE_MIN, RANGE_MAX);
            auto& local_rng = thread_rngs[thread_id];

            #pragma omp for schedule(static)
            for (int i = 0; i < N; ++i) {
                points[i] = {dist(local_rng), dist(local_rng), dist(local_rng)};
            }
        }
    }

    Points getUniquePoints(const Points& pts) const {
        if (pts.empty()) return {};

        Points unique = pts;

        // Sort points
        std::sort(std::execution::par_unseq, unique.begin(), unique.end(),
                  [](const Point3D& a, const Point3D& b) {
            if (std::abs(a[0] - b[0]) > 1e-12) return a[0] < b[0];
            if (std::abs(a[1] - b[1]) > 1e-12) return a[1] < b[1];
            return a[2] < b[2];
        });

        // Remove duplicates with tolerance
        auto it = std::unique(unique.begin(), unique.end(), [](const Point3D& a, const Point3D& b) {
            constexpr double tol = 1e-12;
            return std::abs(a[0] - b[0]) < tol &&
                   std::abs(a[1] - b[1]) < tol &&
                   std::abs(a[2] - b[2]) < tol;
        });

        unique.erase(it, unique.end());
        return unique;
    }

    std::vector<bool> filterInsidePointsAdvanced(const Points& pts,
                                                 const std::vector<std::array<double, 4>>& equations) const {
        std::vector<bool> insideMask(pts.size(), false);
        const double tolerance = TOL;

        // Process equations in groups of 4
        const size_t eq_groups = (equations.size() + 3) / 4;

        #pragma omp parallel for schedule(static, 2048)
        for (size_t i = 0; i < pts.size(); ++i) {
            const auto& pt = pts[i];

            // Broadcast point coordinates
            __m256d px = _mm256_broadcast_sd(&pt[0]);
            __m256d py = _mm256_broadcast_sd(&pt[1]);
            __m256d pz = _mm256_broadcast_sd(&pt[2]);
            __m256d tol_vec = _mm256_broadcast_sd(&tolerance);

            bool isInside = true;

            // Process equations in groups of 4
            for (size_t eq_group = 0; eq_group < eq_groups && isInside; ++eq_group) {
                size_t eq_start = eq_group * 4;
                size_t eq_end = std::min(eq_start + 4, equations.size());

                // Load 4 equations
                alignas(32) double a_vals[4] = {0, 0, 0, 0};
                alignas(32) double b_vals[4] = {0, 0, 0, 0};
                alignas(32) double c_vals[4] = {0, 0, 0, 0};
                alignas(32) double d_vals[4] = {0, 0, 0, 0};

                for (size_t j = 0; j < eq_end - eq_start; ++j) {
                    const auto& eq = equations[eq_start + j];
                    a_vals[j] = eq[0];
                    b_vals[j] = eq[1];
                    c_vals[j] = eq[2];
                    d_vals[j] = eq[3];
                }

                __m256d a = _mm256_load_pd(a_vals);
                __m256d b = _mm256_load_pd(b_vals);
                __m256d c = _mm256_load_pd(c_vals);
                __m256d d = _mm256_load_pd(d_vals);

                // Compute dot products: a*x + b*y + c*z + d
                __m256d dot = _mm256_fmadd_pd(a, px,
                             _mm256_fmadd_pd(b, py,
                             _mm256_fmadd_pd(c, pz, d)));

                // Check outside condition
                __m256d neg_tol = _mm256_sub_pd(_mm256_setzero_pd(), tol_vec);
                __m256d outside_mask = _mm256_cmp_pd(dot, neg_tol, _CMP_GE_OQ);

                // Check if any equation indicates outside
                int mask = _mm256_movemask_pd(outside_mask);

                // Check valid equations
                for (size_t j = 0; j < eq_end - eq_start; ++j) {
                    if (mask & (1 << j)) {
                        isInside = false;
                        break;
                    }
                }
            }

            insideMask[i] = isInside;
        }

        return insideMask;
    }

    std::vector<bool> filterInsidePoints(const Points& pts,
                                       const std::vector<std::array<double, 4>>& equations) const {
        return filterInsidePointsAdvanced(pts, equations);
    }

public:
    TilingV2(int numPoints) : N(numPoints), rng(std::random_device{}()) {}

    void run() {
        generateRandomPoints();

        auto start = std::chrono::high_resolution_clock::now();

        QHullWrapper originalHull;
        auto originalResult = originalHull.computeConvexHull(points);

        auto originalTime = std::chrono::high_resolution_clock::now() - start;

        std::vector<Corner> corners = {
            Corner::TOP_LEFT_NEAR, Corner::TOP_LEFT_FAR,
            Corner::TOP_RIGHT_NEAR, Corner::TOP_RIGHT_FAR,
            Corner::BOTTOM_LEFT_NEAR, Corner::BOTTOM_LEFT_FAR,
            Corner::BOTTOM_RIGHT_NEAR, Corner::BOTTOM_RIGHT_FAR
        };

        start = std::chrono::high_resolution_clock::now();

        std::vector<std::future<Points>> futures;
        futures.reserve(corners.size());

        // Enhanced corner processing with better load balancing
        const int num_threads = std::min(static_cast<int>(corners.size()),
                                        static_cast<int>(std::thread::hardware_concurrency()));

        // Cache global extremes once for all corners
        auto globalExtremes = findAllExtremesFast(points);

        for (auto corner : corners) {
            futures.push_back(std::async(std::launch::async,
                [this, corner]() { return this->processCornerFast(corner); }));
        }

        Points allExtremes;
        size_t total_size = 0;

        // Pre-calculate total size for better memory allocation
        std::vector<Points> results(futures.size());
        for (size_t i = 0; i < futures.size(); ++i) {
            results[i] = futures[i].get();
            total_size += results[i].size();
        }

        allExtremes.reserve(total_size);
        for (const auto& result : results) {
            allExtremes.insert(allExtremes.end(), result.begin(), result.end());
        }

        auto uniqueExtremes = getUniquePoints(allExtremes);
        auto step1Time = std::chrono::high_resolution_clock::now() - start;

        Points remainingPoints = points;
        double hullTime = 0.0;
        double filterTime = 0.0;

        if (uniqueExtremes.size() >= 4) {
            start = std::chrono::high_resolution_clock::now();
            QHullWrapper extremeHull;
            auto extremeResult = extremeHull.computeConvexHull(uniqueExtremes);
            hullTime = std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - start).count();

            if (extremeResult.success) {
                start = std::chrono::high_resolution_clock::now();
                auto insideMask = filterInsidePoints(points, extremeResult.equations);

                // Optimized remaining points collection with filtering
                size_t remaining_count = 0;
                #pragma omp parallel for reduction(+:remaining_count)
                for (size_t i = 0; i < insideMask.size(); ++i) {
                    if (!insideMask[i]) {
                        remaining_count++;
                    }
                }

                remainingPoints.clear();
                remainingPoints.reserve(remaining_count);

                for (size_t i = 0; i < points.size(); ++i) {
                    if (!insideMask[i]) {
                        remainingPoints.push_back(points[i]);
                    }
                }
                filterTime = std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - start).count();
            }
        }

        double totalPreprocessing = std::chrono::duration<double>(step1Time).count() + hullTime + filterTime;

        start = std::chrono::high_resolution_clock::now();
        int finalVertices = 0;
        if (remainingPoints.size() >= 4) {
            QHullWrapper finalHull;
            auto finalResult = finalHull.computeConvexHull(remainingPoints);
            if (finalResult.success) {
                finalVertices = finalResult.vertices.size();
            }
        } else {
            finalVertices = remainingPoints.size();
        }
        auto finalHullTime = std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - start).count();

        double totalOurMethod = totalPreprocessing + finalHullTime;
        double originalTimeSeconds = std::chrono::duration<double>(originalTime).count();

        std::unordered_set<std::string> hullVerticesSet;
        for (int idx : originalResult.vertices) {
            auto& pt = points[idx];
            std::string key = std::to_string(std::round(pt[0] * 1000000)) + "," +
                             std::to_string(std::round(pt[1] * 1000000)) + "," +
                             std::to_string(std::round(pt[2] * 1000000));
            hullVerticesSet.insert(key);
        }

        std::unordered_set<std::string> remainingSet;
        for (const auto& pt : remainingPoints) {
            std::string key = std::to_string(std::round(pt[0] * 1000000)) + "," +
                             std::to_string(std::round(pt[1] * 1000000)) + "," +
                             std::to_string(std::round(pt[2] * 1000000));
            remainingSet.insert(key);
        }

        int preservedHullVertices = 0;
        for (const auto& vertex : hullVerticesSet) {
            if (remainingSet.count(vertex)) {
                preservedHullVertices++;
            }
        }

        double completeness = (static_cast<double>(preservedHullVertices) / originalResult.vertices.size()) * 100.0;

        std::cout << std::fixed << std::setprecision(6);
        std::cout << "Number of vertices on convex hull (all input points): " << originalResult.vertices.size() << std::endl;
        std::cout << "Number of points preserved after preprocessing: " << remainingPoints.size() << std::endl << std::endl;

        std::cout << "--- Performance Details ---" << std::endl;
        std::cout << "Unique extremes collected: " << uniqueExtremes.size() << std::endl;
        std::cout << "Extreme collection time: " << std::chrono::duration<double>(step1Time).count() << " seconds" << std::endl;
        std::cout << "Extreme hull construction time: " << hullTime << " seconds" << std::endl;
        std::cout << "Point filtering time: " << filterTime << " seconds" << std::endl;
        std::cout << "Total preprocessing time: " << totalPreprocessing << " seconds" << std::endl;
        std::cout << "Final hull time: " << finalHullTime << " seconds" << std::endl;
        std::cout << "Total our method time: " << totalOurMethod << " seconds" << std::endl;
        std::cout << "Direct convex hull time: " << originalTimeSeconds << " seconds" << std::endl << std::endl;

        if (totalOurMethod < originalTimeSeconds) {
            double speedup = originalTimeSeconds / totalOurMethod;
            std::cout << "✅ Our method is faster: " << std::setprecision(2) << speedup << "x" << std::endl;
        } else {
            double slowdown = totalOurMethod / originalTimeSeconds;
            std::cout << "❌ Our method is slower: " << std::setprecision(2) << slowdown << "x" << std::endl;
        }

        double reduction = ((static_cast<double>(N) - remainingPoints.size()) / N) * 100.0;
        std::cout << std::endl << "📈 Point reduction: " << std::setprecision(1) << reduction
                  << "% (" << (N - remainingPoints.size()) << " removed from " << N << ")" << std::endl;
        std::cout << std::endl << "🎯 Completeness: " << std::setprecision(1) << completeness
                  << "% (" << preservedHullVertices << "/" << originalResult.vertices.size() << ")" << std::endl;
    }
};

int main() {
    try {
        int N;
        std::cout << "Enter number of points: ";
        std::cin >> N;

        if (N <= 0) {
            std::cerr << "Error: Number of points must be positive" << std::endl;
            return 1;
        }

        TilingV2 algorithm(N);
        algorithm.run();
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}