### Tiling preprocessing algorithm
##Running the Code

Install the qhull library and then run

`g++ -mavx -mfma -O3 -std=c++17 -fopenmp -I/usr/include/libqhull_r -o out ./tiling-3D.cpp -lqhull_r -lm`
