# CUDA_study

Here are the codes of https://zhuanlan.zhihu.com/c_1188568938097819648

```sh
nvcc -o sum_martix sum_martix.cu
./sum_martix
nvprof --print-gpu-trace ./sum_martix
```
