# MPI 异步通信小作业 实验报告 

管思源 2021012702

## 任务一

InfiniBand:


| 编号 | 消息长度 | 延迟 (us) | 带宽(MB/s) |
| ---- | -------- | ------ | ------ |
| 1    | 1        | 1.31 | 4.14 |
| 2    | 2        | 1.26 | 8.42 |
| 3    | 4        | 1.22 | 17.00 |
| 4    | 8        | 1.21 | 33.67 |
| 5    | 16       | 1.21 | 67.97 |
| 6    | 32       | 1.29 | 134.75 |
| 7    | 64       | 1.42 | 235.52 |
| 8    | 128      | 1.53 | 356.32 |
| 9    | 256      | 2.19 | 516.09 |
| 10   | 512      | 2.25 | 1189.39 |
| 11   | 1024     | 2.53 | 1840.10 |
| 12   | 2048     | 3.05 | 2844.54 |
| 13   | 4096     | 4.09 | 3924.90 |
| 14   | 8192     | 5.88 | 4902.90 |
| 15   | 16384    | 7.82 | 5105.94 |
| 16   | 32768    | 10.93 | 10900.69 |
| 17   | 65536    | 18.05 | 11568.74 |
| 18   | 131072   | 33.04 | 11795.97 |
| 19   | 262144   | 33.13     | 11925.50 |
| 20   | 524288   | 58.21 | 12010.54 |
| 21   | 1048576  | 100.02 | 12039.19 |
| 22   | 2097152  | 187.22 | 12047.85 |
| 23   | 4194304  | 363.16 | 12055.85 |









以太网：


| 编号 | 消息长度 | 延迟 (us) | 带宽(MB/s) |
| ---- | -------- | ------ | ------ |
| 1    | 1        | 47.47 | 0.29 |
| 2    | 2        | 47.04 | 0.60 |
| 3    | 4        | 45.03 | 1.19 |
| 4    | 8        | 47.24 | 2.31 |
| 5    | 16       | 47.79 | 4.80 |
| 6    | 32       | 47.80 | 8.75 |
| 7    | 64       | 48.77 | 15.85 |
| 8    | 128      | 50.34 | 32.72 |
| 9    | 256      | 55.25 | 53.22 |
| 10   | 512      | 61.20 | 79.27 |
| 11   | 1024     | 73.26 | 94.86 |
| 12   | 2048     | 90.77 | 104.54 |
| 13   | 4096     | 113.30 | 110.50 |
| 14   | 8192     | 192.49 | 113.96 |
| 15   | 16384    | 220.64 | 115.95 |
| 16   | 32768    | 366.06 | 116.63 |
| 17   | 65536    | 639.77 | 117.09 |
| 18   | 131072   | 1203.12 | 117.36 |
| 19   | 262144   | 2424.31 | 117.51 |
| 20   | 524288   | 4655.50 | 117.56 |
| 21   | 1048576  | 9100.34 | 117.60 |

（为了更好地读出趋势，作图如下）

![result](C:\Users\Frank\Projects\hpc-lab\exp1\result.png)


- 请描述当消息长度增加时，带宽和延迟分别呈现出什么样的趋势？

答：当消息长度增加时，延迟先稳定在某一值附近、后随消息长度加速增加；带宽先随消息长度近线性增加，后稳定在某一值附近。

- 该趋势在两种网络下有何不同？

答：(1) 两种网络的性能指标绝对值有差异 (2) 以太网较InfiniBand在延迟上更晚开始增加，在带宽上更早达到饱和。

- 为什么会有这样的趋势？

答：在传输小信息时，延迟主要体现在网络的固有通信时延、带宽不是限制因素；在传输大信息时，带宽使用饱和，信息的有限传输速度贡献了延迟。

- 对比InfiniBand和以太网络下的带宽和延迟，它们之间的差距是多少？

答：以太网的延迟为InfiniBand的30倍左右，InfiniBand的带宽约为以太网的100倍。

## 任务二

| 编号 | 消息长度  | 计算量 | mpi_sync 总耗时 | mpi_async  总耗时 |
| ---- | --------- | ------ | --------------- | ----------------- |
| 1    | 100000000 | 10     | 968.685         | 705.08            |
| 2    | 100000000 | 20     | 1072.49         | 867.214           |
| 3    | 100000000 | 40     | 1276.34         | 865.51            |
| 4    | 100000000 | 80     | 1514.44         | 800.227           |
| 5    | 100000000 | 160    | 2483.75         | 1600.38           |

- 通信时间和计算时间满足什么关系时，非阻塞通信程序能完美掩盖通信时间？

答：当计算时间大于通信时间时。

- 简述两份代码的不同之处。

答：`mpi_async.cpp`用到了`MPI_Isend`和`MPI_Wait`这两个非阻塞通信函数来替代`mpi_sync.cpp`中的`MPI_Send`。

## 附录（助教批阅时不用管，仅作记录用）

附：作图的`Matlab`代码

```matlab
inf_id = 1:23;
inf_delay = [1.31, 1.26, 1.22, 1.21, 1.21, 1.29, 1.42, 1.53, 2.19, 2.25, 2.53, 3.05, 4.09, 5.88, 7.82, 10.93, 18.05, 33.04, 33.13, 58.21, 100.02, 187.22, 363.16];
inf_band = [4.14, 8.42, 17.00, 33.67, 67.97, 134.75, 235.52, 356.32, 516.09, 1189.39, 1840.10, 2844.54, 3924.90, 4902.90, 5105.94, 10900.69, 11568.74, 11795.97, 11925.50, 12010.54, 12039.19, 12047.85, 12055.85];
ether_id = 1:21;
ether_delay = [47.47, 47.04, 45.03, 47.24, 47.79, 47.80, 48.77, 50.34, 55.25, 61.20, 73.26, 90.77, 113.30, 192.49, 220.64, 366.06, 639.77, 1203.12, 2424.31, 4655.50, 9100.34];
ether_band = [0.29, 0.60, 1.19, 2.31, 4.80, 8.75, 15.85, 32.72, 53.22, 79.27, 94.86, 104.54, 110.50, 113.96, 115.95, 116.63, 117.09, 117.36, 117.51, 117.56, 117.60];
figure;
subplot(3,3,1),
plot(inf_id, log2(inf_delay)),
ylabel("延迟（log_2 us）");
title("InfiniBand");
subplot(3,3,4),
plot(inf_id, log2(inf_band)),
ylabel("带宽（log_2 MB/s）");
subplot(3,3,7),
plot(inf_id, log2(inf_band .* inf_delay)),
ylabel("延迟×带宽（log_2 bytes）");
xlabel("消息长度（log_2 bits）");
subplot(3,3,2),
plot(ether_id, log2(ether_delay)),
title("Ethernet");
subplot(3,3,5),
plot(ether_id, log2(ether_band)),
subplot(3,3,8),
plot(ether_id, log2(ether_delay .* ether_band)),
xlabel("消息长度（log_2 bits）");
subplot(3,3,3),
plot(inf_id, log2(inf_delay), ether_id, log2(ether_delay)),
legend("InfiniBand","Ethernet");
title("对比");
subplot(3,3,6),
plot(inf_id, log2(inf_band), ether_id, log2(ether_band)),
legend("InfiniBand","Ethernet");
subplot(3,3,9),
plot(inf_id, log2(inf_band .* inf_delay), ether_id, log2(ether_delay .* ether_band)),
legend("InfiniBand","Ethernet");
xlabel("消息长度（log_2 bits）");
```

附：任务二计算的输出

```bash
+ srun -N 2 -n 2 ./mpi_sync 10 100000000 10
Iter 0 MPI_Send: 86.5957 ms
Iter 0 Compute: 10.0001 ms
Iter 1 MPI_Send: 86.8895 ms
Iter 1 Compute: 10.0001 ms
Iter 2 MPI_Send: 86.9107 ms
Iter 2 Compute: 10.0001 ms
Iter 3 MPI_Send: 86.8681 ms
Iter 3 Compute: 10.0001 ms
Iter 4 MPI_Send: 86.8424 ms
Iter 4 Compute: 10.0001 ms
Iter 5 MPI_Send: 86.923 ms
Iter 5 Compute: 10.0001 ms
Iter 6 MPI_Send: 86.8462 ms
Iter 6 Compute: 10.0001 ms
Iter 7 MPI_Send: 86.8845 ms
Iter 7 Compute: 10.0001 ms
Iter 8 MPI_Send: 86.9047 ms
Iter 8 Compute: 10.0001 ms
Iter 9 MPI_Send: 86.842 ms
Iter 9 Compute: 10.0001 ms
Total: 968.685 ms
+ srun -N 2 -n 2 ./mpi_async 10 100000000 10
Iter 0 MPI_Isend: 0.050595 ms
Iter 0 Compute: 10.0651 ms
Iter 1 MPI_Isend: 0.001297 ms
Iter 1 Compute: 10.0024 ms
Iter 2 MPI_Isend: 0.000398 ms
Iter 2 Compute: 10.0016 ms
Iter 3 MPI_Isend: 0.000392 ms
Iter 3 Compute: 10.0016 ms
Iter 4 MPI_Isend: 0.000426 ms
Iter 4 Compute: 10.0013 ms
Iter 5 MPI_Isend: 0.000435 ms
Iter 5 Compute: 10.0013 ms
Iter 6 MPI_Isend: 0.000412 ms
Iter 6 Compute: 10.0014 ms
Iter 7 MPI_Isend: 0.000402 ms
Iter 7 Compute: 10.0013 ms
Iter 8 MPI_Isend: 0.000419 ms
Iter 8 Compute: 10.0013 ms
Iter 9 MPI_Isend: 0.000413 ms
Iter 9 Compute: 10.0013 ms
Wait Request: 604.909 ms
Total: 705.08 ms
+ srun -N 2 -n 2 ./mpi_sync 10 100000000 20
Iter 0 MPI_Send: 86.8646 ms
Iter 0 Compute: 20.0002 ms
Iter 1 MPI_Send: 87.2777 ms
Iter 1 Compute: 20.0001 ms
Iter 2 MPI_Send: 87.2967 ms
Iter 2 Compute: 20.0001 ms
Iter 3 MPI_Send: 87.2407 ms
Iter 3 Compute: 20.0001 ms
Iter 4 MPI_Send: 87.3093 ms
Iter 4 Compute: 20.0001 ms
Iter 5 MPI_Send: 87.2538 ms
Iter 5 Compute: 20.0001 ms
Iter 6 MPI_Send: 87.2833 ms
Iter 6 Compute: 20.0001 ms
Iter 7 MPI_Send: 87.2531 ms
Iter 7 Compute: 20.0001 ms
Iter 8 MPI_Send: 87.2439 ms
Iter 8 Compute: 20.0001 ms
Iter 9 MPI_Send: 87.2944 ms
Iter 9 Compute: 20.0001 ms
Total: 1072.49 ms
+ srun -N 2 -n 2 ./mpi_async 10 100000000 20
Iter 0 MPI_Isend: 0.057923 ms
Iter 0 Compute: 20.0708 ms
Iter 1 MPI_Isend: 0.001453 ms
Iter 1 Compute: 20.0023 ms
Iter 2 MPI_Isend: 0.000417 ms
Iter 2 Compute: 20.0017 ms
Iter 3 MPI_Isend: 0.000625 ms
Iter 3 Compute: 20.0014 ms
Iter 4 MPI_Isend: 0.00054 ms
Iter 4 Compute: 20.0015 ms
Iter 5 MPI_Isend: 0.000475 ms
Iter 5 Compute: 20.0014 ms
Iter 6 MPI_Isend: 0.000455 ms
Iter 6 Compute: 20.0015 ms
Iter 7 MPI_Isend: 0.000425 ms
Iter 7 Compute: 20.0013 ms
Iter 8 MPI_Isend: 0.000446 ms
Iter 8 Compute: 20.0014 ms
Iter 9 MPI_Isend: 0.023495 ms
Iter 9 Compute: 20.0025 ms
Wait Request: 666.993 ms
Total: 867.214 ms
+ srun -N 2 -n 2 ./mpi_sync 10 100000000 40
Iter 0 MPI_Send: 86.8358 ms
Iter 0 Compute: 40.0001 ms
Iter 1 MPI_Send: 87.7685 ms
Iter 1 Compute: 40.0001 ms
Iter 2 MPI_Send: 87.7255 ms
Iter 2 Compute: 40.0001 ms
Iter 3 MPI_Send: 87.7404 ms
Iter 3 Compute: 40.0001 ms
Iter 4 MPI_Send: 87.4848 ms
Iter 4 Compute: 40.0001 ms
Iter 5 MPI_Send: 87.7293 ms
Iter 5 Compute: 40.0001 ms
Iter 6 MPI_Send: 87.7373 ms
Iter 6 Compute: 40.0001 ms
Iter 7 MPI_Send: 87.665 ms
Iter 7 Compute: 40.0001 ms
Iter 8 MPI_Send: 87.7774 ms
Iter 8 Compute: 40.0001 ms
Iter 9 MPI_Send: 87.7026 ms
Iter 9 Compute: 40.0001 ms
Total: 1276.34 ms
+ srun -N 2 -n 2 ./mpi_async 10 100000000 40
Iter 0 MPI_Isend: 0.048409 ms
Iter 0 Compute: 40.073 ms
Iter 1 MPI_Isend: 0.00127 ms
Iter 1 Compute: 40.0023 ms
Iter 2 MPI_Isend: 0.000382 ms
Iter 2 Compute: 40.0018 ms
Iter 3 MPI_Isend: 0.000441 ms
Iter 3 Compute: 40.0015 ms
Iter 4 MPI_Isend: 0.000477 ms
Iter 4 Compute: 40.0014 ms
Iter 5 MPI_Isend: 0.034657 ms
Iter 5 Compute: 40.0024 ms
Iter 6 MPI_Isend: 0.000672 ms
Iter 6 Compute: 40.0015 ms
Iter 7 MPI_Isend: 0.000412 ms
Iter 7 Compute: 40.0014 ms
Iter 8 MPI_Isend: 0.000433 ms
Iter 8 Compute: 40.0014 ms
Iter 9 MPI_Isend: 0.000435 ms
Iter 9 Compute: 40.0014 ms
Wait Request: 465.296 ms
Total: 865.51 ms
+ srun -N 2 -n 2 ./mpi_sync 10 100000000 80
Iter 0 MPI_Send: 70.5019 ms
Iter 0 Compute: 80.0001 ms
Iter 1 MPI_Send: 71.4719 ms
Iter 1 Compute: 80.0001 ms
Iter 2 MPI_Send: 71.5578 ms
Iter 2 Compute: 80.0001 ms
Iter 3 MPI_Send: 71.5754 ms
Iter 3 Compute: 80.0001 ms
Iter 4 MPI_Send: 71.5798 ms
Iter 4 Compute: 80.0001 ms
Iter 5 MPI_Send: 71.5219 ms
Iter 5 Compute: 80.0001 ms
Iter 6 MPI_Send: 71.5318 ms
Iter 6 Compute: 80.0001 ms
Iter 7 MPI_Send: 71.5097 ms
Iter 7 Compute: 80.0001 ms
Iter 8 MPI_Send: 71.4408 ms
Iter 8 Compute: 80.0001 ms
Iter 9 MPI_Send: 71.5486 ms
Iter 9 Compute: 80.0001 ms
Total: 1514.44 ms
+ srun -N 2 -n 2 ./mpi_async 10 100000000 80
Iter 0 MPI_Isend: 0.04745 ms
Iter 0 Compute: 80.0604 ms
Iter 1 MPI_Isend: 0.001303 ms
Iter 1 Compute: 80.0027 ms
Iter 2 MPI_Isend: 0.000509 ms
Iter 2 Compute: 80.0017 ms
Iter 3 MPI_Isend: 0.023062 ms
Iter 3 Compute: 80.0023 ms
Iter 4 MPI_Isend: 0.000567 ms
Iter 4 Compute: 80.0015 ms
Iter 5 MPI_Isend: 0.00039 ms
Iter 5 Compute: 80.0015 ms
Iter 6 MPI_Isend: 0.001035 ms
Iter 6 Compute: 80.0016 ms
Iter 7 MPI_Isend: 0.000465 ms
Iter 7 Compute: 80.0016 ms
Iter 8 MPI_Isend: 0.000507 ms
Iter 8 Compute: 80.0014 ms
Iter 9 MPI_Isend: 0.00056 ms
Iter 9 Compute: 80.0015 ms
Wait Request: 0.027979 ms
Total: 800.227 ms
+ srun -N 2 -n 2 ./mpi_sync 10 100000000 160
Iter 0 MPI_Send: 87.0127 ms
Iter 0 Compute: 160 ms
Iter 1 MPI_Send: 88.5872 ms
Iter 1 Compute: 160 ms
Iter 2 MPI_Send: 88.5053 ms
Iter 2 Compute: 160 ms
Iter 3 MPI_Send: 88.5757 ms
Iter 3 Compute: 160 ms
Iter 4 MPI_Send: 88.5961 ms
Iter 4 Compute: 160 ms
Iter 5 MPI_Send: 88.5934 ms
Iter 5 Compute: 160 ms
Iter 6 MPI_Send: 88.5779 ms
Iter 6 Compute: 160 ms
Iter 7 MPI_Send: 88.0209 ms
Iter 7 Compute: 160 ms
Iter 8 MPI_Send: 88.5612 ms
Iter 8 Compute: 160 ms
Iter 9 MPI_Send: 88.5188 ms
Iter 9 Compute: 160 ms
Total: 2483.75 ms
+ srun -N 2 -n 2 ./mpi_async 10 100000000 160
Iter 0 MPI_Isend: 0.048072 ms
Iter 0 Compute: 160.068 ms
Iter 1 MPI_Isend: 0.001393 ms
Iter 1 Compute: 160.002 ms
Iter 2 MPI_Isend: 0.034128 ms
Iter 2 Compute: 160.002 ms
Iter 3 MPI_Isend: 0.001235 ms
Iter 3 Compute: 160.002 ms
Iter 4 MPI_Isend: 0.000547 ms
Iter 4 Compute: 160.002 ms
Iter 5 MPI_Isend: 0.001065 ms
Iter 5 Compute: 160.002 ms
Iter 6 MPI_Isend: 0.000494 ms
Iter 6 Compute: 160.001 ms
Iter 7 MPI_Isend: 0.070827 ms
Iter 7 Compute: 160.002 ms
Iter 8 MPI_Isend: 0.000655 ms
Iter 8 Compute: 160.001 ms
Iter 9 MPI_Isend: 0.021875 ms
Iter 9 Compute: 160.002 ms
Wait Request: 0.04847 ms
Total: 1600.38 ms
```

