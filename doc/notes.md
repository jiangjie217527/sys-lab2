# 实验一内容

SIZE=256 512 1024 4096

thread per block分配方式取默认32

记录时间

数据大小为rand%5

cmd:nvprof --print-gpu-trace xxx
//时间：910us 7.53ms 50.1ms 2.788s

cmd:nvprof xxx
(快3-4倍的原因是每次循环都要计算下标改成了暂时存储在中间变量里)
256:223.61us
512:1.7571ms
1024:11.388ms
4096:673.90ms

SIZE=4096(THREAD_PER_BLOCK=16)
4096:359.51ms

SIZE=4096(THREAD_PER_BLOCK=8)
4096:188.27ms

# 实验二内容

SIZE=256 512 1024 4096(thread_per_block=32)
256:107.2us
512:707.23us
1024:4.6932ms
4096:278.40ms

SIZE=4096(THREAD_PER_BLOCK=16)
4096:178.69ms

SIZE=4096(THREAD_PER_BLOCK=8)
4096:132.65ms
SIZE=4096(THREAD_PER_BLOCK=8)
4096:292.68ms
