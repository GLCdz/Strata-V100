# 构建、配置与验证

本说明适用于Linux、4×V100-SXM2-16GB；原测量使用CUDA12.6、GCC13.3、EPYC7402、约252GiB RAM。不同卡数、显存、CPU拓扑或模型需重新验证。所有生成物放在自己的开发项目或部署目录。

## 1. 获取源码并构建

```bash
git clone https://github.com/GLCdz/Strata-V100.git
cd Strata-V100
cmake -S . -B build-v100 \
  -DCMAKE_BUILD_TYPE=Release \
  -DSTRATA_ENABLE_CUDA=ON \
  -DSTRATA_EXPERIMENTAL_SM60=ON \
  -DCMAKE_CUDA_ARCHITECTURES=70 \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.6/bin/nvcc \
  -DSTRATA_MMQ_TAB=ON \
  -DSTRATA_BUILD_TESTS=ON
cmake --build build-v100 --target strata prefill_mmq_tab_test -j 8
```

将nvcc路径替换成本机CUDA12.x工具链位置。V100不使用CUDA13构建。CMake会获取固定的ggml依赖 `3cf03257f219afbe7334045ff7c6a06ac68c627d`；离线时可用 `-DSTRATA_GGML_DIR=/path/to/llama.cpp-checkout`，该目录必须是同一版本并包含ggml。

找到构建出的 `strata` 和 `libstrata_kq.so`，作为一组放在自己的独立发布目录中；开发目录与运行目录分开。不要让正在运行的服务加载一个仍会被编译覆盖的文件。具体输出位置可通过：

```bash
find build-v100 -type f \( -name strata -o -name libstrata_kq.so -o -name prefill_mmq_tab_test \)
```

## 2. 先生成自己的模型配置

按上游 [AI_SETUP.md](../../docs/AI_SETUP.md)、[OLDER_GPUS.md](../../docs/OLDER_GPUS.md) 完成模型、pack、MTP、tokenizer和需要的视觉组件准备，使用上游setup的 `--no-start` 避免在已有服务旁启动第二个引擎。模型权重从上游指定来源获取，遵守各自许可证，本仓库不分发权重。

确认setup生成的基础JSON里 `exe` 指向上一步的独立引擎，`cwd`、`lib_dirs`和模型路径符合本机布局。视觉组件独立配置，下面的生成器保留它，不为未安装的组件编造路径。

```bash
python3 community/v100/tools/make_config.py \
  --base /your/deployment/base-model.json \
  --output /your/deployment/v100x4-256k.json \
  --gpus 0,1,2,3
```

替换示例路径；输出文件不能已存在。生成器保留基础配置里的模型路径、API密钥及采样设置，必要时生成密钥而不打印，文件权限0600；绑定127.0.0.1。它启用单序列profile，移除batch/parallel配置，启用fit_max_tokens（输出预留超过剩余窗口时钳制输出预算，不截断输入）。它不重启服务、不改系统功率、不分配系统大页。

`--gpus`顺序决定各stage使用哪张卡。CPU绑定是拓扑相关的：先看 `lscpu -e=CPU,CORE,SOCKET,NODE`。参考机主线程CPU0、池CPU1–23、辅助CPU25–47；只有拓扑相同时才加 `--aux-cpus 25-47`。默认不强加该范围，也不自动改变系统CPU策略。

## 3. 系统资源与首次启动

- **大页**：参考机需要23983个2MiB实际分配页，池配置24576页（48GiB）。先确保足够可用RAM，再由管理员设置 `vm.nr_hugepages=24576` 并持久化；只有池数字正确不代表进程已用到大页。启动后检查 `/proc/<engine-pid>/status` 的HugetlbPages。不要为了迁就本profile减少其他服务必需的大页。
- **内存**：专家arena约46.84GiB、PLE锁页约26.82GiB，此外还有活跃会话、工作区和按需缓存。64GiB只是缓存上限，参考机充分测试后系统约用130GiB、余约121GiB；较小内存机器不要照搬。
- **功率**：参考机每卡300W上限。先运行 `nvidia-smi --query-gpu=index,power.limit,power.max_limit --format=csv`，确认卡和供电散热条件；通过本机管理员已有功率工具设置。200/250W对照见BENCHMARKS.md。脚本不自动改功率。
- **显存**：512/1024MiB预留属于本机实测，完整KV时首卡CUDA空闲约544MiB。若专家或图不完整、出现OOM，候选不合格；应增加预留或降低KV驻留窗口重新测试，不能放宽检查假装通过。

为基础配置保留备份。等待已有请求完成，停止旧服务并确认旧引擎退出，再用新的配置启动一个实例：

```bash
python3 -m serve.server --engine strata --config /your/deployment/v100x4-256k.json
```

生产环境使用自己的systemd服务和日志目录；本仓库不附带指向维护者机器路径的服务单元。

启动后运行只读检查：

```bash
python3 community/v100/tools/check_startup.py \
  --config /your/deployment/v100x4-256k.json \
  --log /your/deployment/engine.log \
  --wait 180
```

守卫核对进程实际大页、24576专家、8个快速图、pw2、配置上下文、长度策略和48层MMQ direct。`--url`可指定本机API地址；默认127.0.0.1和配置端口。它针对本profile的启动日志格式，不是通用健康监控，也不是自动调参器。

## 4. 测试

无需GPU的工具测试（在开发目录运行，临时文件归本项目）：

```bash
mkdir -p community/v100/local/tmp
TMPDIR="$PWD/community/v100/local/tmp" \
  python3 -m unittest discover -s community/v100/tests -v
```

原部署已在V100通过MMQ pointer-table的890项比较及GR/GDN parity。复现GPU测试时先停止用户推理，避免测试与服务争抢显存；本仓库保留上游的相关测试目标，GPU测试可执行文件位置见构建输出。不要把无GPU跳过视为通过。

服务器独占、空闲时跑速度测试，先确保配置上下文至少足以容纳选择的长度与输出：

```bash
python3 community/v100/tools/bench.py \
  --config /your/deployment/v100x4-256k.json \
  --output community/v100/local/my-results.jsonl \
  --sizes 4096 16384 32768 130000 240000
```

先丢弃一次warmup；短任务各3次、新鲜输入各2次。脚本复用本报告的论文文本和提示构造，nonce会变化；实际prompt/reused/output计数、引擎计时和客户端TTFT均写入结果，不保存回答正文。若其他客户端抢入或新鲜输入命中缓存，脚本报错，该轮不应计分。已有结果文件不会覆盖。数字差异应结合输出哈希、采样和接受率解释。

本脚本覆盖基本速度测量；不自动重演9组行为测试、缓存填充、强制token边界诊断或持续负载，也不会把速度测试当作质量测试。它未在本次发布准备中额外占用生产GPU跑基准；已发布成绩来自先前明确记录的实测。

## 5. DSH与回退

在本机DSH对应模型声明中将contextWindow设为262144，输出maxTokens保持所需值（参考32768）。等待会话空闲后重载DSH。16缓存槽不提供16路同时解码，新的240K请求仍可能让其他请求等待近一分钟。

回退：等待请求结束，停新实例并确认退出，恢复自己备份的基础配置、原引擎及相应功率策略，然后启动并验证接口。若恢复128K，也把DSH模型声明恢复131072。不要使用本报告里历史机器的绝对路径作为本机部署命令。
