#!/bin/bash
# 批量顺序跑多个训练配置（单卡，逐个跑完再跑下一个，避免并发抢 DataLoader
# worker 的文件描述符资源，见 docs/REPRODUCTION_LOG.md 里 rate_0.3 崩溃的教训）。
# 路径是这台 AutoDL 服务器的实际部署路径，换机器需要相应调整。
cd /root/autodl-tmp/ANGA
PY=/root/autodl-tmp/ANGA_venv/bin/python
LOGDIR=/root/autodl-tmp/batch_logs
mkdir -p $LOGDIR results_archive

run_one() {
  name=$1; shift
  echo "=== RUN START: $name $(date) ==="
  rm -f src/checkpoints/best_model.pth
  rm -f src/metrics/*.csv   # 清空历史 csv，确保下面只会有本次跑产生的这一个
  $PY src/train.py --dataset hatememes "$@" > $LOGDIR/${name}.log 2>&1
  mkdir -p results_archive/$name
  mv src/checkpoints/best_model.pth results_archive/$name/ 2>/dev/null
  mv src/metrics/*.csv results_archive/$name/metrics.csv 2>/dev/null
  cp $LOGDIR/${name}.log results_archive/$name/train.log
  echo "=== RUN END: $name $(date) ==="
}

run_one ablation_000 --missing_type Text --missing_rate 0.7 --no-use_mir --no-use_ga --no-use_sea
run_one ablation_100 --missing_type Text --missing_rate 0.7 --no-use_ga --no-use_sea
run_one ablation_110 --missing_type Text --missing_rate 0.7 --no-use_sea
run_one ablation_101 --missing_type Text --missing_rate 0.7 --no-use_ga
run_one image_0.7 --missing_type Image --missing_rate 0.7
run_one rate_0.1 --missing_type Text --missing_rate 0.1
run_one rate_0.3 --missing_type Text --missing_rate 0.3
run_one rate_0.5 --missing_type Text --missing_rate 0.5
run_one rate_0.9 --missing_type Text --missing_rate 0.9
run_one k_1 --missing_type Text --missing_rate 0.7 --k 1
run_one k_3 --missing_type Text --missing_rate 0.7 --k 3
run_one k_7 --missing_type Text --missing_rate 0.7 --k 7
run_one k_9 --missing_type Text --missing_rate 0.7 --k 9

echo ALL_BATCH_DONE
