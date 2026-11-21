#!/bin/bash
# ============================================================================
# deploy-nodes.sh —— 节点/算子一键部署 + 版本自证（2026-11-16）
#
# 为什么需要它（雪花事故复盘）：
#   代码生效要跨四层，任一层没同步就**静默**跑旧代码：
#     ① _common/server_ops/<fam>.py     族源
#     ② <node>/server_op.py             节点包（sync-common 分发+内联+盖戳）
#     ③ Studio 节点库                    node import 上传
#     ④ 服务端 plugins.d/<fam>_<ver>.py  ensure_plugin 运行时上传
#   事故根因：③ 漏了部分节点（它们执行时把旧版重新注册回来）+ ④ 残留旧族文件
#   （服务端按文件名排序加载、后加载覆盖新版）→ 连续十几轮真机测试跑的都是旧
#   实现，日志只有一行 @旧hash，极易被当成"旧日志残留"忽略。
#
# 本脚本把四层串起来，最后**回查服务端 /ops 的真实版本戳**自证：
#   不一致 → 非零退出 + 红字，绝不静默交付。
#
# 用法：
#   ./_tools/deploy-nodes.sh wan
#   ./_tools/deploy-nodes.sh wan-animate2 wan-pipe
#   DEPLOY_REMOTE=1 ./_tools/deploy-nodes.sh wan      # 同时推服务端+重启+自证
#
# 环境变量：
#   LOCAL=1                 只做本地（sync + import），不碰远端（默认 1）
#   DEPLOY_REMOTE=1         额外做：直推插件 → 清旧族文件 → 重启 → /ops 自证
#   SSH_HOST=root@1.2.3.4   远端主机（DEPLOY_REMOTE=1 时必填）
#   SSH_PORT=23             ssh 端口
#   SSH_PW=xxx              ssh 密码（走 sshpass；不填则用密钥）
#   SVC=http://host:8889    推理服务（自证用）
#   TOKEN=xxx               推理服务 token
# ============================================================================
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"
cd "$ROOT" || exit 1

C_R=$'\033[31m'; C_G=$'\033[32m'; C_Y=$'\033[33m'; C_0=$'\033[0m'
step() { printf '%s== %s ==%s\n' "$C_Y" "$*" "$C_0"; }
ok()   { printf '%s  ✓ %s%s\n' "$C_G" "$*" "$C_0"; }
bad()  { printf '%s  ✗ %s%s\n' "$C_R" "$*" "$C_0"; }
die()  { bad "$*"; exit 1; }

PLUGINS_DIR_REMOTE="${PLUGINS_DIR_REMOTE:-/var/lib/flowx-inference/models/plugins.d}"

family_nodes() {
  case "$1" in
    wan) echo "wan-animate wan-animate2 wan-gen wan-image-encoder-load wan-pipe \
wan-te-load wan-text-encode wan-transformer-load wan-vace wan-vae-load" ;;
    video_mask)    echo "video-person-mask" ;;
    qwen_image)    echo "qwen-image-transformer-load qwen-image-te-load \
qwen-image-vae-load qwen-image-pipe qwen-image-gen" ;;
    video_enhance) echo "video-enhance" ;;
    video_flow)    echo "video-control-preprocess" ;;
    preprocess)    echo "preprocess-openpose preprocess-depth preprocess-canny" ;;
    flux)          echo "flux-unet-load flux-dual-clip-load flux-clip-encode \
flux-empty-latent flux-sampler flux-detail-refine" ;;
    brushnet)      echo "brushnet-load brushnet-apply ksampler-brushnet \
clip-text-encode-task" ;;
    *) echo "" ;;
  esac
}

FAMILIES=(); NODES=()
for a in "$@"; do
  case "$a" in
    wan|video_mask|video_enhance|video_flow|preprocess|flux|brushnet|qwen_image) FAMILIES+=("$a") ;;
    *) NODES+=("$a") ;;
  esac
done
[[ ${#FAMILIES[@]} -eq 0 && ${#NODES[@]} -eq 0 ]] && {
  echo "用法: $0 <族名|节点目录>..."
  echo "族: wan video_mask video_enhance video_flow preprocess flux brushnet qwen_image"; exit 1; }

for f in ${FAMILIES[@]+"${FAMILIES[@]}"}; do
  for n in $(family_nodes "$f"); do NODES+=("$n"); done
done
mapfile -t NODES < <(printf '%s\n' ${NODES[@]+"${NODES[@]}"} | awk 'NF' | sort -u)
[[ ${#NODES[@]} -eq 0 ]] && die "没有解析到节点"

step "目标节点（${#NODES[@]}）: ${NODES[*]}"

# ---------- 第①②层 ----------
step "①② sync-common（族源 → 节点包，内联依赖 + 盖版本戳）"
sync_out="$(python3 _tools/sync-common.py 2>&1)" || die "sync-common 失败：
$sync_out"
printf '%s\n' "$sync_out" | tail -2

declare -A VER=() FAMOF=()
for n in "${NODES[@]}"; do
  [[ -f "$n/server_op.py" ]] || { bad "$n/server_op.py 不存在"; continue; }
  VER[$n]="$(grep -oE 'OP_VERSION = "[0-9a-f]+"' "$n/server_op.py" | head -1 \
             | grep -oE '[0-9a-f]{12}')"
  FAMOF[$n]="$(grep -oE 'OP_FAMILY = "[a-z_]+"' "$n/server_op.py" | head -1 \
               | grep -oE '"[a-z_]+"' | tr -d '"')"
  [[ -z "${VER[$n]:-}" ]] && bad "$n 缺 OP_VERSION（F1 合同）"
done

# ---------- 第③层 ----------
step "③ node import → Studio 节点库"
fail=0
for n in "${NODES[@]}"; do
  [[ -d "$n" ]] || { bad "$n 目录不存在"; fail=1; continue; }
  out="$(flowx-studio node import --type folder --path "$ROOT/$n" --overwrite 2>&1 | tail -1)"
  if grep -q "Imported" <<<"$out"; then
    ok "$(printf '%-24s' "$n") $(grep -oE 'id=[0-9]+' <<<"$out")  ver=${VER[$n]:-?}"
  else
    bad "$(printf '%-24s' "$n") import 失败: $out"; fail=1
  fi
done
[[ $fail -eq 1 ]] && die "有节点导入失败，中止（否则会继续跑旧代码）"

# ---------- 第④层 ----------
if [[ "${DEPLOY_REMOTE:-0}" == "1" ]]; then
  [[ -z "${SSH_HOST:-}" ]] && die "DEPLOY_REMOTE=1 需要 SSH_HOST"
  step "④ 服务端推送 + 清理旧族 + 重启"
  SSH=(ssh -o StrictHostKeyChecking=no -o ConnectTimeout=20)
  SCP=(scp -o StrictHostKeyChecking=no)
  PORT_ARGS=()
  [[ -n "${SSH_PORT:-}" ]] && PORT_ARGS=(-P "${SSH_PORT}") && SSH+=(-p "${SSH_PORT}")
  if [[ -n "${SSH_PW:-}" ]]; then
    SSH=(sshpass -p "$SSH_PW" "${SSH[@]}")
    SCP=(sshpass -p "$SSH_PW" "${SCP[@]}")
  fi

  # 每个族只推一个代表文件，落成 <family>.py
  declare -A REP=()
  for n in "${NODES[@]}"; do
    fam="${FAMOF[$n]:-}"; [[ -z "$fam" ]] && continue
    [[ -n "${REP[$fam]:-}" ]] && continue
    REP[$fam]="$n"
  done
  # **先清空同族文件再推送**（2026-11-16）：服务端残留 <fam>.py 时，
  # 新推 <fam>_<ver>.py 会因"跨文件重名同算子"被拒 409（detect_conflicts）。
  # 顺序必须是：删净 → 推新 → 重启。
  FAMS_PRE="$(printf '%s ' "${!REP[@]}")"
  "${SSH[@]}" "for f in $FAMS_PRE; do rm -f ${PLUGINS_DIR_REMOTE}/\${f}.py ${PLUGINS_DIR_REMOTE}/\${f}_*.py; done" \
    >/dev/null 2>&1 && ok "已清空同族旧文件（$FAMS_PRE）"
  for fam in "${!REP[@]}"; do
    src="${REP[$fam]}/server_op.py"
    if "${SCP[@]}" "${PORT_ARGS[@]}" "$src" \
         "${SSH_HOST}:${PLUGINS_DIR_REMOTE}/${fam}_${VER[${REP[$fam]}]}.py" >/dev/null 2>&1; then
      ok "$(printf '%-24s' "${fam}_${VER[${REP[$fam]}]}.py") ← ${src}"
    else
      bad "$fam.py 推送失败"
    fi
  done

  FAMS="$(printf '%s ' "${!REP[@]}")"
  # 清同族旧文件（<fam>_*.py），重启，打印实际加载的版本
  LOADED="$("${SSH[@]}" "
    for f in $FAMS; do rm -f ${PLUGINS_DIR_REMOTE}/\${f}.py; done
    supervisorctl restart inference >/dev/null 2>&1
    sleep 15
    tail -300 /var/lib/flowx-inference/inference.log" 2>/dev/null)"
  printf '%s\n' "$LOADED" | grep -E '\[plugins\]' | tail -6

  # ---------- 自证 ----------
  if [[ -n "${SVC:-}" && -n "${TOKEN:-}" ]]; then
    step "自证：/ops 实际版本 vs 节点包"
    OPS="$(curl -s -m15 -H "Authorization: Bearer $TOKEN" "$SVC/ops")"
    fail=0
    for fam in "${!REP[@]}"; do
      want="${VER[${REP[$fam]}]}"
      got="$(printf '%s' "$OPS" | python3 -c "
import json,sys
d=json.load(sys.stdin); ops=d.get('ops',[])
fam=sys.argv[1]
vs=sorted({o['name'].split('@')[-1] for o in ops
           if o['name'].startswith(fam+'.') and '@' in o['name']})
print(vs[-1] if vs else '')
" "$fam" 2>/dev/null)"
      if [[ "$got" == "$want" ]]; then
        ok "$(printf '%-16s' "$fam") version=$got"
      else
        bad "$(printf '%-16s' "$fam") 期望 $want 实际 ${got:-<无>}"; fail=1
      fi
      # **算子数量校验（2026-11-16）**：装饰器/缩进被破坏时（例如在类定义
      # 中间插入模块级函数），部分算子会静默漏注册——服务端只剩 8 个而本地
      # 声明 10 个，节点执行时报"算子未注册"，或更糟：拿到旧版算子跑出废片。
      want_n="$(python3 - "$fam" <<'PY2'
import ast,sys
fam=sys.argv[1]
try:
    tree=ast.parse(open(f"_common/server_ops/{fam}.py",encoding="utf-8").read())
except Exception:
    print(-1); raise SystemExit
ns=[n.args[0].value for n in ast.walk(tree)
    if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute)
    and n.func.attr=="register" and n.args
    and isinstance(n.args[0],ast.Constant)]
print(len(set(ns)))
PY2
)"
      got_n="$(printf '%s' "$OPS" | python3 -c "
import json,sys
d=json.load(sys.stdin); fam=sys.argv[1]
print(len({o['name'].split('@')[0] for o in d.get('ops',[])
            if o['name'].startswith(fam+'.')}))" "$fam" 2>/dev/null)"
      if [[ "${want_n:-0}" -gt 0 && "${got_n:-0}" == "${want_n}" ]]; then
        ok "$(printf '%-16s' "$fam") 算子数=$got_n"
      else
        bad "$(printf '%-16s' "$fam") 算子数 期望${want_n} 实际${got_n:-0} —— 有算子漏注册！"
        fail=1
      fi
    done
    [[ $fail -eq 1 ]] && die "版本自证未通过 —— 服务端仍在跑别的代码，别跑真机！"
  else
    printf '%s  （未设 SVC/TOKEN，跳过 /ops 自证）%s\n' "$C_Y" "$C_0"
  fi
fi

step "完成 —— 真机跑之前确认节点日志里的 @版本戳 与上面一致"

# ============================================================================
# 纪律声明（2026-11-16 用户确立，最高优先级）
# ============================================================================
# 1. 【节点/插件代码绝不 ssh 上传】plugins.d 只能由节点运行时
#    ensure_plugin 自注册（本脚本负责放入 Studio 节点库，流水线触发）。
#    scp 直传插件 = 事故根源（无版本戳文件与自注册文件并存互相覆盖）。
# 2. 【服务端框架代码（inference-server/app）可 ssh 上传】但必须：
#    本地先 git commit + 逐文件 AST 校验通过；上传后 restart inference。
#    git pull 优先，网络不通时走此通道。
# 2. 【所有测试走流水线】不用 curl 直接打 /op 接口做游离测试；真机验证
#    = workflow run，产物/日志/版本全部可追溯。
# 3. 【推理服务只走 git 更新】服务端代码（inference-server）更新 =
#    git pull + supervisorctl restart。禁止推单文件补丁（plugins.py 的
#    去重加固就是这么被推丢的）。
# 4. 【服务端拒绝非节点上传】见 app/plugins.py 的 upload_plugin 守卫：
#    只接受带合法版本戳的 <family>_<hash12>.py，其余 403。
# ============================================================================
