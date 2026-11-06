/* pose-gate 节点 widget：骨架人工确认/修正编辑器（契约 apiVersion 1）。
 *
 * 节点卡片内只放状态行 + 小预览 +「打开骨架编辑器」按钮；点击弹出全屏
 * 浮层（position:fixed，与画布手势完全隔离），在浮层里拖拽关键点修正骨架，
 * 「确认放行」后经节点级 service-proxy 提交决策，服务端重渲染骨架继续流水线。
 * 设计期/回放态：只读展示。
 */
(function () {
  var BODY_LIMBS = [ // 1-based openpose limbSeq（同 controlnet_aux draw_bodypose）
    [2, 3], [2, 6], [3, 4], [4, 5], [6, 7], [7, 8], [2, 9], [9, 10],
    [10, 11], [2, 12], [12, 13], [13, 14], [2, 1], [1, 15], [15, 17],
    [1, 16], [16, 18], [3, 17], [6, 18]
  ]
  var BODY_COLORS = [
    [255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0],
    [85, 255, 0], [0, 255, 0], [0, 255, 85], [0, 255, 170], [0, 255, 255],
    [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255], [170, 0, 255],
    [255, 0, 255], [255, 0, 170], [255, 0, 85]
  ]
  var BODY_NAMES = ['鼻', '颈', '右肩', '右肘', '右腕', '左肩', '左肘', '左腕',
    '右髋', '右膝', '右踝', '左髋', '左膝', '左踝', '右眼', '左眼', '右耳', '左耳']
  var HAND_EDGES = [
    [0, 1], [1, 2], [2, 3], [3, 4], [0, 5], [5, 6], [6, 7], [7, 8], [0, 9],
    [9, 10], [10, 11], [11, 12], [0, 13], [13, 14], [14, 15], [15, 16],
    [0, 17], [17, 18], [18, 19], [19, 20]
  ]

  function rgb(c, a) { return 'rgba(' + c[0] + ',' + c[1] + ',' + c[2] + ',' + (a == null ? 1 : a) + ')' }

  function mount(el, props) {
    var cur = props || {}
    var session = null
    var kp = null
    var origSubset = null
    var selected = -1
    var hoverIdx = -1
    var dragging = false
    var dragLast = null
    var dragMode = 'bone'   // bone=骨骼约束(IK) / pan=整链平移 / free=自由拖拽
    var boneLen = {}       // 关节 -> 到父关节的初始长度（像素），拖动时锁定
    var decided = false
    var decidedInfo = ''
    var showHand = true, showFace = false, showBg = true
    var bgImg = null, bgTried = false
    var timer = null
    var disposed = false
    var overlay = null
    var canvas = null, ctx = null
    var hintEl = null
    var activeGateKey = null   // 实际使用的闸门 key（可能与选中执行不同，靠 /ext/gates 发现）

    el.style.cssText = 'display:flex;flex-direction:column;gap:6px;padding:6px;font:12px/1.5 sans-serif;color:#cdd6e4;height:100%;box-sizing:border-box;overflow:hidden'

    var status = document.createElement('div')
    status.style.cssText = 'font-size:12px;color:#8b98ad;white-space:pre-line'
    el.appendChild(status)

    var thumb = document.createElement('img')
    thumb.style.cssText = 'display:none;max-width:100%;border-radius:6px;background:#000;align-self:center'
    el.appendChild(thumb)

    var openBtn = document.createElement('button')
    openBtn.textContent = '🦴 打开骨架编辑器'
    openBtn.style.cssText = 'display:none;padding:8px 12px;border:0;border-radius:6px;font-size:13px;cursor:pointer;color:#fff;background:#7c5cff;font-weight:600'
    openBtn.onclick = openEditor
    el.appendChild(openBtn)

    function proxyUrl(path, method) {
      return '/api/v1/executions/' + cur.execution.id + '/nodes/' +
        encodeURIComponent(cur.nodeId) + '/service-proxy?path=' +
        encodeURIComponent(path) + '&method=' + (method || 'GET')
    }

    function gateKey() {
      // 优先用发现的活跃闸门（UI 选中执行可能落后于实际运行执行）；
      // 未发现时回退到当前选中执行推导
      return activeGateKey || (cur.execution.id + ':' + cur.nodeId)
    }

    function discoverGate() {
      // 经当前执行的 service-proxy（仅需连通性）列出活跃闸门，
      // 按创建时间取本节点最新一个——解决 UI 选中执行与实际运行执行错位
      return fetch(proxyUrl('/ext/gates', 'GET'))
        .then(function (r) { if (!r.ok) throw new Error('' + r.status); return r.json() })
        .then(function (d) {
          var suffix = ':' + cur.nodeId
          var mine = (d.gates || []).filter(function (g) {
            return g.key.slice(-suffix.length) === suffix
          }).sort(function (a, b) { return b.created - a.created })
          if (mine.length) activeGateKey = mine[0].key
          return !!mine.length
        })
    }

    function poll() {
      if (disposed || decided) return
      if (!cur.execution) return
      fetch(proxyUrl('/ext/gate/' + encodeURIComponent(gateKey()), 'GET'))
        .then(function (r) { if (!r.ok) throw new Error('' + r.status); return r.json() })
        .then(function (s) {
          if (s.decision) { decided = true; decidedInfo = s.decision.action; renderState(); return }
          if (!session) {
            session = s
            kp = JSON.parse(JSON.stringify(s.keypoints))
            origSubset = kp.subset.slice()
            oobSet = {}
            ;(kp.out_of_bounds || []).forEach(function (i) { oobSet[i] = 1 })
            buildBoneLen()
            renderState()
          }
        })
        .catch(function () {
          // 当前 key 无会话：尝试重新发现活跃闸门（执行错位/页面刷新恢复）
          if (!session) discoverGate().catch(function () {})
        })
    }

    function submitDecision(action) {
      if (!kp || decided) return
      var body = action === 'render'
        ? { action: 'render', keypoints: kp, include_hand: showHand, include_face: showFace }
        : { action: 'confirm' }
      if (hintEl) hintEl.textContent = '提交中…'
      fetch(proxyUrl('/ext/gate/' + encodeURIComponent(gateKey()) + '/decision', 'POST'),
        { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) })
        .then(function (r) { if (!r.ok) throw new Error('HTTP ' + r.status); return r.json() })
        .then(function () {
          decided = true; decidedInfo = action
          closeEditor()
          renderState()
        })
        .catch(function (e) { if (hintEl) hintEl.textContent = '提交失败：' + e.message })
    }

    function vis(i) {
      return kp.subset[i] > 0.3 && kp.candidate[i][0] >= 0 && kp.candidate[i][1] >= 0
    }

    // 关节名（不带「已选：」前缀，用于画布上跟随标签）
    function nameOf(i) {
      if (i < 18) return BODY_NAMES[i]
      if (i < 24) return ['右大趾', '右小趾', '右脚跟', '左大趾', '左小趾', '左脚跟'][i - 18]
      if (i < 92) return '脸' + (i - 24)
      if (i < 113) return '左指' + (i - 92)
      return '右指' + (i - 113)
    }

    // 连带说明：按当前拖动模式列出会一起动的关节（与所选点相邻的子节点开始）
    function chainText(i) {
      if (dragMode === 'free' || !kp) return ''
      var ns = subtree(i).filter(function (j) { return j !== i && vis(j) }).map(nameOf)
      if (!ns.length) return ''
      return (dragMode === 'bone' ? ' → 绕父关节旋转 ' : ' → 平移 ') + ns.join('/')
    }

    // 检测时被外推到画布外、已钳制的点（服务端 out_of_bounds 标记）
    var oobSet = {}
    function isOOB(i) { return !!oobSet[i] }

    // ---- 骨骼树（COCO-18 + DWPose 足部点）：长度约束拖动的依据 ----
    var PARENT = {
      0: 1, 14: 0, 15: 0, 16: 0, 17: 0,             // 头
      2: 1, 3: 2, 4: 3,                             // 右臂
      5: 1, 6: 5, 7: 6,                             // 左臂
      8: 1, 9: 8, 10: 9, 18: 10, 19: 10, 20: 10,    // 右腿+脚
      11: 1, 12: 11, 13: 12, 21: 13, 22: 13, 23: 13 // 左腿+脚
    }

    // 记录每段骨骼的初始长度（像素）：拖动时锁定，保证不会被拉长拉短
    function buildBoneLen() {
      boneLen = {}
      if (!kp) return
      for (var k in PARENT) {
        var i = +k, p = PARENT[i]
        if (i >= kp.candidate.length) continue
        var dx = (kp.candidate[i][0] - kp.candidate[p][0]) * kp.width
        var dy = (kp.candidate[i][1] - kp.candidate[p][1]) * kp.height
        boneLen[i] = Math.hypot(dx, dy)
      }
    }

    // 某关节的整棵子树（含自己）：拖动时一起旋转/平移
    function subtree(i) {
      var out = [i], stack = [i]
      while (stack.length) {
        var cur = stack.pop()
        for (var k in PARENT) {
          var j = +k
          if (PARENT[j] === cur && kp.subset[j] > 0.3) { out.push(j); stack.push(j) }
        }
      }
      return out
    }

    function draw() {
      if (!kp || !canvas) return
      var dw = canvas.width, dh = canvas.height
      ctx.clearRect(0, 0, dw, dh)
      ctx.fillStyle = '#000'; ctx.fillRect(0, 0, dw, dh)
      if (showBg && bgImg) {
        ctx.globalAlpha = 0.35
        ctx.drawImage(bgImg, 0, 0, dw, dh)
        ctx.globalAlpha = 1
      }
      var px = function (i) { return [kp.candidate[i][0] * dw, kp.candidate[i][1] * dh] }
      var i, a, b, pa, pb, p
      // 拖拽热区半径（与 hitTest 保持一致，做到「看到多大就能点多大」）
      var R = Math.max(16, dw * 0.05)

      // 连带预告：拖动/悬停某关节时，把它会一起动的点用青色虚线圈标出
      var chainSet = {}
      var chainSrc = selected >= 0 ? selected : (hoverIdx >= 0 ? hoverIdx : -1)
      if (dragMode !== 'free' && chainSrc >= 0 && kp) {
        subtree(chainSrc).forEach(function (j) { if (j !== chainSrc) chainSet[j] = 1 })
      }

      // ---- 1. 拖拽热区可视化：每个可见关节一个半透明大圆 ----
      for (var q = 0; q < 134; q++) {
        if (!vis(q)) continue
        if (q >= 24 && q < 92 && !showFace) continue
        if (q >= 92 && !showHand) continue
        p = px(q)
        var isSel = q === selected, isHov = q === hoverIdx, isChain = chainSet[q]
        var rr = q >= 92 ? R * 0.72 : (q < 18 || q < 24 ? R : R * 0.55)
        ctx.beginPath(); ctx.arc(p[0], p[1], rr, 0, 7)
        ctx.fillStyle = isSel ? 'rgba(125,255,170,0.22)'
          : isChain ? 'rgba(90,220,255,0.16)'
            : isHov ? 'rgba(255,255,255,0.16)'
              : 'rgba(255,255,255,0.05)'
        ctx.fill()
        if (isSel) { ctx.setLineDash([]); ctx.strokeStyle = 'rgba(125,255,170,0.95)'; ctx.lineWidth = 2.5 }
        else if (isHov) { ctx.setLineDash([]); ctx.strokeStyle = 'rgba(255,255,255,0.8)'; ctx.lineWidth = 2 }
        else if (isChain) { ctx.setLineDash([2, 3]); ctx.strokeStyle = 'rgba(90,220,255,0.9)'; ctx.lineWidth = 1.8 }
        else if (isOOB(q)) { ctx.setLineDash([3, 3]); ctx.strokeStyle = 'rgba(255,170,60,0.9)'; ctx.lineWidth = 1.6 }
        else { ctx.setLineDash([4, 4]); ctx.strokeStyle = 'rgba(255,255,255,0.20)'; ctx.lineWidth = 1 }
        ctx.stroke(); ctx.setLineDash([])
        // 越界点额外标一个 ⚠ 三角，提醒这个位置是猜的
        if (isOOB(q) && !isSel) {
          ctx.fillStyle = 'rgba(255,170,60,0.95)'
          ctx.font = 'bold ' + Math.max(10, dw * 0.018) + 'px sans-serif'
          ctx.fillText('⚠', p[0] + rr * 0.7, p[1] - rr * 0.7)
        }
      }

      ctx.lineWidth = 4
      for (var l = 0; l < BODY_LIMBS.length; l++) {
        a = BODY_LIMBS[l][0] - 1; b = BODY_LIMBS[l][1] - 1
        if (!vis(a) || !vis(b)) continue
        pa = px(a); pb = px(b)
        ctx.strokeStyle = rgb(BODY_COLORS[l])
        ctx.beginPath(); ctx.moveTo(pa[0], pa[1]); ctx.lineTo(pb[0], pb[1]); ctx.stroke()
      }
      for (i = 0; i < 18; i++) {
        if (!vis(i)) continue
        p = px(i)
        ctx.fillStyle = rgb(BODY_COLORS[i])
        ctx.beginPath(); ctx.arc(p[0], p[1], i === selected ? 12 : 8, 0, 7); ctx.fill()
        if (i === selected) { ctx.strokeStyle = '#fff'; ctx.lineWidth = 3; ctx.stroke() }
      }
      ctx.fillStyle = '#ffff66'
      for (i = 18; i < 24; i++) {
        if (!vis(i)) continue
        p = px(i)
        ctx.beginPath(); ctx.arc(p[0], p[1], i === selected ? 9 : 6, 0, 7); ctx.fill()
        if (i === selected) { ctx.strokeStyle = '#fff'; ctx.lineWidth = 2; ctx.stroke() }
      }
      if (showHand) {
        for (var h = 0; h < 2; h++) {
          var off = h === 0 ? 92 : 113
          ctx.lineWidth = 2; ctx.strokeStyle = 'rgba(255,255,255,0.75)'
          for (var e = 0; e < HAND_EDGES.length; e++) {
            a = off + HAND_EDGES[e][0]; b = off + HAND_EDGES[e][1]
            if (!vis(a) || !vis(b)) continue
            pa = px(a); pb = px(b)
            ctx.beginPath(); ctx.moveTo(pa[0], pa[1]); ctx.lineTo(pb[0], pb[1]); ctx.stroke()
          }
          ctx.fillStyle = '#ff4444'
          for (i = 0; i < 21; i++) {
            if (!vis(off + i)) continue
            p = px(off + i)
            ctx.beginPath(); ctx.arc(p[0], p[1], (off + i) === selected ? 9 : 5.5, 0, 7); ctx.fill()
            if ((off + i) === selected) { ctx.strokeStyle = '#fff'; ctx.lineWidth = 2; ctx.stroke() }
          }
        }
      }
      if (showFace) {
        ctx.fillStyle = 'rgba(255,255,255,0.85)'
        for (i = 24; i < 92; i++) {
          if (!vis(i)) continue
          p = px(i)
          ctx.beginPath(); ctx.arc(p[0], p[1], 2.5, 0, 7); ctx.fill()
        }
      }

      // ---- 2. 拖动中的状态特效：十字准线 + 发光选中点 + 跟随标签 ----
      if (dragging && selected >= 0) {
        p = px(selected)
        ctx.save()
        ctx.setLineDash([7, 7])
        ctx.strokeStyle = 'rgba(125,255,170,0.5)'; ctx.lineWidth = 1.5
        ctx.beginPath()
        ctx.moveTo(0, p[1]); ctx.lineTo(dw, p[1])
        ctx.moveTo(p[0], 0); ctx.lineTo(p[0], dh)
        ctx.stroke()
        ctx.restore()

        ctx.save()
        ctx.shadowColor = 'rgba(125,255,170,0.95)'; ctx.shadowBlur = 20
        ctx.fillStyle = 'rgba(125,255,170,0.85)'
        ctx.beginPath(); ctx.arc(p[0], p[1], R * 0.6, 0, 7); ctx.fill()
        ctx.restore()
        ctx.strokeStyle = '#fff'; ctx.lineWidth = 3
        ctx.beginPath(); ctx.arc(p[0], p[1], R * 0.6, 0, 7); ctx.stroke()

        // 拖动中：青色虚线连到会被一起动的点，一眼看出连带关系
        if (dragMode !== 'free') {
          ctx.save()
          ctx.setLineDash([4, 5]); ctx.strokeStyle = 'rgba(90,220,255,0.75)'; ctx.lineWidth = 2
          var chs = subtree(selected).filter(function (j) { return j !== selected && vis(j) })
          for (var ci = 0; ci < chs.length; ci++) {
            var cp = px(chs[ci])
            ctx.beginPath(); ctx.moveTo(p[0], p[1]); ctx.lineTo(cp[0], cp[1]); ctx.stroke()
          }
          ctx.restore()
        }

        var lab = nameOf(selected)
        // 顺带标出该段骨骼长度，方便核对比例
        if (PARENT[selected] != null) {
          var parP = px(PARENT[selected])
          lab += ' ' + Math.round(Math.hypot(p[0] - parP[0], p[1] - parP[1])) + 'px'
        }
        ctx.font = 'bold ' + Math.max(13, dw * 0.025) + 'px sans-serif'
        var tw = ctx.measureText(lab).width
        var lx = Math.min(Math.max(p[0] - tw / 2, 4), dw - tw - 6)
        var ly = p[1] - R - 12
        if (ly < 20) ly = p[1] + R + 24
        ctx.fillStyle = 'rgba(0,0,0,0.78)'
        ctx.fillRect(lx - 7, ly - 15, tw + 14, 22)
        ctx.fillStyle = '#7dffaa'; ctx.fillText(lab, lx, ly)
      } else if (hoverIdx >= 0 && vis(hoverIdx) && hoverIdx !== selected) {
        // 悬停时在热区上方标出关节名，避免盲猜
        p = px(hoverIdx)
        var lab2 = nameOf(hoverIdx)
        ctx.font = 'bold ' + Math.max(12, dw * 0.023) + 'px sans-serif'
        var tw2 = ctx.measureText(lab2).width
        var lx2 = Math.min(Math.max(p[0] - tw2 / 2, 4), dw - tw2 - 6)
        var ly2 = p[1] - R - 12
        if (ly2 < 20) ly2 = p[1] + R + 24
        ctx.fillStyle = 'rgba(0,0,0,0.7)'
        ctx.fillRect(lx2 - 6, ly2 - 14, tw2 + 12, 20)
        ctx.fillStyle = '#fff'; ctx.fillText(lab2, lx2, ly2)
      }
    }

    function hitTest(x, y) {
      if (!kp) return -1
      var dw = canvas.width, dh = canvas.height
      // 与绘制热区半径一致（R = max(16, dw*0.05)），手部略小
      var baseR = Math.max(16, dw * 0.05)
      var best = -1, bestD = baseR
      var check = function (i, r) {
        if (!vis(i)) return
        var d = Math.hypot(kp.candidate[i][0] * dw - x, kp.candidate[i][1] * dh - y)
        if (d < Math.min(bestD, r)) { bestD = d; best = i }
      }
      for (var i = 0; i < 24; i++) check(i, baseR)
      if (showHand) for (i = 92; i < 134; i++) check(i, baseR * 0.72)
      if (showFace) for (i = 24; i < 92; i++) check(i, baseR * 0.55)
      return best
    }

    function labelOf(i) {
      if (i < 18) return '已选：' + BODY_NAMES[i]
      if (i < 24) return '已选：足部点 ' + (i - 18)
      if (i < 92) return '已选：面部点 ' + (i - 24)
      if (i < 113) return '已选：左手点 ' + (i - 92)
      return '已选：右手点 ' + (i - 113)
    }

    function mkBtn(text, bg) {
      var b = document.createElement('button')
      b.textContent = text
      b.style.cssText = 'padding:10px 16px;border:0;border-radius:8px;font-size:14px;cursor:pointer;color:#fff;background:' + bg + ';font-weight:600'
      return b
    }
    function mkChk(label, checked, onchange) {
      var l = document.createElement('label')
      l.style.cssText = 'font-size:13px;display:flex;align-items:center;gap:4px;cursor:pointer;color:#cdd6e4'
      var c = document.createElement('input')
      c.type = 'checkbox'; c.checked = checked
      c.onchange = function () { onchange(c.checked) }
      l.appendChild(c); l.appendChild(document.createTextNode(label))
      return l
    }

    function openEditor() {
      if (!kp || overlay) return
      // 底图（源图）懒加载
      if (!bgTried && session && session.source_id) {
        bgTried = true
        bgImg = new Image()
        bgImg.onload = function () { draw() }
        bgImg.onerror = function () { bgImg = null }
        bgImg.src = proxyUrl('/images/' + session.source_id, 'GET')
      }
      overlay = document.createElement('div')
      overlay.style.cssText = 'position:fixed;inset:0;z-index:99999;background:rgba(8,10,16,0.96);' +
        'display:flex;flex-direction:column;align-items:center;justify-content:center;gap:10px;padding:12px;box-sizing:border-box'
      // 阻止事件穿透到画布
      ;['pointerdown', 'pointermove', 'pointerup', 'wheel', 'touchstart', 'touchmove', 'click'].forEach(function (ev) {
        overlay.addEventListener(ev, function (e) { e.stopPropagation() }, { passive: false })
      })

      var title = document.createElement('div')
      title.textContent = '🦴 骨架修正：拖动关键点到正确位置'
      title.style.cssText = 'color:#fff;font-size:15px;font-weight:600'
      overlay.appendChild(title)

      canvas = document.createElement('canvas')
      var maxW = Math.min(window.innerWidth * 0.92, 560)
      var maxH = window.innerHeight * 0.62
      var scale = Math.min(maxW / kp.width, maxH / kp.height)
      canvas.width = Math.round(kp.width * scale)
      canvas.height = Math.round(kp.height * scale)
      canvas.style.cssText = 'background:#000;border-radius:8px;touch-action:none;max-width:92vw'
      ctx = canvas.getContext('2d')
      overlay.appendChild(canvas)

      hintEl = document.createElement('div')
      hintEl.style.cssText = 'color:#8b98ad;font-size:13px;min-height:18px;text-align:center'
      hintEl.textContent = '点住彩色圆点拖动；「手/脸」控制是否参与绘制'
      overlay.appendChild(hintEl)

      var bar = document.createElement('div')
      bar.style.cssText = 'display:flex;flex-wrap:wrap;gap:8px;align-items:center;justify-content:center'
      var btnOk = mkBtn('✅ 修正完成·放行', '#2d8a4e')
      var btnPass = mkBtn('⏭ 原样继续', '#55617a')
      var btnHide = mkBtn('🚫 隐藏选中点', '#8a5a2d')
      var btnRestore = mkBtn('↩ 恢复隐藏', '#55617a')
      var btnClose = mkBtn('✕ 关闭', '#3a4152')
      btnOk.onclick = function () { submitDecision('render') }
      btnPass.onclick = function () { submitDecision('confirm') }
      btnHide.onclick = function () {
        if (kp && selected >= 0) { kp.subset[selected] = 0; selected = -1; hintEl.textContent = '已隐藏该点'; draw() }
      }
      btnRestore.onclick = function () {
        if (kp && origSubset) { kp.subset = origSubset.slice(); selected = -1; hintEl.textContent = '已恢复全部隐藏点'; draw() }
      }
      btnClose.onclick = closeEditor
      bar.appendChild(btnOk); bar.appendChild(btnPass); bar.appendChild(btnHide)
      bar.appendChild(btnRestore)
      bar.appendChild(mkChk('手', showHand, function (v) { showHand = v; draw() }))
      bar.appendChild(mkChk('脸', showFace, function (v) { showFace = v; draw() }))
      bar.appendChild(mkChk('底图', showBg, function (v) { showBg = v; draw() }))
      var modeSel = document.createElement('select')
      modeSel.style.cssText = 'padding:7px 8px;border-radius:8px;border:1px solid #3a4152;' +
        'background:#1b1f2a;color:#dbe3ef;font-size:13px;cursor:pointer'
      ;[['bone', '🦴 骨骼约束（长度不变）'], ['pan', '↔ 整链平移'], ['free', '✋ 自由拖拽']]
        .forEach(function (o) {
          var op = document.createElement('option')
          op.value = o[0]; op.textContent = o[1]; op.selected = (dragMode === o[0])
          modeSel.appendChild(op)
        })
      modeSel.onchange = function () {
        dragMode = modeSel.value
        hintEl.textContent = dragMode === 'bone'
          ? '骨骼约束：拖任意关节都绕父关节旋转，所有骨骼长度恒定'
          : dragMode === 'pan'
            ? '整链平移：子树刚性跟随，长度不变但整条链会滑动'
            : '自由拖拽：只动选中的点（骨骼长度会变）'
        draw()
      }
      bar.appendChild(modeSel)
      bar.appendChild(btnClose)
      bar.appendChild(btnClose)
      overlay.appendChild(bar)

      canvas.addEventListener('pointerdown', function (e) {
        if (decided || !kp) return
        e.preventDefault()
        var r = canvas.getBoundingClientRect()
        var i = hitTest(e.clientX - r.left, e.clientY - r.top)
        if (i >= 0) {
          selected = i; dragging = true; hoverIdx = i
          dragLast = [e.clientX - r.left, e.clientY - r.top]
          try { canvas.setPointerCapture(e.pointerId) } catch (_) {}
          // 拖动状态外观：光标 + 高亮边框
          canvas.style.cursor = 'grabbing'
          canvas.style.boxShadow = '0 0 0 2px #7dffaa, 0 0 24px rgba(125,255,170,0.55)'
          document.body.style.overflow = 'hidden'
          hintEl.textContent = nameOf(i) + ' 拖动中…' + chainText(i)
          draw()
        } else {
          hintEl.textContent = '没抓到点：请按住虚线圆圈内部再拖动'
          // 短暂闪烁所有热区，提示可点范围
          var flash = 0
          var iv = setInterval(function () {
            flash++
            if (flash > 3 || dragging || !ctx) { clearInterval(iv); return }
            draw()
          }, 140)
        }
      })
      canvas.addEventListener('pointermove', function (e) {
        if (!kp) return
        var r = canvas.getBoundingClientRect()
        var x = Math.min(Math.max(e.clientX - r.left, 0), r.width)
        var y = Math.min(Math.max(e.clientY - r.top, 0), r.height)
        if (dragging && selected >= 0) {
          if (dragMode === 'free') {
            // 自由：只挪选中的点（长度会变，仅供极端姿势摆位）
            kp.candidate[selected] = [x / r.width, y / r.height]
          } else if (dragMode === 'pan') {
            // 平移：整棵子树刚性平移（长度不变，但整条链会滑走）
            var pdx = (x - dragLast[0]) / r.width, pdy = (y - dragLast[1]) / r.height
            subtree(selected).forEach(function (j) {
              kp.candidate[j] = [kp.candidate[j][0] + pdx, kp.candidate[j][1] + pdy]
            })
          } else {
            // 骨骼约束（IK）：父关节不动，选中点绕父旋转到手指方向，
            // 距离锁定为初始骨骼长度；整棵子树绕父刚体旋转 → 所有段长度恒定
            var tgt = [x / r.width * kp.width, y / r.height * kp.height]
            var par = PARENT[selected]
            var curPx = [kp.candidate[selected][0] * kp.width,
                         kp.candidate[selected][1] * kp.height]
            if (par == null || !vis(par)) {
              // 根关节（颈）：整树平移
              var rdx = (tgt[0] - curPx[0]) / kp.width, rdy = (tgt[1] - curPx[1]) / kp.height
              subtree(selected).forEach(function (j) {
                kp.candidate[j] = [kp.candidate[j][0] + rdx, kp.candidate[j][1] + rdy]
              })
            } else {
              var Ppx = [kp.candidate[par][0] * kp.width, kp.candidate[par][1] * kp.height]
              var L = boneLen[selected] || Math.hypot(curPx[0] - Ppx[0], curPx[1] - Ppx[1])
              if (!(L > 0.5)) L = Math.hypot(curPx[0] - Ppx[0], curPx[1] - Ppx[1]) || 1
              var vx = tgt[0] - Ppx[0], vy = tgt[1] - Ppx[1]
              var vlen = Math.hypot(vx, vy)
              if (vlen > 0.5) {
                var nx = Ppx[0] + vx / vlen * L, ny = Ppx[1] + vy / vlen * L
                var th = Math.atan2(ny - Ppx[1], nx - Ppx[0]) -
                         Math.atan2(curPx[1] - Ppx[1], curPx[0] - Ppx[0])
                var cs = Math.cos(th), sn = Math.sin(th)
                subtree(selected).forEach(function (j) {
                  var jx = kp.candidate[j][0] * kp.width - Ppx[0]
                  var jy = kp.candidate[j][1] * kp.height - Ppx[1]
                  kp.candidate[j] = [
                    (Ppx[0] + jx * cs - jy * sn) / kp.width,
                    (Ppx[1] + jx * sn + jy * cs) / kp.height
                  ]
                })
              }
            }
          }
          dragLast = [x, y]
          // 全部点限制在画布内，避免拖出边界后消失
          for (var z = 0; z < 24; z++) {
            kp.candidate[z] = [
              Math.min(Math.max(kp.candidate[z][0], 0), 1),
              Math.min(Math.max(kp.candidate[z][1], 0), 1)
            ]
          }
          hintEl.textContent = '拖动中：' + nameOf(selected) +
            (PARENT[selected] != null ?
              '  ' + Math.round(boneLen[selected] || 0) + 'px' : '') +
            chainText(selected)
          draw()
        } else if (!decided) {
          // 悬停高亮：提示当前会抓到哪个点
          var h = hitTest(x, y)
          if (h !== hoverIdx) {
            hoverIdx = h
            // 鼠标时用光标形状也提示可拖动
            canvas.style.cursor = h >= 0 ? 'grab' : 'default'
            if (h >= 0) hintEl.textContent = nameOf(h) +
              (isOOB(h) ? ' ⚠检测越界·位置不可靠，建议手改' : '：按住拖动可修正') + chainText(h)
            draw()
          }
        }
      })
      canvas.addEventListener('pointerup', function () {
        if (dragging) {
          hintEl.textContent = (selected >= 0 ? nameOf(selected) + ' 已调整' : '') +
            ' · 可继续拖其它点，或点「✅ 修正完成·放行」'
        }
        dragging = false
        canvas.style.cursor = 'default'
        canvas.style.boxShadow = 'none'
        document.body.style.overflow = ''
        draw()
      })
      canvas.addEventListener('pointercancel', function () {
        dragging = false
        canvas.style.cursor = 'default'
        canvas.style.boxShadow = 'none'
        document.body.style.overflow = ''
        draw()
      })

      document.body.appendChild(overlay)
      draw()
    }

    function closeEditor() {
      if (overlay && overlay.parentNode) overlay.parentNode.removeChild(overlay)
      overlay = null; canvas = null; ctx = null; hintEl = null
      dragging = false; hoverIdx = -1
      document.body.style.overflow = ''
    }

    function renderState() {
      if (decided) {
        openBtn.style.display = 'none'
        status.textContent = '✅ 已放行（' + (decidedInfo === 'render' ? '修正骨架' : '原骨架') + '），流水线继续'
        return
      }
      if (session && !decided) {
        // 闸门会话存活且未决策 = 节点正在等待（以服务端会话为准，不看 UI 选中执行）
        openBtn.style.display = 'block'
        status.textContent = '⏸ 等待人工确认骨架'
        if (session.image_id && !thumb.src) {
          thumb.src = proxyUrl('/images/' + session.image_id + '?thumb=300', 'GET')
          thumb.style.display = 'block'
        }
        return
      }
      openBtn.style.display = 'none'
      if (cur.status === 'running') {
        status.textContent = '⏳ 姿态检测中…（DWPose CPU 推理约 10~30s）'
        return
      }
      if (cur.outputs && cur.outputs.image) {
        status.textContent = '完成（gate_action=' + (cur.outputs.gate_action || '-') + '）'
        return
      }
      status.textContent = '执行到此节点时，点「打开骨架编辑器」确认/修正姿态'
    }

    timer = setInterval(poll, 3000)
    poll()

    renderState()
    return {
      update: function (p) {
        var wasRunning = cur.status
        cur = p || {}
        if (cur.outputs && cur.outputs.gate_action && !decided) {
          decided = true
          decidedInfo = cur.outputs.gate_action === 'render' ? 'render' : 'confirm'
          closeEditor()
        }
        if (cur.status !== wasRunning) poll()
        renderState()
      },
      unmount: function () { disposed = true; if (timer) clearInterval(timer); closeEditor() }
    }
  }

  if (window.FlowXNodeWidget && window.FlowXNodeWidget.define) {
    window.FlowXNodeWidget.define(mount)
  }
  if (typeof module !== 'undefined') module.exports = mount
})()
