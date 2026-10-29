/* pose-gate 节点 widget：骨架人工确认/修正编辑器（契约 apiVersion 1）。
 *
 * 执行中：节点检测出骨架后创建 /ext/gate/{exec}:{node} 会话并等待。
 * 本组件经节点级 service-proxy 轮询会话，渲染可拖拽关键点编辑器
 * （源图半透明底 + openpose 配色骨架），用户拖动错位关节/隐藏误检点后
 * 「确认放行」——服务端按修正关键点重渲染骨架供下游 ControlNet。
 * 设计期/回放态：只读展示参数与结果帧。
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
    var session = null       // {image_id, source_id, keypoints}
    var kp = null            // 编辑中的关键点 {width,height,candidate,subset}
    var origSubset = null
    var selected = -1
    var dragging = false
    var decided = false
    var decidedInfo = ''
    var showHand = true, showFace = false, showBg = true
    var bgImg = null, bgTried = false
    var timer = null
    var disposed = false

    el.style.cssText = 'display:flex;flex-direction:column;gap:6px;padding:6px;font:12px/1.5 sans-serif;color:#cdd6e4;height:100%;box-sizing:border-box;overflow:hidden'

    var status = document.createElement('div')
    status.style.cssText = 'font-size:12px;color:#8b98ad;white-space:pre-line'
    el.appendChild(status)

    var canvas = document.createElement('canvas')
    canvas.style.cssText = 'display:none;background:#000;border-radius:6px;touch-action:none;align-self:center'
    el.appendChild(canvas)
    var ctx = canvas.getContext('2d')

    var tools = document.createElement('div')
    tools.style.cssText = 'display:none;flex-wrap:wrap;gap:6px;align-items:center'
    el.appendChild(tools)

    function mkBtn(text, bg) {
      var b = document.createElement('button')
      b.textContent = text
      b.style.cssText = 'padding:4px 10px;border:0;border-radius:5px;font-size:12px;cursor:pointer;color:#fff;background:' + bg
      return b
    }
    function mkChk(label, checked, onchange) {
      var l = document.createElement('label')
      l.style.cssText = 'font-size:12px;display:flex;align-items:center;gap:3px;cursor:pointer'
      var c = document.createElement('input')
      c.type = 'checkbox'; c.checked = checked
      c.onchange = function () { onchange(c.checked) }
      l.appendChild(c); l.appendChild(document.createTextNode(label))
      return l
    }

    var btnConfirm = mkBtn('✅ 修正完成·放行', '#2d8a4e')
    var btnPass = mkBtn('⏭ 原样继续', '#55617a')
    var btnHide = mkBtn('🚫 隐藏选中点', '#8a5a2d')
    var btnRestore = mkBtn('↩ 恢复隐藏', '#55617a')
    btnConfirm.onclick = function () { submitDecision('render') }
    btnPass.onclick = function () { submitDecision('confirm') }
    btnHide.onclick = function () {
      if (kp && selected >= 0) { kp.subset[selected] = 0; selected = -1; draw() }
    }
    btnRestore.onclick = function () {
      if (kp && origSubset) { kp.subset = origSubset.slice(); selected = -1; draw() }
    }
    tools.appendChild(btnConfirm); tools.appendChild(btnPass)
    tools.appendChild(btnHide); tools.appendChild(btnRestore)
    tools.appendChild(mkChk('手', true, function (v) { showHand = v; draw() }))
    tools.appendChild(mkChk('脸', false, function (v) { showFace = v; draw() }))
    tools.appendChild(mkChk('底图', true, function (v) { showBg = v; draw() }))

    function proxyUrl(path, method) {
      return '/api/v1/executions/' + cur.execution.id + '/nodes/' +
        encodeURIComponent(cur.nodeId) + '/service-proxy?path=' +
        encodeURIComponent(path) + '&method=' + (method || 'GET')
    }

    function gateKey() { return cur.execution.id + ':' + cur.nodeId }

    function poll() {
      if (disposed || decided) return
      if (!cur.execution || cur.status !== 'running') return
      fetch(proxyUrl('/ext/gate/' + encodeURIComponent(gateKey()), 'GET'))
        .then(function (r) { if (!r.ok) throw new Error('' + r.status); return r.json() })
        .then(function (s) {
          if (s.decision) { decided = true; decidedInfo = s.decision.action; renderState(); return }
          if (!session) {
            session = s
            kp = JSON.parse(JSON.stringify(s.keypoints))
            origSubset = kp.subset.slice()
            loadBg()
            renderState()
          }
        })
        .catch(function () { /* 会话未建好/已清理：下轮再试 */ })
    }

    function loadBg() {
      if (bgTried || !session || !session.source_id) return
      bgTried = true
      var img = new Image()
      img.onload = function () { bgImg = img; draw() }
      img.onerror = function () {}
      img.src = proxyUrl('/images/' + session.source_id, 'GET')
    }

    function submitDecision(action) {
      if (!kp || decided) return
      var body = action === 'render'
        ? { action: 'render', keypoints: kp, include_hand: showHand, include_face: showFace }
        : { action: 'confirm' }
      btnConfirm.disabled = btnPass.disabled = true
      fetch(proxyUrl('/ext/gate/' + encodeURIComponent(gateKey()) + '/decision', 'POST'),
        { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) })
        .then(function (r) { if (!r.ok) throw new Error('HTTP ' + r.status); return r.json() })
        .then(function () { decided = true; decidedInfo = action; renderState() })
        .catch(function (e) {
          btnConfirm.disabled = btnPass.disabled = false
          status.textContent = '提交失败：' + e.message
        })
    }

    function vis(i) {
      return kp.subset[i] > 0.3 && kp.candidate[i][0] >= 0 && kp.candidate[i][1] >= 0
    }

    function draw() {
      if (!kp) return
      var dw = Math.min(320, el.clientWidth - 12)
      var scale = dw / kp.width
      var dh = Math.round(kp.height * scale)
      canvas.width = dw; canvas.height = dh
      ctx.clearRect(0, 0, dw, dh)
      if (showBg && bgImg) {
        ctx.globalAlpha = 0.35
        ctx.drawImage(bgImg, 0, 0, dw, dh)
        ctx.globalAlpha = 1
      }
      var px = function (i) { return [kp.candidate[i][0] * dw, kp.candidate[i][1] * dh] }
      // body limbs
      ctx.lineWidth = 3
      for (var l = 0; l < BODY_LIMBS.length; l++) {
        var a = BODY_LIMBS[l][0] - 1, b = BODY_LIMBS[l][1] - 1
        if (!vis(a) || !vis(b)) continue
        var pa = px(a), pb = px(b)
        ctx.strokeStyle = rgb(BODY_COLORS[l])
        ctx.beginPath(); ctx.moveTo(pa[0], pa[1]); ctx.lineTo(pb[0], pb[1]); ctx.stroke()
      }
      // body joints
      for (var i = 0; i < 18; i++) {
        if (!vis(i)) continue
        var p = px(i)
        ctx.fillStyle = rgb(BODY_COLORS[i])
        ctx.beginPath(); ctx.arc(p[0], p[1], i === selected ? 7 : 5, 0, 7); ctx.fill()
        if (i === selected) { ctx.strokeStyle = '#fff'; ctx.lineWidth = 2; ctx.stroke() }
      }
      // foot (18-23)
      ctx.fillStyle = '#ffff66'
      for (i = 18; i < 24; i++) {
        if (!vis(i)) continue
        p = px(i)
        ctx.beginPath(); ctx.arc(p[0], p[1], i === selected ? 6 : 3.5, 0, 7); ctx.fill()
        if (i === selected) { ctx.strokeStyle = '#fff'; ctx.lineWidth = 1.5; ctx.stroke() }
      }
      // hands
      if (showHand) {
        for (var h = 0; h < 2; h++) {
          var off = h === 0 ? 92 : 113
          ctx.lineWidth = 1.5; ctx.strokeStyle = 'rgba(255,255,255,0.75)'
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
            ctx.beginPath(); ctx.arc(p[0], p[1], (off + i) === selected ? 6 : 3, 0, 7); ctx.fill()
            if ((off + i) === selected) { ctx.strokeStyle = '#fff'; ctx.lineWidth = 1.5; ctx.stroke() }
          }
        }
      }
      // face
      if (showFace) {
        ctx.fillStyle = 'rgba(255,255,255,0.85)'
        for (i = 24; i < 92; i++) {
          if (!vis(i)) continue
          p = px(i)
          ctx.beginPath(); ctx.arc(p[0], p[1], 1.8, 0, 7); ctx.fill()
        }
      }
    }

    function hitTest(x, y) {
      if (!kp) return -1
      var dw = canvas.width, dh = canvas.height
      var best = -1, bestD = 12
      var check = function (i) {
        if (!vis(i)) return
        var px = kp.candidate[i][0] * dw, py = kp.candidate[i][1] * dh
        var d = Math.hypot(px - x, py - y)
        if (d < bestD) { bestD = d; best = i }
      }
      for (var i = 0; i < 24; i++) check(i)
      if (showHand) for (i = 92; i < 134; i++) check(i)
      if (showFace) for (i = 24; i < 92; i++) check(i)
      return best
    }

    canvas.addEventListener('pointerdown', function (e) {
      if (decided || !kp) return
      var r = canvas.getBoundingClientRect()
      var i = hitTest(e.clientX - r.left, e.clientY - r.top)
      if (i >= 0) {
        selected = i; dragging = true
        canvas.setPointerCapture(e.pointerId)
        status.textContent = labelOf(i) + '（拖动修正位置，或点「隐藏选中点」去除误检）'
        draw()
      }
    })
    canvas.addEventListener('pointermove', function (e) {
      if (!dragging || selected < 0 || !kp) return
      var r = canvas.getBoundingClientRect()
      var x = Math.min(Math.max(e.clientX - r.left, 0), r.width)
      var y = Math.min(Math.max(e.clientY - r.top, 0), r.height)
      kp.candidate[selected] = [x / r.width, y / r.height]
      draw()
    })
    canvas.addEventListener('pointerup', function () { dragging = false })

    function labelOf(i) {
      if (i < 18) return '已选：' + BODY_NAMES[i]
      if (i < 24) return '已选：足部点 ' + (i - 18)
      if (i < 92) return '已选：面部点 ' + (i - 24)
      if (i < 113) return '已选：左手点 ' + (i - 92)
      return '已选：右手点 ' + (i - 113)
    }

    function renderState() {
      if (decided) {
        canvas.style.display = 'none'; tools.style.display = 'none'
        status.textContent = '✅ 已放行（' + (decidedInfo === 'render' ? '修正骨架' : '原骨架') + '），流水线继续'
        return
      }
      if (session && cur.status === 'running') {
        canvas.style.display = 'block'; tools.style.display = 'flex'
        status.textContent = '⏸ 等待人工确认：拖动关键点修正骨架，「手/脸」控制是否绘制\n' +
          '改完点「修正完成·放行」；无需修改点「原样继续」'
        draw()
        return
      }
      if (cur.status === 'running') {
        status.textContent = '⏳ 姿态检测中…（DWPose CPU 推理约 10~30s）'
        return
      }
      if (cur.outputs && cur.outputs.image) {
        status.textContent = '完成（gate_action=' + (cur.outputs.gate_action || '-') + '）'
        return
      }
      status.textContent = '执行时在此确认/修正骨架（拖关键点、隐藏误检点）'
    }

    timer = setInterval(poll, 3000)

    renderState()
    return {
      update: function (p) {
        var wasRunning = cur.status
        cur = p || {}
        if (cur.outputs && cur.outputs.gate_action && !decided) {
          decided = true
          decidedInfo = cur.outputs.gate_action === 'render' ? 'render' : 'confirm'
        }
        if (cur.status !== wasRunning) poll()
        renderState()
      },
      unmount: function () { disposed = true; if (timer) clearInterval(timer) }
    }
  }

  if (window.FlowXNodeWidget && window.FlowXNodeWidget.define) {
    window.FlowXNodeWidget.define(mount)
  }
  if (typeof module !== 'undefined') module.exports = mount
})()
