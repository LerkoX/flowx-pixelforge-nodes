/* ===== zero123 widget：上帝视角相机位姿控件 + 结果展示（three.js r128 vendored）=====
 *
 * 交互模型（v3）：
 *   节点内 = 只读展示：小型「上帝视角」示意图（球面线框 + 球心参考图 + 球面上的
 *            相机图标 + 视野锥）+ azimuth/elevation 读数 + 生成结果图。不响应拖拽。
 *   点击节点 → 弹出大面板：左键在球面上拖拽即把相机移到该位置（不用精确点中图标），
 *            右键拖拽旋转观察视角，滚轮缩放。松手写回参数。
 *            只写参数、不触发生成（生成由流水线运行触发）。
 *
 * 角度定义（与 OrbitControls / Qwen Multiangle Camera 习惯一致）：
 *   azimuth   0=正面(+Z) 90=右侧(+X) 180=背面(-Z) -90=左侧(-X)
 *   elevation 0=水平 正=俯视(在上方) 负=仰视(在下方)
 *   相机位置 = R*(sin(az)cos(el), sin(el), cos(az)cos(el))，镜头始终朝向球心。
 *
 * WebGL 健壮性：画布上同包多实例会同时初始化（并发创建 context 在部分环境失败），
 *   故按全局序号错峰 + 属性降级（antialias → 无）+ 失败退避重试。
 *
 * 参考图三条链：① params.ref_image_path（本地路径，未执行即可预览）
 *   ② props.media.input 物化产物 ③ outputs.__inputs_resolved → 节点级 service-proxy。
 */
(function () {
  var DEG = Math.PI / 180
  var RAD = 180 / Math.PI
  var R = 1.6 // 球半径（世界单位）

  var glSeq = 0 // 全局实例序号：用于错峰初始化 WebGL

  function clamp(v, a, b) { return v < a ? a : (v > b ? b : v) }
  function num(v, d) { var n = parseFloat(v); return isFinite(n) ? n : d }
  function normAz(a) { a = a % 360; if (a > 180) a -= 360; if (a < -180) a += 360; return a }
  function mediaUrl(p) { return '/api/v1/media/file?path=' + encodeURIComponent(p) }

  // 移动端/触屏：布局紧凑、触摸目标加大（与 Studio 的 768px 断点对齐）
  function isTouchUI() {
    try {
      if (window.innerWidth <= 768) return true
      if (window.matchMedia && window.matchMedia('(pointer: coarse)').matches) return true
    } catch (e) { /* ignore */ }
    return false
  }

  // 球面方位标签（0/90/180/-90），帮助判断朝向
  function makeLabel(text, color) {
    var c = document.createElement('canvas')
    c.width = 128; c.height = 64
    var g = c.getContext('2d')
    g.fillStyle = color || '#8b96a5'
    g.font = 'bold 44px sans-serif'
    g.textAlign = 'center'; g.textBaseline = 'middle'
    g.fillText(text, 64, 32)
    var tex = new THREE.CanvasTexture(c)
    var sp = new THREE.Sprite(new THREE.SpriteMaterial({ map: tex, transparent: true, depthTest: false }))
    sp.scale.set(0.42, 0.21, 1)
    return sp
  }

  /* =============== 弹窗样式（注入一次，媒体查询做移动端适配）=============== */
  var STYLE_ID = 'zero123-widget-style'
  function ensureStyle() {
    if (document.getElementById(STYLE_ID)) return
    var s = document.createElement('style')
    s.id = STYLE_ID
    s.textContent = [
      '.z123-mask{position:fixed;inset:0;z-index:10000;background:rgba(0,0,0,.78);',
      'display:flex;align-items:center;justify-content:center;padding:16px;overscroll-behavior:contain}',
      '.z123-panel{width:min(92vw,940px);height:min(90vh,780px);background:#0f1217;',
      'border:1px solid #263041;border-radius:12px;display:flex;flex-direction:column;',
      'overflow:hidden;box-shadow:0 24px 60px rgba(0,0,0,.6)}',
      '.z123-head{display:flex;align-items:center;gap:12px;padding:10px 14px;',
      'border-bottom:1px solid #1e2634;flex:0 0 auto;flex-wrap:wrap}',
      '.z123-title{font-size:13px;font-weight:600;color:#e5e9f0;white-space:nowrap}',
      '.z123-num{font-size:12px;color:#93a2b8;font-variant-numeric:tabular-nums;white-space:nowrap}',
      '.z123-spacer{flex:1}',
      '.z123-close{background:#1b2330;border:1px solid #2b3546;color:#c3ccda;border-radius:6px;',
      'width:30px;height:28px;cursor:pointer;font-size:13px;flex:0 0 auto}',
      '.z123-stage{position:relative;flex:1 1 auto;min-height:0}',
      '.z123-foot{display:flex;flex-wrap:wrap;gap:6px;align-items:center;padding:8px 12px;',
      'border-top:1px solid #1e2634;flex:0 0 auto}',
      '.z123-presets{display:contents}',
      '.z123-preset{background:#1b2330;border:1px solid #2b3546;color:#c3ccda;border-radius:6px;',
      'padding:4px 9px;font-size:11px;cursor:pointer}',
      '.z123-hint{margin-left:auto;font-size:11px;color:#6b7788}',
      '.z123-sliders{display:flex;flex-wrap:wrap;gap:12px;align-items:center;width:100%;margin-bottom:2px}',
      '.z123-slider{display:flex;align-items:center;gap:6px;font-size:11px;color:#93a2b8}',
      '.z123-slider input{width:120px;accent-color:#38bdf8}',
      '.z123-slider span:last-child{font-variant-numeric:tabular-nums;color:#c9d2dd;min-width:34px;text-align:right}',
      /* 手机竖屏：全屏面板 + 加大触摸目标 */
      '@media (max-width:768px){',
      '.z123-mask{padding:0}',
      '.z123-panel{width:100%;height:100%;height:100dvh;border:none;border-radius:0}',
      '.z123-head{padding:calc(8px + env(safe-area-inset-top)) 10px 8px;gap:8px}',
      '.z123-num{font-size:11px}',
      '.z123-close{min-width:40px;height:36px;font-size:15px}',
      '.z123-stage{min-height:160px}',
      '.z123-foot{padding:8px 10px calc(10px + env(safe-area-inset-bottom));max-height:46vh;overflow-y:auto}',
      '.z123-presets{display:flex;gap:6px;width:100%;overflow-x:auto;padding-bottom:2px;',
      '-webkit-overflow-scrolling:touch}',
      '.z123-preset{flex:0 0 auto;padding:8px 12px;font-size:12px;min-height:36px}',
      '.z123-hint{margin-left:0;width:100%;font-size:10px}',
      '.z123-sliders{flex-direction:column;gap:4px}',
      '.z123-slider{width:100%}',
      '.z123-slider span:first-child{min-width:60px}',
      '.z123-slider input{flex:1;width:auto;height:32px}',
      '}',
      /* 矮屏（手机横屏 / 小窗口）：再压底栏，优先保住 3D 视口 */
      '@media (max-height:520px){',
      '.z123-mask{padding:0}',
      '.z123-panel{width:100%;height:100%;height:100dvh;border:none;border-radius:0}',
      '.z123-hint{display:none}',
      '.z123-foot{max-height:38vh}',
      '.z123-stage{min-height:120px}',
      '.z123-slider input{height:26px}',
      '.z123-preset{min-height:32px;padding:6px 10px}',
      '}',
    ].join('')
    document.head.appendChild(s)
  }

  /* =============== 上帝视角位姿视图 ===============
   * interactive=false：仅渲染小图（节点卡片内）
   * interactive=true ：左键拖拽球面移动相机（弹窗大图）
   */
  function createPoseViewer(opts) {
    opts = opts || {}
    var interactive = !!opts.interactive
    var onChange = opts.onChange || function () {}
    var seq = glSeq++

    var wrap = document.createElement('div')
    wrap.style.cssText = 'position:relative;width:100%;height:100%;overflow:hidden;background:#11141a;' +
      (interactive ? 'touch-action:none;cursor:grab' : '')

    var tip = document.createElement('div')
    tip.style.cssText = 'position:absolute;inset:0;display:flex;align-items:center;' +
      'justify-content:center;font-size:11px;color:#6f7b8c;pointer-events:none;' +
      'text-align:center;padding:8px;line-height:1.5'
    tip.textContent = '3D 视口初始化…'
    wrap.appendChild(tip)

    var renderer = null, scene = null, camera = null, controls = null
    var raf = 0, disposed = false, tries = 0
    var state = { az: 0, el: 0 }
    var planeMesh = null, icon = null, texture = null, textureUrl = null, pendingTexture = null
    var ro = null

    function renderOnce() { if (renderer && scene && !disposed) renderer.render(scene, camera) }
    function loop() {
      if (disposed) return
      raf = requestAnimationFrame(loop)
      if (controls) controls.update()
      if (renderer) renderer.render(scene, camera)
    }

    function placeIcon() {
      if (!icon) return
      var az = state.az * DEG, el = state.el * DEG
      var ce = Math.cos(el)
      icon.position.set(R * Math.sin(az) * ce, R * Math.sin(el), R * Math.cos(az) * ce)
      icon.lookAt(0, 0, 0) // 本地 +Z 指向球心
    }

    function resize() {
      if (!renderer || disposed) return
      var w = wrap.clientWidth || 1, hgt = wrap.clientHeight || 1
      renderer.setSize(w, hgt, false)
      camera.aspect = w / hgt
      camera.updateProjectionMatrix()
      if (!interactive) renderOnce()
    }

    function tryCreateRenderer() {
      // 属性降级：部分环境（软件渲染/老驱动）开 antialias 会创建失败
      var variants = [{ antialias: true }, { antialias: false }, {}]
      for (var i = 0; i < variants.length; i++) {
        try { return new THREE.WebGLRenderer(variants[i]) } catch (e) { /* 试下一种 */ }
      }
      return null
    }

    function buildScene() {
      renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, isTouchUI() ? 1.75 : 2))
      renderer.setClearColor(0x11141a, 1)
      renderer.domElement.style.cssText = 'display:block;width:100%;height:100%'
      wrap.insertBefore(renderer.domElement, tip)
      tip.style.display = 'none'

      scene = new THREE.Scene()
      // 外部观察相机（「上帝」位置）：初始 3/4 俯视
      camera = new THREE.PerspectiveCamera(42, 1, 0.05, 100)
      camera.position.set(2.1, 1.55, 2.7)

      scene.add(new THREE.HemisphereLight(0xffffff, 0x223047, 1.15))
      var dirLight = new THREE.DirectionalLight(0xffffff, 0.75)
      dirLight.position.set(3, 4, 2)
      scene.add(dirLight)

      // 球面线框 + 极淡实心球（体积感）
      scene.add(new THREE.Mesh(
        new THREE.SphereGeometry(R, 20, 12),
        new THREE.MeshBasicMaterial({ color: 0x33415a, wireframe: true, transparent: true, opacity: 0.55 })
      ))
      scene.add(new THREE.Mesh(
        new THREE.SphereGeometry(R * 0.995, 32, 20),
        new THREE.MeshBasicMaterial({ color: 0x181e28, transparent: true, opacity: 0.62 })
      ))

      // 赤道环（强化水平面）
      var eqPts = []
      for (var i = 0; i <= 96; i++) {
        var t = i / 96 * Math.PI * 2
        eqPts.push(new THREE.Vector3(Math.sin(t) * R, 0, Math.cos(t) * R))
      }
      var eqGeo = new THREE.BufferGeometry().setFromPoints(eqPts)
      scene.add(new THREE.Line(eqGeo, new THREE.LineBasicMaterial({
        color: 0x506480, transparent: true, opacity: 0.9,
      })))

      // 球心参考图（双面贴片）
      planeMesh = new THREE.Mesh(
        new THREE.PlaneGeometry(1.15, 1.15),
        new THREE.MeshBasicMaterial({ color: 0x39445a, side: THREE.DoubleSide })
      )
      scene.add(planeMesh)

      // 方位标签：正面 0°(+Z) / 右 90°(+X) / 背面 180°(-Z) / 左 -90°(-X)
      var lf = makeLabel('0°', '#34d399'); lf.position.set(0, 0.15, R * 1.04)
      var lr = makeLabel('90°'); lr.position.set(R * 1.04, 0.15, 0)
      var lb = makeLabel('180°'); lb.position.set(0, 0.15, -R * 1.04)
      var ll = makeLabel('-90°'); ll.position.set(-R * 1.04, 0.15, 0)
      scene.add(lf); scene.add(lr); scene.add(lb); scene.add(ll)

      // 球面上的相机图标：机身 + 镜头 + 视野锥（+Z 朝球心）
      icon = new THREE.Group()
      icon.add(new THREE.Mesh(
        new THREE.BoxGeometry(0.26, 0.2, 0.3),
        new THREE.MeshBasicMaterial({ color: 0x38bdf8 })
      ))
      var lens = new THREE.Mesh(
        new THREE.CylinderGeometry(0.075, 0.115, 0.2, 16),
        new THREE.MeshBasicMaterial({ color: 0xf8fafc })
      )
      lens.rotation.x = Math.PI / 2 // 圆柱默认沿 Y → 沿 Z
      lens.position.z = 0.22        // +Z 侧 = 朝球心
      icon.add(lens)
      var frustum = new THREE.Mesh(
        new THREE.ConeGeometry(0.5, R - 0.3, 20, 1, true),
        new THREE.MeshBasicMaterial({
          color: 0x38bdf8, transparent: true, opacity: 0.14,
          side: THREE.DoubleSide, depthWrite: false,
        })
      )
      frustum.rotation.x = -Math.PI / 2      // 锥尖转向 -Z（相机侧）
      frustum.position.z = (R - 0.3) / 2 + 0.12 // 底面朝球心
      icon.add(frustum)
      scene.add(icon)

      controls = new THREE.OrbitControls(camera, renderer.domElement)
      controls.enableDamping = true
      controls.dampingFactor = 0.11
      controls.minDistance = 2.2
      controls.maxDistance = 9
      controls.target.set(0, 0, 0)
      controls.enablePan = false
      // 左键留给「拖拽相机」（自己处理），右键旋转观察，滚轮缩放
      controls.mouseButtons = { LEFT: null, MIDDLE: THREE.MOUSE.DOLLY, RIGHT: THREE.MOUSE.ROTATE }
      // 触屏：单指留给拖拽相机（自己处理），双指同时旋转观察 + 缩放
      controls.touches = { ONE: null, TWO: THREE.TOUCH.DOLLY_ROTATE }
      // 小图只读：关闭 OrbitControls，否则它会抢走指针并阻止 click 冒泡
      controls.enabled = interactive
      controls.update()

      if (typeof ResizeObserver !== 'undefined') {
        ro = new ResizeObserver(resize); ro.observe(wrap)
      }
      resize()
      placeIcon()
      if (pendingTexture) applyTexture(pendingTexture)
      if (interactive) loop()
      else renderOnce()
    }

    function tryInit() {
      if (disposed || renderer) return
      var r = tryCreateRenderer()
      if (!r) {
        tries++
        if (tries <= 3) {
          tip.textContent = '3D 视口初始化…（重试 ' + tries + '/3）'
          setTimeout(tryInit, 350 * tries)
        } else {
          tip.textContent = '此环境不支持 WebGL，3D 预览不可用（仍可用预设按钮改角度）'
        }
        return
      }
      renderer = r
      try {
        buildScene()
      } catch (e) {
        renderer = null
        tip.style.display = 'flex'
        tip.textContent = '3D 初始化失败：' + (e && e.message)
      }
    }

    /* ---------- 左键拖拽球面：移动相机 ---------- */
    var raycaster = new THREE.Raycaster()
    var sphereMath = new THREE.Sphere(new THREE.Vector3(0, 0, 0), R)
    var dragging = false
    var hitTmp = new THREE.Vector3()

    function ndc(e) {
      var rect = renderer.domElement.getBoundingClientRect()
      return new THREE.Vector2(
        ((e.clientX - rect.left) / rect.width) * 2 - 1,
        -((e.clientY - rect.top) / rect.height) * 2 + 1
      )
    }

    function applyHit() {
      var az = Math.atan2(hitTmp.x, hitTmp.z) * RAD
      var el = Math.asin(clamp(hitTmp.y / R, -1, 1)) * RAD
      state.az = normAz(Math.round(az))
      state.el = Math.round(clamp(el, -89, 89))
      placeIcon()
      renderOnce()
      onChange(state.az, state.el, false) // 拖动中只刷新读数，松手才写回
    }

    function onPointerDown(e) {
      if (!renderer || (e.button !== undefined && e.button !== 0)) return
      var touch = e.pointerType === 'touch' || e.pointerType === 'pen'
      if (touch && !e.isPrimary) {
        // 第二根手指：结束单指拖拽，把双指手势让给 OrbitControls（旋转+缩放）
        if (dragging) { dragging = false; wrap.style.cursor = 'grab' }
        return
      }
      raycaster.setFromCamera(ndc(e), camera)
      if (!raycaster.ray.intersectSphere(sphereMath, hitTmp)) return // 球外空白 → 不处理
      dragging = true
      wrap.style.cursor = 'grabbing'
      try { renderer.domElement.setPointerCapture(e.pointerId) } catch (err) { /* ignore */ }
      applyHit()
      // 触屏不阻断冒泡：OrbitControls 需同时收到两指事件才能做双指手势
      if (!touch) {
        e.preventDefault()
        e.stopPropagation()
      }
    }

    function onPointerMove(e) {
      if (!dragging || !renderer) return
      raycaster.setFromCamera(ndc(e), camera)
      if (!raycaster.ray.intersectSphere(sphereMath, hitTmp)) return
      applyHit()
      if (e.pointerType !== 'touch' && e.pointerType !== 'pen') {
        e.preventDefault()
        e.stopPropagation()
      }
    }

    function onPointerUp(e) {
      if (!dragging) return
      dragging = false
      wrap.style.cursor = 'grab'
      try { renderer.domElement.releasePointerCapture(e.pointerId) } catch (err) { /* ignore */ }
      onChange(state.az, state.el, true) // 松手写回参数
    }

    if (interactive) {
      wrap.addEventListener('pointerdown', onPointerDown, true)
      wrap.addEventListener('pointermove', onPointerMove, true)
      wrap.addEventListener('pointerup', onPointerUp, true)
      wrap.addEventListener('pointercancel', onPointerUp, true)
    }

    /* ---------- 对外 API ---------- */
    function setAngles(az, el) {
      state.az = normAz(Math.round(num(az, 0)))
      state.el = Math.round(clamp(num(el, 0), -89, 89))
      placeIcon()
      renderOnce()
      onChange(state.az, state.el, false)
    }

    function getAngles() { return { az: state.az, el: state.el } }

    function applyTexture(url) {
      new THREE.TextureLoader().load(url, function (tex) {
        if (disposed) { tex.dispose(); return }
        if (texture) texture.dispose()
        texture = tex
        if (!planeMesh) return
        planeMesh.material.map = tex
        planeMesh.material.color.set(0xffffff)
        planeMesh.material.needsUpdate = true
        var iw = tex.image && tex.image.width, ih = tex.image && tex.image.height
        if (iw && ih) {
          var h = 1.15 * ih / iw
          planeMesh.geometry.dispose()
          planeMesh.geometry = new THREE.PlaneGeometry(1.15, Math.min(h, 1.5))
        }
        renderOnce()
      }, undefined, function () { /* 加载失败：保留灰色占位，不影响调角度 */ })
    }

    function setTextureUrl(url) {
      if (!url || url === textureUrl) return
      textureUrl = url
      pendingTexture = url
      if (renderer) applyTexture(url)
    }

    function dispose() {
      disposed = true
      cancelAnimationFrame(raf)
      if (ro) ro.disconnect()
      if (interactive) {
        wrap.removeEventListener('pointerdown', onPointerDown, true)
        wrap.removeEventListener('pointermove', onPointerMove, true)
        wrap.removeEventListener('pointerup', onPointerUp, true)
        wrap.removeEventListener('pointercancel', onPointerUp, true)
      }
      if (renderer) {
        if (controls) controls.dispose()
        if (scene) {
          scene.traverse(function (o) {
            if (o.geometry) o.geometry.dispose()
            if (o.material) {
              if (o.material.map && o.material.map.dispose) o.material.map.dispose()
              o.material.dispose()
            }
          })
        }
        renderer.dispose()
        if (renderer.forceContextLoss) renderer.forceContextLoss()
      }
      wrap.innerHTML = ''
    }

    // 错峰初始化：同包多实例并发建 context 在软件渲染环境易失败
    setTimeout(tryInit, seq * 160)

    return {
      el: wrap,
      setAngles: setAngles,
      getAngles: getAngles,
      setTextureUrl: setTextureUrl,
      dispose: dispose,
    }
  }

  /* =============== 节点卡片内的组件 =============== */
  function mount(el, props) {
    el.style.cssText += ';display:flex;flex-direction:column;overflow:hidden;background:#14171c;' +
      'border-radius:8px;position:relative'

    var cur = props || {}
    var viewer = null, modal = null
    var curAz = normAz(Math.round(num(cur.params && cur.params.azimuth, 0)))
    var curEl = Math.round(clamp(num(cur.params && cur.params.elevation, 0), -89, 89))

    // ---- 顶部状态条 ----
    var status = document.createElement('div')
    status.style.cssText = 'padding:4px 8px;font-size:11px;color:#9aa4b2;white-space:nowrap;' +
      'overflow:hidden;text-overflow:ellipsis'
    status.textContent = '点击节点展开调相机'

    // ---- 上帝视角小图（只读）----
    var poseBox = document.createElement('div')
    poseBox.style.cssText = 'position:relative;flex:1 1 auto;min-height:120px;'

    // ---- 角度读数 ----
    var readout = document.createElement('div')
    readout.style.cssText = 'padding:4px 8px;font-size:11px;color:#c9d2dd;display:flex;' +
      'flex-wrap:wrap;gap:4px 10px;align-items:center;justify-content:space-between'
    var azLabel = document.createElement('span')
    var elLabel = document.createElement('span')
    azLabel.style.whiteSpace = elLabel.style.whiteSpace = 'nowrap'
    var expandBtn = document.createElement('span')
    expandBtn.textContent = '⛶ 调整'
    expandBtn.style.cssText = 'color:#60a5fa;cursor:pointer;user-select:none;' +
      'padding:2px 6px;border-radius:6px;background:rgba(96,165,250,.1);white-space:nowrap'
    readout.appendChild(azLabel); readout.appendChild(elLabel); readout.appendChild(expandBtn)

    // ---- 结果图（只展示生成结果）----
    var resBox = document.createElement('div')
    resBox.style.cssText = 'display:none;padding:0 8px 6px'
    var resImg = document.createElement('img')
    resImg.style.cssText = 'display:block;margin:0 auto;max-width:100%;max-height:190px;' +
      'width:auto;height:auto;border-radius:4px;background:#0b1220'
    resBox.appendChild(resImg)

    el.appendChild(status); el.appendChild(poseBox); el.appendChild(readout); el.appendChild(resBox)

    function paintReadout(az, el) {
      azLabel.textContent = 'azimuth ' + az + '°'
      elLabel.textContent = 'elevation ' + el + '°'
    }

    function writeParams(az, el) {
      curAz = az; curEl = el
      if (!cur.onParamsChange) return
      var p = Object.assign({}, cur.params || {})
      p.azimuth = String(az)
      p.elevation = String(el)
      cur.onParamsChange(p)
    }

    viewer = createPoseViewer({
      interactive: false,
      onChange: function (az, el) { paintReadout(az, el) },
    })
    poseBox.appendChild(viewer.el)
    viewer.setAngles(curAz, curEl)

    function inputImageUrl() {
      var refPath = cur.params && cur.params.ref_image_path
      if (refPath && typeof refPath === 'string' && refPath.trim() !== '') {
        return mediaUrl(refPath.trim())
      }
      var pinned = cur.media && cur.media.input
      if (pinned) return pinned
      var resolved = (cur.outputs && cur.outputs.__inputs_resolved) || {}
      var ref = resolved.image
      var inId = ref && (typeof ref === 'string' ? ref : (ref.$id || ref.id || ''))
      if (inId && cur.execution && cur.execution.id && cur.nodeId) {
        return '/api/v1/executions/' + cur.execution.id + '/nodes/' +
          encodeURIComponent(cur.nodeId) + '/service-proxy?path=' +
          encodeURIComponent('/images/' + inId) + '&method=GET'
      }
      return null
    }

    /* ---------- 弹窗：大尺寸上帝视角 ---------- */
    function openModal() {
      if (modal) return
      ensureStyle()
      var mobile = isTouchUI()
      var savedOverflow = document.body.style.overflow
      document.body.style.overflow = 'hidden'

      var mask = document.createElement('div')
      mask.className = 'z123-mask'
      var panel = document.createElement('div')
      panel.className = 'z123-panel'

      var head = document.createElement('div')
      head.className = 'z123-head'
      var title = document.createElement('div')
      title.className = 'z123-title'
      title.textContent = '相机位姿'
      var mAz = document.createElement('span'); mAz.className = 'z123-num'
      var mEl = document.createElement('span'); mEl.className = 'z123-num'
      var spacer = document.createElement('div'); spacer.className = 'z123-spacer'
      var closeBtn = document.createElement('button')
      closeBtn.className = 'z123-close'
      closeBtn.textContent = '✕'
      head.appendChild(title); head.appendChild(mAz); head.appendChild(mEl)
      head.appendChild(spacer); head.appendChild(closeBtn)

      var stage = document.createElement('div')
      stage.className = 'z123-stage'

      var foot = document.createElement('div')
      foot.className = 'z123-foot'

      // 滑块兜底：WebGL 不可用（或需要精确数值）时仍能调角度，与球面拖拽双向同步
      var slideRow = document.createElement('div')
      slideRow.className = 'z123-sliders'

      function mkSlider(labelText, min, max, step, val, onInput) {
        var box = document.createElement('label')
        box.className = 'z123-slider'
        var name = document.createElement('span')
        name.textContent = labelText
        var inp = document.createElement('input')
        inp.type = 'range'; inp.min = String(min); inp.max = String(max)
        inp.step = String(step)
        inp.value = String(val)
        var out = document.createElement('span')
        out.textContent = val + '°'
        inp.addEventListener('input', function () {
          var v = parseInt(inp.value, 10) || 0
          out.textContent = v + '°'
          onInput(v)
        })
        // 阻止拖滑块时冒泡到画布
        inp.addEventListener('pointerdown', function (e) { e.stopPropagation() })
        inp.addEventListener('click', function (e) { e.stopPropagation() })
        box.appendChild(name); box.appendChild(inp); box.appendChild(out)
        return { box: box, set: function (v) { inp.value = String(v); out.textContent = v + '°' } }
      }

      var azSlider = mkSlider('azimuth', -180, 180, 5, curAz, function (v) {
        big.setAngles(v, big.getAngles().el)
        writeParams(v, big.getAngles().el)
      })
      var elSlider = mkSlider('elevation', -89, 89, 1, curEl, function (v) {
        big.setAngles(big.getAngles().az, v)
        writeParams(big.getAngles().az, v)
      })
      slideRow.appendChild(azSlider.box); slideRow.appendChild(elSlider.box)

      var presetRow = document.createElement('div')
      presetRow.className = 'z123-presets'

      var hint = document.createElement('div')
      hint.className = 'z123-hint'
      hint.textContent = mobile
        ? (cur.onParamsChange ? '单指拖球面移相机 · 双指转视角' : '只读：回放态不可改参数')
        : (cur.onParamsChange
          ? '左键拖球面移相机 · 右键旋转观察 · 滚轮缩放 · 松手写回参数'
          : '只读：回放态不可改参数（回编辑态可调）')

      panel.appendChild(head); panel.appendChild(stage); panel.appendChild(foot)

      var big = createPoseViewer({
        interactive: true,
        onChange: function (az, el, commit) {
          mAz.textContent = 'azimuth ' + az + '°'
          mEl.textContent = 'elevation ' + el + '°'
          paintReadout(az, el)
          viewer.setAngles(az, el) // 大图拖拽时小图同步
          azSlider.set(az); elSlider.set(el)
          if (commit) writeParams(az, el)
        },
      })
      stage.appendChild(big.el)
      big.setAngles(curAz, curEl)
      big.setTextureUrl(inputImageUrl())
      mAz.textContent = 'azimuth ' + curAz + '°'
      mEl.textContent = 'elevation ' + curEl + '°'

      var presets = [
        ['正面', 0, 0], ['右 90°', 90, 0], ['背面', 180, 0], ['左 -90°', -90, 0],
        ['俯视 45°', 0, 45], ['仰视 -30°', 0, -30],
      ]
      presets.forEach(function (p) {
        var b = document.createElement('button')
        b.className = 'z123-preset'
        b.textContent = p[0]
        b.onclick = function (e) {
          e.stopPropagation()
          big.setAngles(p[1], p[2])
          mAz.textContent = 'azimuth ' + p[1] + '°'
          mEl.textContent = 'elevation ' + p[2] + '°'
          azSlider.set(p[1]); elSlider.set(p[2])
          viewer.setAngles(p[1], p[2])
          writeParams(p[1], p[2])
        }
        presetRow.appendChild(b)
      })
      foot.appendChild(presetRow)
      foot.appendChild(hint)
      foot.appendChild(slideRow)

      function close() {
        if (!modal) return
        document.removeEventListener('keydown', onKey)
        big.dispose()
        if (mask.parentNode) mask.parentNode.removeChild(mask)
        document.body.style.overflow = savedOverflow
        modal = null
      }
      function onKey(e) { if (e.key === 'Escape') close() }

      closeBtn.onclick = function (e) { e.stopPropagation(); close() }
      mask.onclick = function (e) { if (e.target === mask) close() }
      panel.onclick = function (e) { e.stopPropagation() }
      document.addEventListener('keydown', onKey)

      document.body.appendChild(mask)
      mask.appendChild(panel)
      modal = {
        close: close,
        refresh: function () {
          big.setAngles(curAz, curEl)
          big.setTextureUrl(inputImageUrl())
        },
      }
    }

    function openIfAllowed(e) {
      if (e) { e.stopPropagation(); e.preventDefault() }
      openModal()
    }
    el.addEventListener('click', openIfAllowed)
    expandBtn.addEventListener('click', openIfAllowed)

    var resLoadedFor = null
    function loadResult(url) {
      if (!url || url === resLoadedFor) return
      resLoadedFor = url
      resBox.style.display = 'block'
      resImg.src = url
    }

    function update(p) {
      cur = p || {}
      if (cur.status === 'running') status.textContent = '生成中…'
      var az = normAz(Math.round(num(cur.params && cur.params.azimuth, 0)))
      var el = Math.round(clamp(num(cur.params && cur.params.elevation, 0), -89, 89))
      if (az !== curAz || el !== curEl) {
        curAz = az
        curEl = el
        viewer.setAngles(curAz, curEl)
        if (modal) modal.refresh()
      }
      var texUrl = inputImageUrl()
      if (texUrl) {
        viewer.setTextureUrl(texUrl)
        if (modal) modal.refresh()
      }
      var prev = cur.preview && cur.preview.url
      if (prev) loadResult(prev)
    }

    function unmount() {
      if (modal) { modal.close(); modal = null }
      el.removeEventListener('click', openIfAllowed)
      if (viewer) { viewer.dispose(); viewer = null }
    }

    paintReadout(curAz, curEl)
    update(props)
    return { update: update, unmount: unmount }
  }

  if (window.FlowXNodeWidget && window.FlowXNodeWidget.define) {
    window.FlowXNodeWidget.define(mount)
  }
})()
