/* ===== tripo-sr widget：GLB 预览（three.js r128 vendored，离线可用）=====
 * 数据源：outputs.mesh（MESH 对象 ID）→ 经 Studio 节点级 service-proxy 拉
 * /ext/meshes/{oid}（GLB 字节流，插件契约 v2 端点）→ GLTFLoader.parse 渲染。
 * 媒体本体不进 outputs/metadata（遵守媒体传递纪律，仅 oid 引用）。
 */
(function () {
  function mount(el, props) {
    el.style.cssText += ';display:flex;flex-direction:column;overflow:hidden;background:#14171c;border-radius:8px'
    var status = document.createElement('div')
    status.style.cssText = 'padding:4px 8px;font-size:11px;color:#9aa4b2;white-space:nowrap;overflow:hidden;text-overflow:ellipsis'
    status.textContent = '等待执行…'
    var canvasBox = document.createElement('div')
    canvasBox.style.cssText = 'flex:1;min-height:0;position:relative'
    var bar = document.createElement('div')
    bar.style.cssText = 'display:flex;gap:6px;padding:4px 8px;align-items:center'
    var reloadBtn = document.createElement('button')
    reloadBtn.textContent = '⟳ 重载'
    var dlBtn = document.createElement('a')
    dlBtn.textContent = '⬇ 下载'
    dlBtn.style.cssText = 'font-size:11px;color:#7cc4ff;text-decoration:none;cursor:pointer;display:none'
    reloadBtn.style.cssText = 'font-size:11px;background:#2a2f38;color:#c9d2dd;border:1px solid #3a4150;border-radius:4px;padding:2px 8px;cursor:pointer'
    bar.appendChild(reloadBtn); bar.appendChild(dlBtn)
    el.appendChild(status); el.appendChild(canvasBox); el.appendChild(bar)

    var cur = props || {}
    var renderer = null, scene = null, camera = null, controls = null, raf = 0
    var loadedFor = null, loading = false, autoSpin = true

    function proxyUrl(path) {
      if (!cur.execution || !cur.execution.id || !cur.nodeId) return null
      return '/api/v1/executions/' + cur.execution.id + '/nodes/' +
        encodeURIComponent(cur.nodeId) + '/service-proxy?path=' +
        encodeURIComponent(path) + '&method=GET'
    }

    function ensureGL() {
      if (renderer) return true
      try {
        renderer = new THREE.WebGLRenderer({ antialias: true })
        renderer.setClearColor(0x14171c)
        canvasBox.appendChild(renderer.domElement)
        scene = new THREE.Scene()
        camera = new THREE.PerspectiveCamera(45, 1, 0.01, 100)
        camera.position.set(1.2, 0.9, 1.6)
        var hemi = new THREE.HemisphereLight(0xffffff, 0x334, 1.0)
        var dir = new THREE.DirectionalLight(0xffffff, 0.8)
        dir.position.set(2, 3, 1.5)
        scene.add(hemi); scene.add(dir)
        scene.add(new THREE.GridHelper(2, 10, 0x2a2f38, 0x22262e))
        controls = new THREE.OrbitControls(camera, renderer.domElement)
        controls.enableDamping = true
        controls.addEventListener('start', function () { autoSpin = false })
        resize()
        animate()
        return true
      } catch (e) {
        status.textContent = 'WebGL 不可用: ' + e.message
        return false
      }
    }

    function resize() {
      if (!renderer) return
      var w = canvasBox.clientWidth || 300, hgt = canvasBox.clientHeight || 240
      renderer.setSize(w, hgt)
      camera.aspect = w / hgt
      camera.updateProjectionMatrix()
    }

    function animate() {
      raf = requestAnimationFrame(animate)
      if (!renderer) return
      if (autoSpin && scene && scene.userData.mesh) {
        scene.userData.mesh.rotation.y += 0.008
      }
      if (controls) controls.update()
      renderer.render(scene, camera)
    }

    function clearMesh() {
      var old = scene && scene.userData.mesh
      if (old) {
        scene.remove(old)
        old.traverse(function (o) {
          if (o.geometry) o.geometry.dispose()
          if (o.material) (Array.isArray(o.material) ? o.material : [o.material]).forEach(function (m) { m.dispose() })
        })
        scene.userData.mesh = null
      }
    }

    function loadMesh(oid) {
      var url = proxyUrl('/ext/meshes/' + oid)
      if (!url) { status.textContent = '回放态缺执行上下文，无法代理拉取'; return }
      if (!ensureGL()) return
      loading = true
      status.textContent = '拉取网格 ' + oid.slice(0, 8) + '…'
      fetch(url).then(function (r) {
        if (!r.ok) throw new Error('HTTP ' + r.status + '（对象可能已过期）')
        return r.arrayBuffer()
      }).then(function (buf) {
        new THREE.GLTFLoader().parse(buf, '', function (gltf) {
          clearMesh()
          var obj = gltf.scene
          // 居中 + 归一化到单位尺度
          var box = new THREE.Box3().setFromObject(obj)
          var size = new THREE.Vector3(); box.getSize(size)
          var center = new THREE.Vector3(); box.getCenter(center)
          var s = 1.2 / (Math.max(size.x, size.y, size.z) || 1)
          obj.scale.setScalar(s)
          obj.position.sub(center.multiplyScalar(s))
          scene.add(obj)
          scene.userData.mesh = obj
          var kb = (buf.byteLength / 1024).toFixed(0)
          status.textContent = '✅ ' + oid.slice(0, 8) + ' · ' + kb + ' KB（拖动旋转/滚轮缩放）'
          dlBtn.href = url; dlBtn.download = 'mesh_' + oid.slice(0, 8) + '.glb'
          dlBtn.style.display = ''
          loadedFor = oid; loading = false
        }, function (e) {
          status.textContent = 'GLB 解析失败: ' + (e.message || e)
          loading = false
        })
      }).catch(function (e) {
        status.textContent = '⚠️ ' + e.message
        loading = false
      })
    }

    reloadBtn.onclick = function () {
      var oid = cur.outputs && cur.outputs.mesh
      if (oid) { loadedFor = null; loadMesh(oid) }
    }

    function update(p) {
      cur = p || {}
      resize()
      var oid = cur.outputs && cur.outputs.mesh
      if (oid && oid !== loadedFor && !loading) loadMesh(oid)
      if (!oid && cur.status === 'running') status.textContent = '重建中…（完成后自动加载预览）'
    }

    function unmount() {
      cancelAnimationFrame(raf)
      clearMesh()
      if (renderer) { renderer.dispose(); renderer.forceContextLoss && renderer.forceContextLoss() }
      renderer = scene = camera = controls = null
    }

    update(props)
    return { update: update, unmount: unmount }
  }

  if (window.FlowXNodeWidget && window.FlowXNodeWidget.define) {
    window.FlowXNodeWidget.define(mount)
  }
})()
