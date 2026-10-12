"""image-batch-save mock：不访问服务，验证参数注入与输出提取。"""
from flowx_client import emit, param

print("[image-batch-save][mock] images=%s dir=%s prefix=%s"
      % (str(param("images"))[:40], param("output_dir", "~/flowx-e2e-out"),
         param("filename_prefix", "batch")))
emit(count=2, paths='@["/tmp/batch_001.png", "/tmp/batch_002.png"]',
     filenames='@["batch_001.png", "batch_002.png"]', dir="/tmp")
