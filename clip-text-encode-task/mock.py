"""clip-text-encode-task mock：模拟编码，输出伪 conditioning_id。"""
from flowx_client import emit, param

task = param("task", "none")
print(f"[clip-text-encode-task][mock] task={task} (encode skipped)")
emit(conditioning="mock-cond-task")
