"""load-audio mock：不访问服务，验证参数注入与输出形态。"""
from flowx_client import emit, param

_path = param("audio_path", "/tmp/mock.wav")
print("[load-audio][mock] audio_path=%s" % _path)
emit(audio="mock-audio-obj", seconds=5.0, sample_rate=32000)
