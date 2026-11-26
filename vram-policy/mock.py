"""vram-policy Mock。"""
from flowx_client import emit


def main():
    emit(tr_offload="sequential", tr_group_blocks=0,
         te_offload="sequential", te_group_blocks=0,
         policy="saver(mock)")


if __name__ == "__main__":
    main()
