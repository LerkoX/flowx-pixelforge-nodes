"""image-person-mask Mock：中间竖带灰度蒙版。"""
from flowx_client import emit


def main():
    emit(mask="", count=1, coverage=0.2, width=832, height=1488)


if __name__ == "__main__":
    main()
