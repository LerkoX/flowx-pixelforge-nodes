import os
from flowx_client import emit, param


def main():
    keys = sorted(k for k in os.environ
                  if k.startswith(("FLOWX_PARAM_", "SERVICE_", "FLOWX_EXECUTION")))
    lines = [f"{k}={os.environ[k]}" for k in keys]
    with open(param("out_file"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines), flush=True)
    emit(ok="done")


if __name__ == "__main__":
    main()
