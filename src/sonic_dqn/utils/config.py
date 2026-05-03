import yaml
from pathlib import Path


def load_config(path: str, overrides: list[str] | None = None) -> dict:
    with open(path) as f:
        cfg = yaml.safe_load(f)

    for override in overrides or []:
        key, _, value = override.partition("=")
        parts = key.strip().split(".")
        node = cfg
        for part in parts[:-1]:
            node = node[part]
        # try to cast to int/float/bool before storing as string
        raw = value.strip()
        if raw.lower() == "true":
            node[parts[-1]] = True
        elif raw.lower() == "false":
            node[parts[-1]] = False
        else:
            try:
                node[parts[-1]] = int(raw)
            except ValueError:
                try:
                    node[parts[-1]] = float(raw)
                except ValueError:
                    node[parts[-1]] = raw

    return cfg