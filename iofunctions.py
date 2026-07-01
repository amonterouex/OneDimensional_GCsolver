from dataclasses import dataclass


@dataclass
class InData:
    mu: float = 0.0
    temp: float = 0.0
    pot_file: str = ""
    eps: float = 0.0
    r0: float = 0.0
    type: str = ""
    out_file: str = ""


def read_parameters(filename: str, indata: InData) -> None:
    parameters: dict[str, str] = {}
    with open(filename, "r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if ":" not in line:
                raise ValueError(f"Wrong format in line {line_number}: {line!r}")
            key, value = line.split(":", 1)
            parameters[key.strip()] = value.strip()

    required = {"mu", "temp", "pot_file", "eps", "r0", "type", "out_file"}
    missing = required - parameters.keys()
    if missing:
        raise ValueError("Missing parameters: " + ", ".join(sorted(missing)))

    indata.mu = float(parameters["mu"])
    indata.temp = float(parameters["temp"])
    indata.pot_file = parameters["pot_file"]
    indata.eps = float(parameters["eps"])
    indata.r0 = float(parameters["r0"])
    indata.type = parameters["type"]
    indata.out_file = parameters["out_file"]
