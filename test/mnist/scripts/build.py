from pathlib import Path
import subprocess
from typing import Literal
import shutil
import argparse

WASM_OPT_FLAGS = [
    "--enable-gc",
    "--enable-multivalue",
    "--enable-reference-types",
    "--enable-bulk-memory",
    "--enable-simd",
]

Target = Literal["wasm", "wasm-gc"]
Profile = Literal["release", "debug"]
Encoding = Literal["utf8", "utf16"]

MOON_FLAGS = []


class InvalidSourceError(Exception):
    pass


def moon_build(
    target: Target, profile: Profile, package: str, flags: list[str] = MOON_FLAGS
) -> Path:
    args = ["moon", "build", "--target", target]
    if profile == "debug":
        args.append("--debug")
    args.extend(flags)
    subprocess.run(args, check=True)
    return Path("target") / target / profile / "build" / package / f"{package}.wasm"


def wasm_tools_component_embed(
    wit: Path,
    wasm: Path,
    encoding: Encoding = "utf16",
) -> Path:
    if not wasm.name.endswith(".wasm"):
        raise InvalidSourceError("Wasm file must have .wasm extension")
    destination = wasm.with_name(
        wasm.name.removesuffix(".wasm") + f".{encoding}.embedded.wasm"
    )
    args: list[str] = [
        "wasm-tools",
        "component",
        "embed",
        str(wit),
        str(wasm),
        "-o",
        str(destination),
    ]
    if encoding == "utf16":
        args.append("--encoding")
        args.append("utf16")
    subprocess.run(
        args,
        check=True,
    )
    return destination


def wasm_tools_component_new(
    embedded_wasm: Path,
) -> Path:
    destination = embedded_wasm.with_name(
        embedded_wasm.name.removesuffix(".embedded.wasm") + ".component.wasm"
    )
    subprocess.run(
        [
            "wasm-tools",
            "component",
            "new",
            embedded_wasm,
            "-o",
            destination,
        ],
        check=True,
    )
    return destination


def wasm_opt(wasm: Path, flags: list[str] = WASM_OPT_FLAGS) -> Path:
    destination = wasm.with_name(wasm.name.removesuffix(".wasm") + ".opt.wasm")
    subprocess.run(
        [
            "wasm-opt",
            *flags,
            wasm,
            "-O3",
            "-o",
            destination,
        ],
        check=True,
    )
    return destination


def wasm_tools_print(wasm: Path) -> Path:
    destination = wasm.with_name(wasm.name.removesuffix(".wasm") + ".wat")
    subprocess.run(
        [
            "wasm-tools",
            "print",
            wasm,
            "-o",
            destination,
        ],
        check=True,
    )
    return destination


JCO = ["corepack", "pnpm", "exec", "jco"]
JCO_FLAGS = ["--no-nodejs-compat"]


def jco_transpile(
    wasm: Path, outdir: Path, name: str, flags: list[str] = JCO_FLAGS
) -> tuple[Path, Path, Path]:
    destinations = (
        outdir / f"{name}.js",
        outdir / f"{name}.d.ts",
        outdir / f"{name}.core.wasm",
    )
    subprocess.run(
        [
            *JCO,
            "transpile",
            *flags,
            wasm,
            "-o",
            outdir,
            "--name",
            name,
        ],
        check=True,
    )
    return destinations


def wasmtime_py_bindgen(wasm: Path, out_dir: Path) -> Path:
    subprocess.run(
        [
            "python3",
            "-m",
            "wasmtime.bindgen",
            wasm,
            "--out-dir",
            out_dir,
        ],
        check=True,
    )
    return out_dir


def component_embed_new_transpile(
    wit: Path, wasm: Path, outdir: Path, name: str
) -> tuple[Path, Path, Path]:
    embedded_wasm = wasm_tools_component_embed(wit, wasm)
    component_wasm = wasm_tools_component_new(embedded_wasm)
    return jco_transpile(component_wasm, outdir, name)


def web_wasm_release():
    wit = Path("wit")
    wasm = moon_build("wasm", "release", "gen")
    opt_wasm = wasm_opt(wasm)
    component_embed_new_transpile(wit, opt_wasm, Path("web"), "mnist-model")


def web_wasm_debug():
    wit = Path("wit")
    wasm = moon_build("wasm", "debug", "gen")
    component_embed_new_transpile(wit, wasm, Path("web"), "mnist-model")


def web_wasm_gc_release():
    wit = Path("wit")
    wasm = moon_build("wasm", "release", "gen")
    _, _, core_wasm = component_embed_new_transpile(
        wit, wasm, Path("web"), "mnist-model"
    )
    wasm_gc = moon_build("wasm-gc", "release", "gen")
    opt_wasm_gc = wasm_opt(wasm_gc)
    shutil.copy(opt_wasm_gc, core_wasm)


def web_wasm_gc_debug():
    wit = Path("wit")
    wasm = moon_build("wasm", "release", "gen")
    _, _, core_wasm = component_embed_new_transpile(
        wit, wasm, Path("web"), "mnist-model"
    )
    wasm_gc = moon_build("wasm-gc", "debug", "gen")
    shutil.copy(wasm_gc, core_wasm)


def cli_wasm(profile: Profile):
    wit = Path("wit")
    wasm = moon_build("wasm", profile, "gen")
    if profile == "release":
        wasm = wasm_opt(wasm)
    embedded_wasm = wasm_tools_component_embed(wit, wasm, "utf8")
    component_wasm = wasm_tools_component_new(embedded_wasm)
    return wasmtime_py_bindgen(component_wasm, Path("cli", "mnist", "model"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--target",
        choices=["wasm", "wasm-gc"],
        default="wasm",
        help="Target to build",
    )
    parser.add_argument(
        "--profile",
        choices=["release", "debug"],
        default="release",
        help="Profile to build",
    )
    parser.add_argument(
        "out",
        choices=["web", "cli"],
        default="web",
        help="Output format",
    )
    args = parser.parse_args()

    if args.out == "cli":
        if args.target == "wasm":
            cli_wasm(args.profile)
        else:
            raise ValueError("CLI target must be wasm")
    elif args.target == "wasm":
        if args.profile == "release":
            web_wasm_release()
        else:
            web_wasm_debug()
    else:
        if args.profile == "release":
            web_wasm_gc_release()
        else:
            web_wasm_gc_debug()


if __name__ == "__main__":
    main()
