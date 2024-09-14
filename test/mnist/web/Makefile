DEBUG = target/wasm/debug/build
RELEASE = target/wasm/release/build
WIT = wit
WAT = $(DEBUG)/gen/gen.wat
WASM = $(DEBUG)/gen/gen.wasm

.PHONY: default $(WAT) $(WASM) clean run

default: main

$(WASM):
	moon build --target wasm -g

$(WAT):
	moon build --target wasm -g --output-wat

$(DEBUG)/gen/gen.embedded.wasm: $(WIT) $(WASM)
	wasm-tools component embed $(WIT) $(WASM) -o $@ --encoding utf16

$(DEBUG)/gen/gen.component.wasm: $(DEBUG)/gen/gen.embedded.wasm
	wasm-tools component new $< -o $@

$(DEBUG)/gen/gen.component.wat: $(DEBUG)/gen/gen.component.wasm
	wasm-tools print $< > $@

main: $(DEBUG)/gen/gen.component.wasm
	node_modules/.bin/jco transpile $< -o src/ --name mnist

clean:
	moon clean

serve: main
	pnpm exec serve -s src/
